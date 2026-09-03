"""PS 26147 Universal Blind SDR Interceptor - PyQt6 Interactive Dashboard."""
import sys
import os
import json
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QDoubleSpinBox, QTextEdit,
    QGroupBox, QGridLayout, QProgressBar, QComboBox, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg

from src.pipeline import run_full_pipeline

class DspWorkerThread(QThread):
    """Background DSP worker thread executing run_full_pipeline on real file captures."""
    result_ready = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, filepath: str, user_fs: float | None = None):
        super().__init__()
        self.filepath = filepath
        self.user_fs = user_fs

    def run(self):
        try:
            self.status_update.emit(f"Ingesting & analyzing: {os.path.basename(self.filepath)}...")
            res = run_full_pipeline(self.filepath, user_fs=self.user_fs)
            self.result_ready.emit(res)
        except Exception as e:
            self.error_signal.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PS 26147 - Universal Blind SDR Signal Interceptor & Multi-Hypothesis Demodulator")
        self.resize(1360, 920)
        self.worker = None
        self.ground_truth = {}
        self.load_ground_truth()

        # Main Widget & Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # 1. Header Toolbar & Dataset Quick Navigator
        header_group = QGroupBox("Signal Capture Input & Dataset Explorer")
        header_vbox = QVBoxLayout(header_group)

        # Row 1: File Dialog, Blind Fs Toggle & Manual Fs
        row1_layout = QHBoxLayout()
        self.btn_load = QPushButton("📁 Browse File (.iq, .wav, .raw, .cf32)")
        self.btn_load.clicked.connect(self.select_file)
        row1_layout.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file selected")
        self.lbl_file.setStyleSheet("font-weight: bold; color: #4A90E2;")
        row1_layout.addWidget(self.lbl_file, stretch=1)

        self.chk_auto_fs = QCheckBox("✨ Auto / Blind Fs")
        self.chk_auto_fs.setChecked(True)
        self.chk_auto_fs.toggled.connect(self.on_auto_fs_toggled)
        row1_layout.addWidget(self.chk_auto_fs)

        self.lbl_fs_label = QLabel("Fs (Hz):")
        row1_layout.addWidget(self.lbl_fs_label)
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(1e3, 100e6)
        self.spin_fs.setValue(200000.0)
        self.spin_fs.setSingleStep(10000.0)
        self.spin_fs.setEnabled(False)  # Auto-fs is active by default
        row1_layout.addWidget(self.spin_fs)

        self.btn_run = QPushButton("⚡ Run Interceptor Pipeline")
        self.btn_run.setStyleSheet("background-color: #2ECC71; color: white; font-weight: bold; padding: 6px 16px; font-size: 13px;")
        self.btn_run.clicked.connect(self.start_processing)
        row1_layout.addWidget(self.btn_run)
        header_vbox.addLayout(row1_layout)

        # Row 2: Dataset Quick Selector (100 .iq + 100 .wav files)
        row2_layout = QHBoxLayout()
        row2_layout.addWidget(QLabel("Dataset Quick Explorer:"))
        self.combo_dataset = QComboBox()
        self.populate_dataset_combo()
        self.combo_dataset.currentIndexChanged.connect(self.on_dataset_selection_changed)
        row2_layout.addWidget(self.combo_dataset, stretch=1)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_prev.clicked.connect(self.select_prev_file)
        row2_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next ▶")
        self.btn_next.clicked.connect(self.select_next_file)
        row2_layout.addWidget(self.btn_next)
        header_vbox.addLayout(row2_layout)

        main_layout.addWidget(header_group)

        # 2. Visualizations Area (Spectrum & Constellation)
        viz_layout = QHBoxLayout()

        # Spectrum Plot
        self.spectrum_plot = pg.PlotWidget(title="Baseband Power Spectral Density (PSD)")
        self.spectrum_plot.setLabel('left', 'Power', units='dB')
        self.spectrum_plot.setLabel('bottom', 'Frequency', units='kHz')
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen(color='#00D2FF', width=1.5))
        viz_layout.addWidget(self.spectrum_plot, stretch=1)

        # Constellation Scatter Plot
        self.const_plot = pg.PlotWidget(title="Synchronized IQ Constellation Diagram")
        self.const_plot.setLabel('left', 'Q (Quadrature)')
        self.const_plot.setLabel('bottom', 'I (In-Phase)')
        self.const_plot.showGrid(x=True, y=True, alpha=0.3)
        self.const_plot.setAspectLocked(True)
        self.const_scatter = pg.ScatterPlotItem(size=5, pen=None, brush=pg.mkBrush(255, 204, 0, 160))
        self.const_plot.addItem(self.const_scatter)
        viz_layout.addWidget(self.const_plot, stretch=1)

        # Time-Frequency Waterfall (spectrogram)
        self.waterfall_plot = pg.PlotWidget(title="Time-Frequency Waterfall (Spectrogram)")
        self.waterfall_plot.setLabel('left', 'Time', units='ms')
        self.waterfall_plot.setLabel('bottom', 'Frequency', units='kHz')
        self.waterfall_plot.showGrid(x=True, y=True, alpha=0.2)
        self.waterfall_img = pg.ImageItem()
        self.waterfall_plot.addItem(self.waterfall_img)
        # Perceptually-ordered colour map so power reads correctly in a projected demo.
        _wf_stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        _wf_colors = np.array([
            [0, 0, 40, 255], [20, 20, 140, 255], [0, 170, 170, 255],
            [255, 200, 40, 255], [255, 255, 220, 255]
        ], dtype=np.ubyte)
        self.waterfall_img.setLookupTable(pg.ColorMap(_wf_stops, _wf_colors).getLookupTable(0.0, 1.0, 256))
        viz_layout.addWidget(self.waterfall_plot, stretch=1)

        main_layout.addLayout(viz_layout, stretch=4)

        # 3. Comprehensive Telemetry HUD & Status Badges
        hud_group = QGroupBox("Signal Classification & Demodulation Telemetry HUD")
        hud_layout = QGridLayout(hud_group)

        self.lbl_status = QLabel("IDLE")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #888888;")
        hud_layout.addWidget(QLabel("Pipeline Status:"), 0, 0)
        hud_layout.addWidget(self.lbl_status, 0, 1)

        self.lbl_mod = QLabel("---")
        hud_layout.addWidget(QLabel("Detected Modulation:"), 0, 2)
        hud_layout.addWidget(self.lbl_mod, 0, 3)

        self.lbl_fs_hud = QLabel("---")
        hud_layout.addWidget(QLabel("Sample Rate (Fs):"), 1, 0)
        hud_layout.addWidget(self.lbl_fs_hud, 1, 1)

        self.lbl_cfo = QLabel("---")
        hud_layout.addWidget(QLabel("CFO Estimate:"), 1, 2)
        hud_layout.addWidget(self.lbl_cfo, 1, 3)

        self.lbl_baud = QLabel("---")
        hud_layout.addWidget(QLabel("Baud Rate (SPS):"), 2, 0)
        hud_layout.addWidget(self.lbl_baud, 2, 1)

        self.lbl_evm = QLabel("---")
        hud_layout.addWidget(QLabel("EVM & SNR:"), 2, 2)
        hud_layout.addWidget(self.lbl_evm, 2, 3)

        self.lbl_fec = QLabel("---")
        hud_layout.addWidget(QLabel("FEC & Interleaver:"), 3, 0)
        hud_layout.addWidget(self.lbl_fec, 3, 1)

        self.lbl_entropy = QLabel("---")
        hud_layout.addWidget(QLabel("Shannon Entropy:"), 3, 2)
        hud_layout.addWidget(self.lbl_entropy, 3, 3)

        main_layout.addWidget(hud_group, stretch=2)

        # 4. Decoded Payload Terminal
        payload_group = QGroupBox("Decoded Payload Content & CRC-32 Verification")
        payload_layout = QVBoxLayout(payload_group)
        self.txt_payload = QTextEdit()
        self.txt_payload.setReadOnly(True)
        self.txt_payload.setFont(QFont("Courier New", 10))
        self.txt_payload.setStyleSheet("background-color: #1E1E1E; color: #00FF66;")
        payload_layout.addWidget(self.txt_payload)

        main_layout.addWidget(payload_group, stretch=3)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.selected_file = None

        # Auto-select first dataset file if available
        if self.combo_dataset.count() > 0:
            self.combo_dataset.setCurrentIndex(0)

    def on_auto_fs_toggled(self, checked: bool):
        self.spin_fs.setEnabled(not checked)

    def load_ground_truth(self):
        gt_path = os.path.join("dataset", "ground_truth.json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, "r") as f:
                    self.ground_truth = json.load(f)
            except Exception:
                pass

    def populate_dataset_combo(self):
        self.combo_dataset.clear()
        iq_dir = os.path.join("dataset", "iq")
        wav_dir = os.path.join("dataset", "wav")

        if os.path.exists(iq_dir):
            for f in sorted(os.listdir(iq_dir)):
                if f.endswith(".iq"):
                    gt = self.ground_truth.get(f, {})
                    label = f"IQ: {f} (Mod={gt.get('modulation', '?')}, FEC={gt.get('fec_type', '?')})"
                    self.combo_dataset.addItem(label, os.path.join(iq_dir, f))

        if os.path.exists(wav_dir):
            for f in sorted(os.listdir(wav_dir)):
                if f.endswith(".wav"):
                    gt = self.ground_truth.get(f, {})
                    label = f"WAV: {f} (Mod={gt.get('modulation', '?')}, FEC={gt.get('fec_type', '?')})"
                    self.combo_dataset.addItem(label, os.path.join(wav_dir, f))

    def on_dataset_selection_changed(self, index: int):
        if index >= 0:
            fpath = self.combo_dataset.itemData(index)
            if fpath and os.path.exists(fpath):
                self.selected_file = fpath
                fname = os.path.basename(fpath)
                self.lbl_file.setText(fname)
                gt = self.ground_truth.get(fname, {})
                if "sample_rate" in gt:
                    self.spin_fs.setValue(float(gt["sample_rate"]))

    def select_prev_file(self):
        idx = self.combo_dataset.currentIndex()
        if idx > 0:
            self.combo_dataset.setCurrentIndex(idx - 1)

    def select_next_file(self):
        idx = self.combo_dataset.currentIndex()
        if idx < self.combo_dataset.count() - 1:
            self.combo_dataset.setCurrentIndex(idx + 1)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open SDR Capture File", "",
            "SDR Files (*.iq *.raw *.cf32 *.cs16 *.wav *.bin);;All Files (*)"
        )
        if file_path:
            self.selected_file = file_path
            self.lbl_file.setText(os.path.basename(file_path))

    def start_processing(self):
        if not self.selected_file or not os.path.exists(self.selected_file):
            self.lbl_status.setText("ERROR: Select a valid capture file first.")
            self.lbl_status.setStyleSheet("color: #E74C3C; font-weight: bold;")
            return

        self.btn_run.setEnabled(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.lbl_status.setText("PROCESSING...")
        self.lbl_status.setStyleSheet("color: #F39C12; font-weight: bold;")

        user_fs = None if self.chk_auto_fs.isChecked() else self.spin_fs.value()
        self.worker = DspWorkerThread(self.selected_file, user_fs=user_fs)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_result_ready(self, res: dict):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.btn_run.setEnabled(True)

        # Status badge
        status = res.get("status", "UNKNOWN")
        crc_valid = res.get("crc_valid", False)
        if crc_valid:
            self.lbl_status.setText(f"SUCCESS (CRC VALID) - [{status}]")
            self.lbl_status.setStyleSheet("color: #2ECC71; font-weight: bold; font-size: 14px;")
        elif status == "DEMOD_NO_CRC":
            self.lbl_status.setText("DEMODULATED (NO CRC MATCH)")
            self.lbl_status.setStyleSheet("color: #E67E22; font-weight: bold; font-size: 14px;")
        else:
            self.lbl_status.setText(f"{status}")
            self.lbl_status.setStyleSheet("color: #E74C3C; font-weight: bold; font-size: 14px;")

        # HUD values
        fs_hz = res.get("fs_hz", 0.0)
        fs_est = res.get("fs_estimated", True)
        self.lbl_fs_hud.setText(f"{fs_hz/1000.0:.1f} kHz ({'Blind' if fs_est else 'Manual'})")

        self.lbl_mod.setText(f"{res.get('modulation', '---')}")
        self.lbl_cfo.setText(f"{res.get('cfo_hz', 0.0):.2f} Hz")
        sps_val = res.get('sps', res.get('estimated_sps', 0.0))
        self.lbl_baud.setText(f"{res.get('baud_rate', 0.0):.1f} Baud (SPS={sps_val:.2f})")
        
        evm_val = res.get('evm_pct', 0.0)
        snr_val = res.get('snr_est_db', 0.0)
        self.lbl_evm.setText(f"EVM: {evm_val:.1f}% | SNR: {snr_val:.1f} dB")

        self.lbl_fec.setText(f"{res.get('fec_type', 'NONE')} | Deint: {res.get('interleaver', 'NONE')}")
        ent = res.get("entropy", 0.0)
        ent_cls = res.get("payload_class", res.get("entropy_classification", "UNKNOWN"))
        self.lbl_entropy.setText(f"{ent:.3f} bits/byte ({ent_cls})")

        # Constellation visualization
        payload_syms = res.get("payload_syms", res.get("constellation_symbols", np.array([])))
        if payload_syms is not None and len(payload_syms) > 0:
            sub_syms = payload_syms[:2000]
            self.const_scatter.setData(x=np.real(sub_syms).astype(float), y=np.imag(sub_syms).astype(float))
            self.const_plot.enableAutoRange()

        # Spectrum PSD visualization
        spec_freqs = res.get("spec_freqs", res.get("spectrum_freqs"))
        spec_psd = res.get("spec_psd", res.get("spectrum_psd"))
        if spec_freqs is not None and spec_psd is not None and len(spec_freqs) > 0:
            psd_db = 10.0 * np.log10(spec_psd + 1e-12)
            self.spectrum_curve.setData(x=(spec_freqs / 1000.0).astype(float), y=psd_db.astype(float))
            self.spectrum_plot.enableAutoRange()

        # Time-frequency waterfall visualization
        self.update_waterfall(res.get("waterfall_iq"), float(res.get("fs_hz", 0.0)))

        # Decoded payload terminal
        payload_bytes = res.get("payload", b"")
        if isinstance(payload_bytes, bytes):
            payload_ascii = ''.join([chr(b) if 32 <= b <= 126 or b in (10, 13) else '.' for b in payload_bytes])
            payload_hex = payload_bytes.hex()
        elif isinstance(payload_bytes, str):
            payload_ascii = payload_bytes
            payload_hex = payload_bytes.encode('utf-8', errors='ignore').hex()
        else:
            payload_ascii = res.get("payload_ascii", "")
            payload_hex = res.get("payload_hex", "")
        branch = res.get("branch_name", "NONE")

        # Ground truth comparison if available
        fname = os.path.basename(self.selected_file)
        gt = self.ground_truth.get(fname, {})
        gt_info = ""
        if gt:
            gt_mod = gt.get('modulation', 'N/A')
            est_mod = res.get('modulation', 'N/A')
            mod_match = "MATCH" if str(gt_mod).upper() == str(est_mod).upper() else "MISMATCH"

            gt_fec = str(gt.get('fec_type', 'N/A'))
            est_fec = str(res.get('fec_type', 'N/A'))
            fec_match = "MATCH" if (gt_fec.upper() in est_fec.upper() or est_fec.upper() in gt_fec.upper()) else "MISMATCH"

            gt_cfo = float(gt.get('cfo_hz', 0.0))
            est_cfo = float(res.get('cfo_hz', 0.0))
            cfo_diff = abs(gt_cfo - est_cfo)
            cfo_match = f"OK (Diff: {cfo_diff:.1f} Hz)" if cfo_diff < 50.0 else f"Diff: {cfo_diff:.1f} Hz"

            gt_payload = gt.get('payload_ascii') or gt.get('payload_hex') or ""

            gt_info = f"""
======================================================================
  GROUND TRUTH BENCHMARK VERIFICATION
======================================================================
  Target Modulation:   {gt_mod:<15} -> Model Output: {est_mod:<15} [{mod_match}]
  Target FEC Type:     {gt_fec:<15} -> Model Output: {est_fec:<15} [{fec_match}]
  Target Carrier CFO:  {gt_cfo:.1f} Hz        -> Model Output: {est_cfo:.1f} Hz [{cfo_match}]
  Ground Truth Payload: {gt_payload[:60]}...
======================================================================
"""

        terminal_output = f"""{gt_info}
[PIPELINE SUMMARY]
Status:             {status}
CRC-32 Valid:       {crc_valid}
Winning Hypothesis: {branch}
Sample Rate (Fs):   {fs_hz:.1f} Hz ({'Blindly Determined' if fs_est else 'User-Specified'})
Detected Mod:       {res.get('modulation', 'UNKNOWN')}
Carrier CFO:        {res.get('cfo_hz', 0.0):.3f} Hz
Symbol Rate:        {res.get('baud_rate', 0.0):.1f} Baud (Estimated SPS = {sps_val:.2f})
EVM:                {evm_val:.1f}% | SNR: {snr_val:.1f} dB
FEC Code:           {res.get('fec_type', 'NONE')}
Interleaver:        {res.get('interleaver', 'NONE')}
Bit Slip:           {res.get('bit_slip', 0)}
Shannon Entropy:    {ent:.4f} bits/byte ({ent_cls})

----------------------------------------------------------------------
[DECODED PAYLOAD ASCII]
{payload_ascii}

----------------------------------------------------------------------
[DECODED PAYLOAD HEX]
{payload_hex}
"""
        self.txt_payload.setText(terminal_output.strip())

    def update_waterfall(self, iq: np.ndarray | None, fs_hz: float):
        """
        Renders the time-frequency waterfall (spectrogram) of the captured baseband.

        Works for both complex captures (.iq/.cf32/.raw, two-sided spectrum) and real-valued
        .wav audio promoted to an analytic signal during ingestion, since a complex input
        simply yields a two-sided spectrogram from the same call.
        """
        if iq is None or len(iq) < 64 or fs_hz <= 0:
            return
        try:
            import scipy.signal
            iq = np.asarray(iq)
            nperseg = int(min(256, max(32, 2 ** int(np.floor(np.log2(max(32, len(iq) // 64)))))))
            freqs, times, sxx = scipy.signal.spectrogram(
                iq, fs=fs_hz, nperseg=nperseg, noverlap=nperseg // 2,
                return_onesided=False, mode='psd', detrend=False
            )
            # fftshift so negative frequencies plot to the left of DC.
            order = np.argsort(freqs)
            freqs = freqs[order]
            sxx = sxx[order, :]

            sxx_db = 10.0 * np.log10(np.abs(sxx) + 1e-12)
            # Clip to a fixed dynamic range above the noise floor so the signal stays visible
            # regardless of absolute capture gain.
            floor_db = float(np.percentile(sxx_db, 5))
            peak_db = float(np.max(sxx_db))
            if peak_db - floor_db < 1.0:
                peak_db = floor_db + 1.0

            # ImageItem indexes [x, y]; x = frequency, y = time.
            self.waterfall_img.setImage(sxx_db.T, autoLevels=False, levels=(floor_db, peak_db))
            f_khz = freqs / 1000.0
            t_ms = times * 1000.0
            self.waterfall_img.setRect(pg.QtCore.QRectF(
                float(f_khz[0]), float(t_ms[0]),
                float(f_khz[-1] - f_khz[0]), float(max(t_ms[-1] - t_ms[0], 1e-6))
            ))
            self.waterfall_plot.setXRange(float(f_khz[0]), float(f_khz[-1]), padding=0.02)
            self.waterfall_plot.setYRange(float(t_ms[0]), float(t_ms[-1]), padding=0.02)
        except Exception as e:
            self.txt_payload.append(f"[Waterfall] Render failed: {e}")

    def on_error(self, err_msg: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.btn_run.setEnabled(True)
        self.lbl_status.setText(f"ERROR: {err_msg}")
        self.lbl_status.setStyleSheet("color: #E74C3C; font-weight: bold; font-size: 14px;")

def launch_gui():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_gui()
