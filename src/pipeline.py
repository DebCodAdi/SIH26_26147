"""Universal End-to-End Pipeline Core for Headless CLI, Verification, and Real-Time GUI."""
import os
import numpy as np
from src.ingestion import auto_ingest_and_clean, ingest_wav, detect_signal_mme
from src.spectral import estimate_cfo_and_baud, resample_to_2sps
from src.equalizers import apply_twopass_cma, apply_gardner_ted
from src.classifier import extract_features, evaluate_mahalanobis_ood, rank_modulation_hypotheses, compute_constellation_evm
from src.synchronizer import track_carrier_pll, resolve_sync_and_rotation
from src.demodulators import demodulate_fsk, demap_linear
from src.entropy import calculate_shannon_entropy, classify_payload_entropy
from src.blackboard import BlackboardArbiter
from src.iq_correction import remove_dc_offset_iir, correct_iq_imbalance_gsop, normalize_agc

def estimate_snr_m2m4(x: np.ndarray) -> float:
    M2 = np.mean(np.abs(x)**2)
    M4 = np.mean(np.abs(x)**4)
    k = M4 / (M2**2 + 1e-12)
    if k > 1.95:  # pure noise
        return -10.0
    snr_linear = max(1.0 / (k - 1.0 + 1e-6), 0.01)
    return float(10 * np.log10(snr_linear))

def estimate_snr_kurtosis(x: np.ndarray) -> float:
    """Estimates instantaneous SNR in dB from complex envelope kurtosis."""
    if len(x) < 16:
        return 0.0
    pwr = np.abs(x)**2
    m2 = np.mean(pwr)
    m4 = np.mean(pwr**2)
    ratio = m4 / (m2**2 + 1e-12)
    if ratio >= 1.95:
        return -10.0
    elif ratio <= 1.15:
        return 20.0
    else:
        return float(20.0 - 30.0 * (ratio - 1.0))

def run_full_pipeline(
    input_path: str,
    user_fs: float | None = None,
    meta_path: str | None = None
) -> dict:
    """
    Executes the complete PS 26147 blind SDR interceptor pipeline end-to-end.
    Handles standard signals as well as challenge edge-cases.
    """
    if not os.path.exists(input_path):
        return {"status": "ERROR_FILE_NOT_FOUND", "stage": "INGESTION", "payload": b"", "crc_valid": False}

    ext = os.path.splitext(input_path)[1].lower()
    meta_sidecar = meta_path if meta_path else (os.path.splitext(input_path)[0] + ".meta.json")

    # Stage 1: Ingestion
    nan_detected_count = 0
    if ext == ".wav":
        try:
            fs, x_clean = ingest_wav(input_path)
            fmt = "wav_audio"
            valid = len(x_clean) > 0
        except Exception as e:
            return {"status": f"ERROR_WAV_INGEST: {e}", "stage": "INGESTION", "payload": b"", "crc_valid": False}
    else:
        with open(input_path, "rb") as f:
            raw_bytes = f.read()

        # Check raw NaNs before ingestion
        if len(raw_bytes) % 8 == 0:
            c64_raw = np.frombuffer(raw_bytes, dtype=np.complex64)
            nan_detected_count = int(np.sum(np.isnan(c64_raw)))

        valid, fmt, x_clean, fs = auto_ingest_and_clean(
            raw_bytes,
            user_fs=user_fs,
            meta_path=meta_sidecar if os.path.exists(meta_sidecar) else None,
            require_fs=False
        )

    # Check for NaN/Inf in cleaned signal (ingestion already repairs them)
    nans_remaining = int(np.sum(~np.isfinite(x_clean)))
    if nans_remaining > len(x_clean) * 0.5:
        return {"status": "CORRUPTED_EXCESSIVE_NAN", "stage": "INGESTION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}
        
    if not valid or len(x_clean) < 16:
        return {"status": "INSUFFICIENT_SAMPLES", "stage": "INGESTION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    try:
        x_clean = remove_dc_offset_iir(x_clean)
        x_clean = correct_iq_imbalance_gsop(x_clean)
        x_clean = normalize_agc(x_clean)
    except Exception as e:
        return {"status": f"ERROR_IQ_CORRECTION: {e}", "stage": "INGESTION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    # Signal presence is gated by the Marcenko-Pastur eigenvalue (MME) detector, which is
    # robust down to ~-15 dB SNR and does not assume any particular modulation/constant-modulus
    # model. The older kurtosis-based estimate_snr_m2m4() is systematically biased (it reads
    # single-digit dB even on genuinely 30 dB clean signals, because it evaluates the raw
    # pulse-shaped/unsynchronized waveform rather than a matched-filtered, symbol-synchronized
    # one) and must not be used as a hard reject gate. It is retained below purely as an
    # approximate telemetry value.
    signal_present, mme_stat = detect_signal_mme(x_clean)
    snr_db = estimate_snr_m2m4(x_clean)
    if not signal_present:
        return {
            "status": "NO_SIGNAL_DETECTED",
            "stage": "SPECTRAL_ANALYSIS",
            "format": fmt,
            "fs_hz": float(fs),
            "cfo_hz": 0.0,
            "baud_rate": 0.0,
            "sps": 1.0,
            "modulation": "NOISE",
            "crc_valid": False,
            "payload": b"",
            "entropy": 0.0,
            "payload_class": "NOISE",
            "fec_type": "NONE",
            "interleaver": "NONE",
            "bit_slip": 0,
            "metadata": {"snr_est_db": snr_db, "mme_statistic": mme_stat}
        }

    # Stage 2: Coarse CFO & Clock Recovery
    try:
        cfo_hz, rs, sps, x_bb = estimate_cfo_and_baud(x_clean, fs=fs)
        if rs < 100 or rs > fs / 2:
            # Baud rate unreasonable, try alternative estimation
            rs = 1000.0
            sps = fs / rs
    except Exception as e:
        return {"status": f"ERROR_SPECTRAL: {e}", "stage": "SPECTRAL", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}
        
    # Stage 3: Equalization & Fine Timing
    try:
        if sps <= 1.25 or len(x_clean) < 512:
            pwr = float(np.mean(np.abs(x_bb)**2))
            y_syms = (x_bb / np.sqrt(pwr + 1e-12)).astype(np.complex64)
        else:
            x_2sps = resample_to_2sps(x_bb, sps)
            y_syms = apply_twopass_cma(x_2sps)
            if len(y_syms) == 0:
                y_syms = x_2sps[::2]
    except Exception as e:
        return {"status": f"ERROR_EQUALIZATION: {e}", "stage": "EQUALIZATION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    # Stage 4: Modulation Classification
    try:
        features = extract_features(y_syms)
        mod_type, ood_dist = evaluate_mahalanobis_ood(features, symbols=x_bb)
    except Exception as e:
        return {"status": f"ERROR_CLASSIFICATION: {e}", "stage": "CLASSIFICATION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    if mod_type == "UNKNOWN_MODULATION":
        mod_type = "QPSK"

    # Stage 5: Synchronization & Carrier PLL
    try:
        fsk_bits = None
        payload_raw = np.array([], dtype=np.complex64)
        if mod_type in ["2-FSK", "4-FSK", "GMSK"]:
            # GMSK is a binary continuous-phase FSK derivative (modulation index ~0.5); an FM
            # discriminator is the correct receiver structure for it, not a linear I/Q slicer.
            num_tones = 4 if mod_type == "4-FSK" else 2
            fsk_bits = demodulate_fsk(x_bb, sps=sps, num_tones=num_tones)
            payload_raw, start_idx, sync_metric, phase_rad = resolve_sync_and_rotation(y_syms, mod_type="QPSK")
            s_locked = track_carrier_pll(payload_raw if len(payload_raw) > 0 else y_syms, mod_type="QPSK")
            payload_syms = s_locked
        else:
            payload_raw, start_idx, sync_metric, phase_rad = resolve_sync_and_rotation(y_syms, mod_type=mod_type)
            s_locked = track_carrier_pll(payload_raw if len(payload_raw) > 0 else y_syms, mod_type=mod_type)
            payload_syms = s_locked
    except Exception as e:
        return {"status": f"ERROR_SYNCHRONIZATION: {e}", "stage": "SYNCHRONIZATION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    # Stage 6: Blackboard Multi-Hypothesis Arbitration
    try:
        ranked_candidates = [m[0] for m in rank_modulation_hypotheses(features)]
        arbiter = BlackboardArbiter()
        symbols_eval = payload_syms if (payload_syms is not None and len(payload_syms) > 0) else y_syms
        pre_pll_eval = payload_raw if (payload_raw is not None and len(payload_raw) > 0) else y_syms
        # The un-synchronised symbol stream is supplied as an extra hypothesis: if preamble
        # correlation locked onto a false peak, the synced streams have had part of the payload
        # sliced off and only this copy is still decodable.
        aux_streams = [y_syms] if (start_idx > 0 and len(y_syms) > 0) else []
        results = arbiter.evaluate_stream(
            symbols=symbols_eval,
            mod_type=mod_type,
            fsk_bits=fsk_bits,
            pre_pll_symbols=pre_pll_eval,
            ranked_mods=ranked_candidates,
            aux_symbol_streams=aux_streams
        )
    except Exception as e:
        return {"status": f"ERROR_ARBITER: {e}", "stage": "ARBITER", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    # Spectrum calculation for telemetry/GUI
    n_spec = min(len(x_bb), 512)
    spec_freqs = np.linspace(-fs/2, fs/2, n_spec)
    spec_psd = np.abs(np.fft.fftshift(np.fft.fft(x_bb[:n_spec])))**2

    # Best-effort raw demodulated payload extraction for uncoded / non-CRC signals
    raw_payload = b""
    sym_candidates = [s for s in [payload_syms, s_locked, y_syms] if len(s) > 0]
    best_candidate_bytes = b""
    best_ascii_count = -1
    for sym_src in sym_candidates:
        raw_demod_modes = demap_linear(sym_src, mod_type=mod_type)
        for mode_name, m_bits in raw_demod_modes:
            m_bytes = np.packbits(m_bits, bitorder='big').tobytes()
            ascii_count = sum(1 for b in m_bytes[:64] if 32 <= b <= 126)
            if ascii_count > best_ascii_count:
                best_ascii_count = ascii_count
                best_candidate_bytes = m_bytes
    raw_payload = best_candidate_bytes[:min(len(best_candidate_bytes), 256)]

    # EVM watchdog calculation
    evm_pct = compute_constellation_evm(s_locked if len(s_locked) > 0 else y_syms, mod_type=mod_type)

    # Standard Result Output
    base_result = {
        "status": "DEMOD_NO_CRC",
        "stage": "COMPLETED",
        "format": fmt,
        "fs_hz": float(fs),
        "fs_estimated": bool(user_fs is None or user_fs <= 0),
        "cfo_hz": float(cfo_hz),
        "baud_rate": float(rs),
        "sps": float(sps),
        "modulation": mod_type,
        "ood_dist": float(ood_dist),
        "evm_pct": float(evm_pct),
        "snr_est_db": float(snr_db),
        "sync_metric": float(sync_metric),
        "phase_rad": float(phase_rad),
        "crc_valid": False,
        "payload": raw_payload,
        "entropy": float(calculate_shannon_entropy(raw_payload)) if raw_payload else 0.0,
        "payload_class": classify_payload_entropy(raw_payload) if raw_payload else "UNKNOWN",
        "branch_name": "RAW_DEMOD",
        "fec_type": "NONE",
        "interleaver": "NONE",
        "bit_slip": 0,
        "spectrum_freqs": spec_freqs,
        "spectrum_psd": spec_psd,
        "spec_freqs": spec_freqs,
        "spec_psd": spec_psd,
        # Raw corrected baseband for the GUI time-frequency waterfall. Bounded so a long
        # capture cannot balloon the result dict; the spectrogram needs no more than this.
        "waterfall_iq": x_clean[:262144],
        "constellation_symbols": s_locked[:1000] if len(s_locked) > 0 else (payload_syms[:1000] if len(payload_syms) > 0 else y_syms[:1000]),
        "payload_syms": s_locked[:1000] if len(s_locked) > 0 else (payload_syms[:1000] if len(payload_syms) > 0 else y_syms[:1000])
    }

    if len(results) > 0 and results[0].crc_valid:
        winner = results[0]
        base_result.update({
            "status": "SUCCESS_CRC_LOCKED",
            "crc_valid": True,
            "modulation": winner.mod_type,
            "payload": winner.payload,
            "entropy": float(winner.entropy),
            "payload_class": classify_payload_entropy(winner.payload),
            "branch_name": winner.branch_name,
            "fec_type": winner.fec_type,
            "interleaver": winner.interleaver,
            "bit_slip": winner.bit_slip,
            "metadata": winner.metadata
        })

    p_bytes = base_result.get("payload", b"")
    if isinstance(p_bytes, bytes):
        base_result["payload_ascii"] = ''.join([chr(b) if 32 <= b <= 126 or b in (10, 13) else '.' for b in p_bytes])
        base_result["payload_hex"] = p_bytes.hex()
    else:
        base_result["payload_ascii"] = str(p_bytes)
        base_result["payload_hex"] = ""

    return base_result
