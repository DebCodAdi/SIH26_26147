"""Phase 2: Blind Spectral Estimation, Wideband Non-Linear CFO Wipeoff & Polyphase Rational Resampling."""
import math
from fractions import Fraction
import numpy as np
import scipy.signal

def estimate_blind_fs(x: np.ndarray, default_fs: float | None = None) -> tuple[float, bool]:
    """
    Blindly determines the hardware capture sampling frequency Fs (Hz) from discrete samples x[n].
    Uses normalized cyclostationary baud extraction, occupied bandwidth roll-off,
    and SDR hardware sampling grid optimization.
    
    Returns:
        (fs_hz: float, is_blindly_estimated: bool)
    """
    if default_fs is not None and default_fs > 10.0:
        return float(default_fs), False

    if len(x) < 256:
        return 200000.0, True

    n_len = min(len(x), 32768)
    x_sub = x[:n_len]
    
    # 1. Cyclostationary Prominence Peak on Power Envelope
    pwr_env = np.abs(x_sub)**2
    pwr_env = pwr_env - np.mean(pwr_env)
    
    n_fft = min(65536, max(4096, 2**int(np.ceil(np.log2(len(pwr_env))))))
    fft_env = np.abs(np.fft.rfft(pwr_env * np.hanning(len(pwr_env)), n=n_fft))
    freqs_norm = np.fft.rfftfreq(n_fft, d=1.0)
    
    valid_mask = (freqs_norm >= 0.04) & (freqs_norm <= 0.48)
    peaks, props = scipy.signal.find_peaks(fft_env[valid_mask], prominence=np.std(fft_env[valid_mask])*0.8)
    
    if len(peaks) > 0:
        best_p = peaks[np.argmax(props['prominences'])]
        alpha_norm = float(freqs_norm[valid_mask][best_p])
    else:
        alpha_norm = 0.20

    # 2. Map normalized symbol rate alpha_norm to standard SDR sampling grid
    sdr_standard_fs_grid = [
        2400.0, 4800.0, 8000.0, 9600.0, 12000.0, 16000.0, 20000.0, 24000.0, 30000.0,
        32000.0, 36000.0, 40000.0, 48000.0, 60000.0, 64000.0, 80000.0, 96000.0,
        100000.0, 125000.0, 150000.0, 200000.0, 250000.0, 300000.0, 400000.0,
        500000.0, 1000000.0, 2000000.0
    ]
    standard_bauds = [
        600.0, 1200.0, 2400.0, 4800.0, 9600.0, 12000.0, 16000.0, 19200.0, 20000.0,
        24000.0, 25000.0, 32000.0, 36000.0, 38400.0, 40000.0, 48000.0, 50000.0,
        64000.0, 96000.0, 100000.0, 125000.0, 250000.0
    ]

    best_fs = 200000.0
    best_score = float('inf')

    for cand_fs in sdr_standard_fs_grid:
        cand_rs = alpha_norm * cand_fs
        for b in standard_bauds:
            baud_err = abs(cand_rs - b) / b
            sps = cand_fs / b
            sps_round_err = abs(sps - round(sps))
            prio = 0.5 if b in (50000.0, 20000.0, 4800.0, 9600.0, 40000.0, 48000.0, 96000.0) else 1.0
            score = (baud_err * 3.0 + sps_round_err * 1.5) * prio
            if score < best_score:
                best_score = score
                best_fs = cand_fs

    return float(best_fs), True

def estimate_cfo_and_baud(x: np.ndarray, fs: float) -> tuple[float, float, float, np.ndarray]:
    """
    Estimates coarse CFO across full Nyquist band (|f| <= 0.45 Fs) using M-th power non-linear
    carrier recovery & Welch PSD, with parabolic peak interpolation.
    Extracts multi-path baud rate (Rs, SPS) via envelope-squared, instantaneous frequency differential,
    and delay-and-multiply cyclostationary lines. Wipes carrier offset.
    """
    n_len = min(len(x), 32768)
    x_sub = x[:n_len]
    
    # Adaptive FFT resolution
    n_fft = min(65536, max(4096, 2**int(np.ceil(np.log2(len(x_sub))))))
    w = np.hanning(len(x_sub))
    df = fs / n_fft

    # Widen search to full Nyquist baseband (|f_cfo| <= 0.45 * fs)
    max_cfo = fs * 0.45

    # 1. x^4 line for QPSK / 16-QAM
    x4 = (x_sub ** 4) * w
    fft_x4 = np.abs(np.fft.fft(x4, n_fft))
    f_bins4 = np.fft.fftfreq(n_fft, d=1.0/fs)
    mask4 = np.abs(f_bins4) <= (4.0 * max_cfo)
    fft_x4_masked = fft_x4.copy()
    fft_x4_masked[~mask4] = 0.0
    idx4 = int(np.argmax(fft_x4_masked))
    a4, b4, c4 = fft_x4[(idx4-1)%n_fft], fft_x4[idx4], fft_x4[(idx4+1)%n_fft]
    denom4 = (2.0 * b4 - a4 - c4)
    delta4 = 0.5 * (c4 - a4) / denom4 if abs(denom4) > 1e-12 else 0.0
    f_est4 = float((f_bins4[idx4] + delta4 * df) / 4.0)
    snr4 = float(fft_x4[idx4] / (np.mean(fft_x4[mask4]) + 1e-12)) if np.any(mask4) else 0.0

    # 2. x^2 line for BPSK / 2-FSK
    x2 = (x_sub ** 2) * w
    fft_x2 = np.abs(np.fft.fft(x2, n_fft))
    f_bins2 = np.fft.fftfreq(n_fft, d=1.0/fs)
    mask2 = np.abs(f_bins2) <= (2.0 * max_cfo)
    fft_x2_masked = fft_x2.copy()
    fft_x2_masked[~mask2] = 0.0
    idx2 = int(np.argmax(fft_x2_masked))
    a2, b2, c2 = fft_x2[(idx2-1)%n_fft], fft_x2[idx2], fft_x2[(idx2+1)%n_fft]
    denom2 = (2.0 * b2 - a2 - c2)
    delta2 = 0.5 * (c2 - a2) / denom2 if abs(denom2) > 1e-12 else 0.0
    f_est2 = float((f_bins2[idx2] + delta2 * df) / 2.0)
    snr2 = float(fft_x2[idx2] / (np.mean(fft_x2[mask2]) + 1e-12)) if np.any(mask2) else 0.0

    # 3. x^8 line for 8-PSK
    x8 = (x_sub ** 8) * w
    fft_x8 = np.abs(np.fft.fft(x8, n_fft))
    f_bins8 = np.fft.fftfreq(n_fft, d=1.0/fs)
    mask8 = np.abs(f_bins8) <= (8.0 * max_cfo)
    fft_x8_masked = fft_x8.copy()
    fft_x8_masked[~mask8] = 0.0
    idx8 = int(np.argmax(fft_x8_masked))
    a8, b8, c8 = fft_x8[(idx8-1)%n_fft], fft_x8[idx8], fft_x8[(idx8+1)%n_fft]
    denom8 = (2.0 * b8 - a8 - c8)
    delta8 = 0.5 * (c8 - a8) / denom8 if abs(denom8) > 1e-12 else 0.0
    f_est8 = float((f_bins8[idx8] + delta8 * df) / 8.0)
    snr8 = float(fft_x8[idx8] / (np.mean(fft_x8[mask8]) + 1e-12)) if np.any(mask8) else 0.0

    # 4. Welch PSD peak for tone / carrier presence (coarse frequency centroid)
    nperseg = min(len(x), 4096)
    freqs, psd = scipy.signal.welch(x, fs, return_onesided=False, nperseg=nperseg)
    freqs_c, psd_c = np.fft.fftshift(freqs), np.fft.fftshift(psd)
    k_peak = int(np.argmax(psd_c))
    f_est_psd = float(freqs_c[k_peak])

    # Disambiguate M-th power estimates by aligning with coarse PSD centroid
    f_est2 = float(f_est2 + round((f_est_psd - f_est2) / (fs / 2.0)) * (fs / 2.0))
    f_est4 = float(f_est4 + round((f_est_psd - f_est4) / (fs / 4.0)) * (fs / 4.0))
    f_est8 = float(f_est8 + round((f_est_psd - f_est8) / (fs / 8.0)) * (fs / 8.0))

    # Select best CFO estimate by comparing non-linear carrier lines
    scores = [
        (snr2 / 1.5, f_est2),
        (snr4 / 1.0, f_est4),
        (snr8 / 0.8, f_est8),
    ]
    scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_f = scores[0]

    if best_score >= 2.5:
        f_cfo = best_f
    elif abs(f_est_psd) > fs * 0.05:
        if 0 < k_peak < len(psd_c) - 1:
            alpha, beta, gamma = float(psd_c[k_peak - 1]), float(psd_c[k_peak]), float(psd_c[k_peak + 1])
            denom = alpha - 2.0 * beta + gamma
            delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-12 else 0.0
            f_cfo = float(freqs_c[k_peak] + delta * (freqs_c[1] - freqs_c[0]))
        else:
            f_cfo = f_est_psd
    else:
        f_cfo = 0.0

    # Wipe coarse CFO
    n = np.arange(len(x))
    x_bb = x * np.exp(-1j * 2.0 * np.pi * f_cfo * n / fs)

    # 5. Multi-Path Baud Rate Extraction via Spectral Prominence
    # Path A: Envelope-Squared (Linear Modulations)
    env_sq = np.abs(x_bb)**2
    env_sq_zero = env_sq - np.mean(env_sq)
    w_env = np.hanning(len(env_sq_zero))
    fft_lin = np.abs(np.fft.rfft(env_sq_zero * w_env))
    f_bins = np.fft.rfftfreq(len(env_sq_zero), d=1.0/fs)

    med_filter_len = min(31, len(fft_lin) // 8 * 2 + 1)
    if med_filter_len >= 3:
        smooth_lin = scipy.signal.medfilt(fft_lin, med_filter_len) + 1e-6
        prom_lin = fft_lin / smooth_lin
    else:
        prom_lin = fft_lin

    # Path B: Instantaneous Frequency Differential (FSK)
    inst_freq = np.angle(x_bb[1:] * np.conj(x_bb[:-1]))
    inst_freq_zero = inst_freq - np.mean(inst_freq)
    w_fsk = np.hanning(len(inst_freq_zero))
    fft_fsk = np.abs(np.fft.rfft(inst_freq_zero * w_fsk))
    f_bins_fsk = np.fft.rfftfreq(len(inst_freq_zero), d=1.0/fs)

    if med_filter_len >= 3:
        smooth_fsk = scipy.signal.medfilt(fft_fsk, med_filter_len) + 1e-6
        prom_fsk = fft_fsk / smooth_fsk
    else:
        prom_fsk = fft_fsk

    # Path C: Delay-and-Multiply Cyclostationary Line (D = 1)
    if len(x_bb) > 2:
        dm = x_bb[1:] * np.conj(x_bb[:-1])
        dm_zero = dm - np.mean(dm)
        fft_dm = np.abs(np.fft.rfft(np.real(dm_zero) * w_fsk))
        if med_filter_len >= 3:
            smooth_dm = scipy.signal.medfilt(fft_dm, med_filter_len) + 1e-6
            prom_dm = fft_dm / smooth_dm
        else:
            prom_dm = fft_dm
    else:
        prom_dm = prom_lin

    mask = (f_bins > fs * 0.02) & (f_bins < fs * 0.48)
    mask_fsk = (f_bins_fsk > fs * 0.02) & (f_bins_fsk < fs * 0.48)

    max_prom_lin = float(np.max(prom_lin[mask])) if np.any(mask) else 0.0
    max_prom_fsk = float(np.max(prom_fsk[mask_fsk])) if np.any(mask_fsk) else 0.0
    max_prom_dm  = float(np.max(prom_dm[mask_fsk])) if np.any(mask_fsk) else 0.0

    prom_scores = [(max_prom_lin, 'LIN'), (max_prom_fsk, 'FSK'), (max_prom_dm, 'DM')]
    prom_scores.sort(key=lambda item: item[0], reverse=True)
    best_prom, best_path = prom_scores[0]

    if best_path == 'LIN' and np.any(mask):
        rs = float(f_bins[mask][np.argmax(prom_lin[mask])])
    elif best_path == 'FSK' and np.any(mask_fsk):
        rs = float(f_bins_fsk[mask_fsk][np.argmax(prom_fsk[mask_fsk])])
    elif np.any(mask_fsk):
        rs = float(f_bins_fsk[mask_fsk][np.argmax(prom_dm[mask_fsk])])
    else:
        rs = float(fs / 4.0)

    # 6. Check for 1 SPS (unfiltered symbol-spaced captures)
    r1 = np.abs(np.corrcoef(x_bb[1:], x_bb[:-1])[0, 1]) if len(x_bb) > 2 else 0.0
    if r1 < 0.25 or best_prom < 1.5:
        rs = float(fs)
        sps = 1.0
    else:
        rs = max(rs, fs * 0.005)
        sps = float(np.clip(fs / rs, 1.0, 50.0))

    return f_cfo, rs, sps, x_bb.astype(np.complex64)

def resample_to_2sps(x_bb: np.ndarray, sps: float) -> np.ndarray:
    """
    Polyphase rational FIR resampling to exactly 2 SPS without tap explosion.
    Target ratio is 2.0 / sps.
    """
    if abs(sps - 2.0) < 1e-4:
        return x_bb.astype(np.complex64)

    ratio = 2.0 / sps
    frac = Fraction.from_float(ratio).limit_denominator(64)
    up = int(frac.numerator)
    down = int(frac.denominator)

    # Fallback bounds check
    if up > 128 or down > 128:
        up, down = int(round(ratio * 16)), 16

    return scipy.signal.resample_poly(x_bb, up, down).astype(np.complex64)
