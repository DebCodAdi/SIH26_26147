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

    # IMPORTANT — what is and is not knowable here.
    #
    # The only thing measurable from the samples is alpha_norm = Rs/Fs, a *dimensionless
    # ratio*. Absolute Fs is therefore NOT recoverable from sample values: a 200 kHz capture
    # of a 50 kBaud signal and a 400 kHz capture of a 100 kBaud signal produce a bit-identical
    # sample array. Sample rate is metadata, and the trustworthy sources for it are the WAV
    # RIFF header, a SigMF sidecar, or the operator (--fs). This function is an explicitly
    # prior-driven best guess for headerless raw captures, and always reports is_estimated=True.
    #
    # The previous formulation searched Fs and baud independently and scored
    # |alpha_norm*Fs - baud|, which is exactly zero for every (Fs, baud) pair sharing the
    # measured ratio. With a strict < comparison the winner was then decided by grid ordering
    # rather than by the signal, which produced arbitrary answers (measured: 200k->400k,
    # 100k->400k, 400k->4800).
    #
    # This formulation is well posed: each standard baud implies exactly one Fs, which is then
    # snapped to the nearest standard capture rate and scored on snap error and SPS
    # plausibility. Genuinely tied ratios are broken by an explicit popularity prior instead of
    # by list order.
    if alpha_norm <= 1e-9:
        return 200000.0, True

    # Ordered most- to least-common in practice; index becomes a small tie-break penalty.
    fs_popularity = [
        200000.0, 250000.0, 1000000.0, 2000000.0, 100000.0, 48000.0, 96000.0, 192000.0,
        500000.0, 400000.0, 125000.0, 300000.0, 150000.0, 64000.0, 32000.0, 24000.0,
        20000.0, 16000.0, 12000.0, 9600.0, 8000.0, 4800.0, 2400.0,
    ]
    popularity_rank = {f: i for i, f in enumerate(fs_popularity)}

    best_fs = 200000.0
    best_score = float('inf')
    for b in standard_bauds:
        implied_fs = b / alpha_norm
        cand_fs = min(sdr_standard_fs_grid, key=lambda f: abs(f - implied_fs))
        snap_err = abs(cand_fs - implied_fs) / implied_fs
        sps = cand_fs / b
        if sps < 1.0 or sps > 64.0:
            continue
        sps_round_err = abs(sps - round(sps))
        rank_penalty = 0.02 * popularity_rank.get(cand_fs, len(fs_popularity))
        score = snap_err * 3.0 + sps_round_err * 1.5 + rank_penalty
        if score < best_score:
            best_score = score
            best_fs = cand_fs

    return float(best_fs), True

def _whitened_line_spectrum(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Full-resolution whitened spectrum of a real cyclostationary feature signal.

    Resolution matters more than segment averaging here: captures are often only a few hundred
    symbols long, so splitting them into Welch segments costs more in frequency resolution than
    it buys in variance (measured: ~35 kHz symbol-rate error with 256-sample segments versus
    ~70 Hz with the full-length transform). A running-median whitening keeps a genuine narrow
    cyclic line prominent against whatever shape the broadband floor has.
    """
    sig = sig - np.mean(sig)
    if len(sig) < 32:
        return np.zeros(0), np.zeros(0)
    mag = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    med_len = min(31, (len(mag) // 8) * 2 + 1)
    if med_len >= 3:
        floor = scipy.signal.medfilt(mag, med_len) + 1e-12
        mag = mag / floor
    return freqs, mag


def estimate_baud_rate(x_bb: np.ndarray, fs: float) -> tuple[float, float]:
    """
    Blind symbol-rate estimation from cyclostationary lines, by cross-path consensus.

    Three complementary feature signals each expose a symbol-rate line:
      * squared envelope |x|^2                 -> linear modulations (PSK/QAM)
      * instantaneous frequency                -> FSK-family modulations
      * delay-and-multiply Re{x[n]x*[n-1]}     -> general cyclostationary line

    Taking the single strongest peak from whichever path happened to score highest is fragile:
    at 20 dB SNR that produced symbol-rate estimates scattered from 5 kHz to 92 kHz against a
    true 50 kHz. Independent feature paths agreeing on a frequency is far stronger evidence
    than one path's argmax, so candidates that two paths corroborate are preferred, and the
    peak is parabolically interpolated for sub-bin resolution.

    Returns (baud_rate_hz, confidence) where confidence is the whitened peak height.
    """
    n = len(x_bb)
    if n < 64:
        return fs / 4.0, 0.0

    env_sq = np.abs(x_bb) ** 2
    if n > 2:
        dm = x_bb[1:] * np.conj(x_bb[:-1])
        inst_freq = np.angle(dm)
        delay_mult = np.real(dm)
    else:
        inst_freq = np.zeros(1)
        delay_mult = np.zeros(1)

    candidates: list[tuple[float, float]] = []  # (frequency, confidence)
    for feature in (env_sq, inst_freq, delay_mult):
        freqs, mag = _whitened_line_spectrum(feature, fs)
        if len(freqs) == 0:
            continue
        band = (freqs > fs * 0.02) & (freqs < fs * 0.48)
        if not np.any(band):
            continue
        f_band, m_band = freqs[band], mag[band]
        k = int(np.argmax(m_band))
        f_peak = float(f_band[k])
        if 0 < k < len(m_band) - 1:
            a, b, c = float(m_band[k - 1]), float(m_band[k]), float(m_band[k + 1])
            denom = a - 2.0 * b + c
            if abs(denom) > 1e-12:
                delta = 0.5 * (a - c) / denom
                if abs(delta) <= 1.0 and len(f_band) > 1:
                    f_peak += delta * float(f_band[1] - f_band[0])
        candidates.append((f_peak, float(m_band[k])))

    if not candidates:
        return fs / 4.0, 0.0

    # Coarse but noise-robust prior: for a Nyquist-shaped linear modulation the occupied
    # bandwidth is approximately the symbol rate (times 1+rolloff). Unlike a spectral-line
    # argmax this integrates energy across many bins, so it degrades gracefully with SNR and
    # can be used to reject line candidates that are wildly implausible.
    bw_estimate = estimate_occupied_bandwidth(x_bb, fs)
    # Once the occupied bandwidth approaches the full capture bandwidth the measurement is
    # noise-dominated rather than signal-derived, so it is only trusted below that point.
    # (Measured: at 10 dB it reports 150 kHz for a 50 kHz symbol rate on a 200 kHz capture.)
    if not (0.0 < bw_estimate < fs * 0.5):
        bw_estimate = 0.0

    tol = max(fs * 0.01, 1.0)
    best_rs, best_score = candidates[0][0], -1.0
    for i, (f_i, c_i) in enumerate(candidates):
        agree = sum(1 for j, (f_j, _) in enumerate(candidates) if j != i and abs(f_j - f_i) < tol)
        score = c_i * (1.0 + 2.0 * agree)
        if bw_estimate > 0.0:
            # Penalise candidates far from the bandwidth-derived symbol rate. A typical
            # excess bandwidth of 0-35% means the true ratio sits near 1.0-1.35.
            ratio = f_i / bw_estimate
            if 0.7 <= ratio <= 1.45:
                score *= 3.0
            elif ratio < 0.4 or ratio > 2.2:
                score *= 0.2
        if score > best_score:
            best_score, best_rs = score, f_i

    # If no detected line is plausible against a *trusted* bandwidth measurement, prefer the
    # bandwidth-derived symbol rate — it degrades far more gracefully than a line argmax.
    if bw_estimate > 0.0 and not (0.7 <= best_rs / bw_estimate <= 1.45):
        best_rs = bw_estimate

    best_conf = max(c for _, c in candidates)
    return float(best_rs), float(best_conf)


def estimate_occupied_bandwidth(x: np.ndarray, fs: float, energy_fraction: float = 0.90) -> float:
    """
    Occupied bandwidth (Hz) containing `energy_fraction` of the above-noise-floor power.

    Estimated from a segment-averaged PSD with the noise floor subtracted, so it stays usable
    at low SNR where narrow-line detection fails. For a linear modulation this approximates
    the symbol rate scaled by (1 + excess bandwidth).
    """
    if len(x) < 64:
        return 0.0
    nperseg = int(min(len(x), max(64, 2 ** int(np.floor(np.log2(max(64, len(x) // 4)))))))
    freqs, psd = scipy.signal.welch(
        x, fs, nperseg=nperseg, noverlap=nperseg // 2,
        return_onesided=False, detrend=False
    )
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    noise_floor = float(np.percentile(psd, 20))
    psd_net = np.maximum(psd - noise_floor, 0.0)
    total = float(np.sum(psd_net))
    if total <= 0.0:
        return 0.0

    # Symmetric growth around the spectral centroid until the energy fraction is captured.
    centre = int(np.argmax(psd_net))
    acc = float(psd_net[centre])
    lo = hi = centre
    target = total * energy_fraction
    while acc < target and (lo > 0 or hi < len(psd_net) - 1):
        take_low = lo > 0 and (hi >= len(psd_net) - 1 or psd_net[lo - 1] >= psd_net[hi + 1])
        if take_low:
            lo -= 1
            acc += float(psd_net[lo])
        else:
            hi += 1
            acc += float(psd_net[hi])
    return float(abs(freqs[hi] - freqs[lo]))


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

    # Select best CFO estimate by comparing non-linear carrier lines.
    # Cross-validate against the 3-way consensus median first: a rectangular/wide-sidelobe
    # pulse shape can put a spurious spectral line at a +-symbol-rate offset from the true
    # M-th power carrier tone, which can occasionally out-score the true line in a single
    # candidate while the other two (different M, different sidelobe sensitivity) still land
    # near the truth. Down-weight any candidate that disagrees with the consensus median by
    # more than a modest fraction of Fs before ranking by raw SNR score.
    med_f = float(np.median([f_est2, f_est4, f_est8]))
    consensus_tol = max(fs * 0.01, 5.0)
    raw_scores = [snr2 / 1.5, snr4 / 1.0, snr8 / 0.8]
    f_ests = [f_est2, f_est4, f_est8]
    scores = [
        (raw_scores[i] if abs(f_ests[i] - med_f) < consensus_tol else raw_scores[i] * 0.1, f_ests[i])
        for i in range(3)
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

    # 5. Baud Rate Extraction via segment-averaged cyclostationary line detection.
    rs, best_prom = estimate_baud_rate(x_bb, fs)

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
