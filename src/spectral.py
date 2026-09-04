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

    Returns (baud_rate_hz, confidence) where confidence in [0, 1] is how many independent
    feature paths corroborated the chosen line (0.0 = none, i.e. do not trust the value).
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
    best_rs, best_score, best_agree = candidates[0][0], -1.0, 0
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
            best_score, best_rs, best_agree = score, f_i, agree

    # If no detected line is plausible against a *trusted* bandwidth measurement, prefer the
    # bandwidth-derived symbol rate — it degrades far more gracefully than a line argmax.
    if bw_estimate > 0.0 and not (0.7 <= best_rs / bw_estimate <= 1.45):
        best_rs = bw_estimate
        best_agree = 0

    # Confidence is CORROBORATION between independent feature paths, not peak height.
    # Peak height is not a reliability measure here: the whitened spectrum always has peaks
    # above its own median, so the old value sat at ~3.1-3.5 whether the answer was right
    # (50 kHz) or garbage (67 kHz at 0 dB). Corroboration is calibrated against correctness --
    # measured median symbol-rate error, true rate 50 kHz, 24 captures per SNR:
    #     SNR    paths agree -> error     no path agrees -> error
    #     30 dB              0 Hz                     39286 Hz
    #     20 dB           6270 Hz                     39662 Hz
    #     15 dB          10583 Hz                     33810 Hz
    #     10 dB          14286 Hz                     33565 Hz
    #      5 dB   (never agrees)                      22287 Hz
    # so 0.0 genuinely means "do not trust this number", which is what a caller needs.
    best_conf = min(1.0, best_agree / 2.0)
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


def estimate_cfo_and_baud(x: np.ndarray, fs: float,
                          diag: dict | None = None) -> tuple[float, float, float, np.ndarray]:
    """
    Estimates the carrier frequency offset by M-th power non-linear carrier recovery applied to
    symbol-rate samples, with the modulation order chosen by phase-symmetry nesting and the
    modulo ambiguity resolved against a coarse PSD-centroid estimate.
    Extracts the baud rate (Rs, SPS) via envelope-squared, instantaneous frequency differential,
    and delay-and-multiply cyclostationary lines. Wipes carrier offset.

    If `diag` is a dict it is populated with per-estimate diagnostics, notably
    'baud_confidence' in [0, 1]: 0.0 means no independent feature path corroborated the
    symbol-rate line, i.e. the returned baud rate must not be trusted. The 4-tuple return is
    unchanged so existing callers keep working.
    """
    # ------------------------------------------------------------------------------------
    # Carrier recovery: M-th power applied to SYMBOL-RATE samples.
    #
    # x**M strips the modulation only if the samples are close to ideal constellation points.
    # Applied to the oversampled, pulse-shaped waveform (as this function previously did) the
    # inter-symbol transitions are not constellation points, so x**M smears energy instead of
    # producing a clean line -- and the smearing compounds with M, which is why x**8 (the only
    # valid order for 8PSK) was worst affected. Measured at 30 dB, 8PSK carrier error:
    # full-rate 2883 Hz, symbol-rate 3.8 Hz.
    #
    # Doing timing before carrier is not circular: the symbol rate is estimated from the power
    # envelope |x|^2, which is invariant to carrier offset, since |x*exp(j*2*pi*f*n/fs)| == |x|.
    # ------------------------------------------------------------------------------------
    nperseg = min(len(x), 4096)
    freqs, psd = scipy.signal.welch(x, fs, return_onesided=False, nperseg=nperseg)
    freqs_c, psd_c = np.fft.fftshift(freqs), np.fft.fftshift(psd)
    _net = np.maximum(psd_c - float(np.percentile(psd_c, 25)), 0.0)
    # Power-weighted centroid: the occupied band of a linearly modulated signal is symmetric
    # about its carrier, so this is a coarse but noise-robust carrier estimate (measured
    # accurate to <=1.3 kHz at 30 dB). It is used only to resolve the modulo ambiguity of the
    # fine estimate below. The PSD argmax, used previously, is merely the loudest bin inside a
    # noisy modulated band and read 18.7 kHz on a signal whose true offset was 150 Hz.
    f_coarse = float(np.sum(freqs_c * _net) / (np.sum(_net) + 1e-20))

    rs_pre, _ = estimate_baud_rate(x, fs)
    f_cfo = f_coarse
    if np.isfinite(rs_pre) and rs_pre > 0:
        decim = max(1, int(round(fs / rs_pre)))
        # Pick the decimation phase carrying the most energy (crude eye-opening choice).
        best_phase, best_energy = 0, -1.0
        for phase in range(decim):
            energy = float(np.sum(np.abs(x[phase::decim]) ** 2))
            if energy > best_energy:
                best_energy, best_phase = energy, phase
        z = x[best_phase::decim]

        if len(z) >= 32:
            fs_dec = fs / decim
            n_fft_d = int(min(65536, max(4096, 2 ** int(np.ceil(np.log2(len(z)))) * 8)))
            w_d = np.hanning(len(z))
            df_dec = fs_dec / n_fft_d
            f_est = {}
            line_significance = {}
            for M in (2, 4, 8):
                fft_m = np.abs(np.fft.fft((z ** M) * w_d, n_fft_d))
                bins_m = np.fft.fftfreq(n_fft_d, d=1.0 / fs_dec)
                idx = int(np.argmax(fft_m))
                a, b, c = fft_m[(idx - 1) % n_fft_d], fft_m[idx], fft_m[(idx + 1) % n_fft_d]
                denom = (2.0 * b - a - c)
                delta = 0.5 * (c - a) / denom if abs(denom) > 1e-12 else 0.0
                f_raw = float((bins_m[idx] + delta * df_dec) / M)
                # The M-th power estimate is ambiguous modulo fs_dec/M; take the
                # representative nearest the coarse centroid.
                ambiguity = fs_dec / M
                f_est[M] = f_raw + round((f_coarse - f_raw) / ambiguity) * ambiguity
                # Peak-to-median WITHIN this order's own spectrum. This is a valid statistic
                # (unlike comparing peak heights ACROSS orders) because it asks only whether
                # this spectrum contains a line at all.
                line_significance[M] = float(fft_m[idx] / (np.median(fft_m) + 1e-12))

            # Order selection by phase-symmetry NESTING, not by line strength. Comparing
            # strength across orders is ill posed: each power transforms the noise
            # differently, so peak/mean (and a robust peak/MAD z-score, also tested) are
            # biased toward low M regardless of whether a coherent line exists -- for 8PSK,
            # whose only valid order is M=8, every strength statistic still ranked M=2 first.
            # The removable symmetries are nested instead:
            #     BPSK   (2-fold): x^2, x^4, x^8 all yield a line
            #     QPSK   (4-fold):      x^4, x^8 yield a line
            #     16-QAM (4-fold):      x^4, x^8 yield a line
            #     8PSK   (8-fold):           only x^8 yields a line
            # so a valid order is corroborated by the next higher order while an invalid
            # order's argmax is noise and agrees with nothing. Pick the LOWEST corroborated
            # order, because x^M multiplies the frequency error by M.
            # Tolerance is a few bins of the decimated resolution; sweeping it over 8/16/32
            # bins left BPSK, QPSK and 8PSK selections unchanged, so it is not knife-edge.
            #
            # When nothing corroborates, the previous code fell through to f_est[8] -- the
            # LEAST reliable candidate, since x^8 amplifies noise the most. That is backwards,
            # and it was the dominant source of carrier-recovery variance at 20 dB: measured
            # over 8 QPSK seeds, f_est[4] was correct to 1.0-14.3 Hz on every one, yet f8 was
            # selected on 7 of 8 because |f4-f8| (139-4255 Hz) exceeded the tolerance. Feeding
            # that wrong offset downstream drove pre-FEC BER to 0.40-1.00; supplying the true
            # offset to the same chain gave BER 0.0000 on 7 of those 8 seeds.
            #
            # So f8 is used as a last resort only when the x^8 spectrum actually contains a
            # line. The bound is derived, not tuned: for pure complex Gaussian noise the |FFT|
            # magnitudes are Rayleigh and the expected ratio of the maximum over N bins to the
            # median is sqrt(ln N / ln 2). Monte-Carlo over 200 noise realisations puts the
            # 95th percentile at 3.44-3.76 for the transform sizes used here, against a
            # predicted 3.46-3.87, so a peak below that is indistinguishable from noise.
            # Measured line significance at M=8: 8.7 for 8PSK at 30 dB (a real line) versus
            # 2.1-2.7 for every modulation at 20 dB (no line for anyone, 8PSK included).
            tol = 8.0 * df_dec
            noise_floor_ratio = float(np.sqrt(np.log(n_fft_d) / np.log(2.0)))
            if abs(f_est[2] - f_est[4]) < tol:
                f_cfo = f_est[2]
            elif abs(f_est[4] - f_est[8]) < tol:
                f_cfo = f_est[4]
            elif line_significance[8] > noise_floor_ratio:
                f_cfo = f_est[8]
            else:
                f_cfo = f_est[4]

    # Wipe coarse CFO
    n = np.arange(len(x))
    x_bb = x * np.exp(-1j * 2.0 * np.pi * f_cfo * n / fs)

    # 5. Baud Rate Extraction via segment-averaged cyclostationary line detection.
    rs, baud_conf = estimate_baud_rate(x_bb, fs)

    # 6. Symbol-spaced (1 SPS) capture detection.
    #
    # This previously read `if r1 < 0.25 or best_prom < 1.5: rs = fs`, where r1 is the
    # adjacent-sample correlation. That asserted a symbol-spaced capture whenever adjacent
    # samples were weakly correlated -- but noise decorrelates adjacent samples too, so a
    # perfectly ordinary 4-samples-per-symbol capture trips it once SNR drops. Measured on a
    # 200 kHz capture of a 50 kBaud signal (true answer 50 kHz, sps 4):
    #     30 dB r1=0.889   20 dB r1=0.751   15 dB r1=0.553
    #     10 dB r1=0.308    5 dB r1=0.140 -> fires -> reports rs = Fs = 200000 (150 kHz error)
    # i.e. the single worst symbol-rate error in the benchmark was fabricated by this line,
    # and reported with no indication that it was a guess.
    #
    # Low adjacent correlation is therefore NOT sufficient evidence of a symbol-spaced
    # capture; at low SNR it is indistinguishable from a noise-dominated oversampled one.
    # Rather than fabricate Fs, keep the measured cyclostationary estimate and let
    # `baud_confidence` (0.0 when no independent feature path corroborates it) tell the caller
    # the number is untrustworthy. Distinguishing a genuine 1 SPS capture at low SNR needs a
    # different detector and is recorded as future work in IMPROVEMENTS.md.
    r1 = float(np.abs(np.corrcoef(x_bb[1:], x_bb[:-1])[0, 1])) if len(x_bb) > 2 else 0.0
    rs = max(rs, fs * 0.005)
    sps = float(np.clip(fs / rs, 1.0, 50.0))

    if diag is not None:
        diag.update({
            "baud_confidence": float(baud_conf),
            "baud_rate_hz": float(rs),
            "sps": float(sps),
            "adjacent_correlation": r1,
            "cfo_hz": float(f_cfo),
        })

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
