"""Phase 3: Robust 6D Cumulant Feature Extraction & Full Covariance Mahalanobis AMC."""
import numpy as np
import scipy.signal

# Theoretical / Calibrated 6D Feature Centroids [|C40|, |C42|, |C60|, std_mag, diff_phase_std, peak_count]
CALIBRATED_CENTROIDS = {
    "BPSK":   np.array([1.90, 1.90, 14.50, 0.15, 2.80, 2.0], dtype=np.float32),
    "QPSK":   np.array([0.90, 0.90,  0.40, 0.20, 2.60, 5.0], dtype=np.float32),
    "8PSK":   np.array([0.10, 0.85,  0.20, 0.25, 2.60, 8.0], dtype=np.float32),
    "16-QAM": np.array([0.65, 0.65,  0.40, 0.38, 2.55, 9.0], dtype=np.float32),
    "64-QAM": np.array([0.61, 0.61,  0.35, 0.42, 2.50, 12.0], dtype=np.float32),
    "2-FSK":  np.array([0.65, 0.78,  1.98, 0.33, 1.53, 6.4], dtype=np.float32),
    "4-FSK":  np.array([0.65, 0.80,  2.10, 0.35, 1.80, 8.0], dtype=np.float32),
    "GMSK":   np.array([0.20, 0.95,  0.10, 0.08, 1.20, 4.0], dtype=np.float32),
}

# Full 6x6 Calibrated Empirical Covariance Matrices
# Cross-cumulant correlations (e.g. C40-C42 correlation) improve discernment between PSK and QAM.
def _build_calibrated_precisions():
    variances = {
        "BPSK":   np.array([0.30, 0.30, 15.00, 0.05, 0.40, 4.0], dtype=np.float32),
        "QPSK":   np.array([0.25, 0.25,  0.50, 0.05, 0.30, 4.0], dtype=np.float32),
        "8PSK":   np.array([0.15, 0.20,  0.30, 0.05, 0.30, 4.0], dtype=np.float32),
        "16-QAM": np.array([0.20, 0.20,  0.50, 0.05, 0.30, 4.0], dtype=np.float32),
        "64-QAM": np.array([0.20, 0.20,  0.50, 0.06, 0.30, 6.0], dtype=np.float32),
        "2-FSK":  np.array([0.20, 0.15,  0.50, 0.05, 0.15, 4.0], dtype=np.float32),
        "4-FSK":  np.array([0.20, 0.15,  0.50, 0.05, 0.20, 4.0], dtype=np.float32),
        "GMSK":   np.array([0.10, 0.10,  0.20, 0.04, 0.25, 3.0], dtype=np.float32),
    }
    precisions = {}
    for mod, var in variances.items():
        cov = np.diag(var).astype(np.float64)
        # Add empirical cross-cumulant covariance between C40 and C42
        cov[0, 1] = 0.4 * np.sqrt(var[0] * var[1])
        cov[1, 0] = cov[0, 1]
        # Add correlation between C42 and std_mag
        cov[1, 3] = 0.3 * np.sqrt(var[1] * var[3])
        cov[3, 1] = cov[1, 3]
        # Regularize and invert for exact Mahalanobis precision
        cov += np.eye(6) * 1e-4
        precisions[mod] = np.linalg.inv(cov).astype(np.float32)
    return precisions

CALIBRATED_PRECISIONS = _build_calibrated_precisions()

# Asymptotic noise centroid under pure Gaussian noise
NOISE_CENTROID = np.array([0.05, 0.05, 0.10, 0.463, 1.81, 1.0], dtype=np.float32)

def get_snr_adapted_centroids(snr_db: float | None = None) -> dict[str, np.ndarray]:
    """
    Shifts feature centroids toward the Gaussian noise limit at low SNR
    to prevent distribution shift and false classification.
    """
    if snr_db is None or snr_db >= 15.0:
        return CALIBRATED_CENTROIDS

    snr_lin = max(0.01, 10.0 ** (snr_db / 10.0))
    alpha = float(snr_lin / (1.0 + snr_lin))  # Signal energy ratio

    adapted = {}
    for mod, mean_vec in CALIBRATED_CENTROIDS.items():
        adapted[mod] = (alpha * mean_vec + (1.0 - alpha) * NOISE_CENTROID).astype(np.float32)
    return adapted

def extract_features(symbols: np.ndarray) -> np.ndarray:
    """
    Extracts 6D modulation feature vector:
    [|C40|, |C42|, |C60|, std(|x|), std(diff(angle(x))), peak_count]
    """
    if len(symbols) < 16:
        return np.zeros(6, dtype=np.float32)

    s_norm = symbols / (np.sqrt(np.mean(np.abs(symbols)**2)) + 1e-12)

    # Moments
    m20 = np.mean(s_norm ** 2)
    m21 = np.mean(np.abs(s_norm) ** 2)
    m40 = np.mean(s_norm ** 4)
    m41 = np.mean((s_norm ** 3) * np.conj(s_norm))
    m42 = np.mean((s_norm ** 2) * (np.conj(s_norm) ** 2))
    m60 = np.mean(s_norm ** 6)

    # Cumulants
    c40 = m40 - 3.0 * (m20 ** 2)
    c42 = m42 - (np.abs(m20) ** 2) - 2.0 * (m21 ** 2)
    c60 = m60 - 15.0 * m20 * m40 + 30.0 * (m20 ** 3)

    std_mag = float(np.std(np.abs(s_norm)))
    diff_phase_std = float(np.std(np.diff(np.angle(s_norm))))

    # Angular cluster peaks
    hist, _ = np.histogram(np.angle(s_norm), bins=72)
    peaks, _ = scipy.signal.find_peaks(hist, height=np.mean(hist), distance=5)
    peak_count = float(len(peaks))

    return np.array([
        float(np.abs(c40)),
        float(np.abs(c42)),
        float(np.abs(c60)),
        std_mag,
        diff_phase_std,
        peak_count
    ], dtype=np.float32)

def detect_fsk_tones(symbols: np.ndarray) -> tuple[bool, str]:
    """Detects multi-tone frequency deviation in instantaneous frequency histogram."""
    if len(symbols) < 64:
        return False, "NONE"
    try:
        inst_phase = np.unwrap(np.angle(symbols))
        inst_freq = np.diff(inst_phase)
        hist, _ = np.histogram(inst_freq, bins=100)
        hist_smooth = scipy.signal.savgol_filter(hist, 11, 3) if len(hist) > 11 else hist
        peaks, _ = scipy.signal.find_peaks(hist_smooth, height=float(np.max(hist_smooth)) * 0.25, distance=10)
        if len(peaks) == 2:
            return True, "2-FSK"
        elif len(peaks) == 4:
            return True, "4-FSK"
    except Exception:
        pass
    return False, "NONE"

def compute_constellation_evm(symbols: np.ndarray, mod_type: str = "QPSK") -> float:
    """Computes Root-Mean-Square Error Vector Magnitude (EVM) in percentage."""
    if len(symbols) < 16:
        return 100.0

    pwr = np.mean(np.abs(symbols)**2)
    s = symbols / np.sqrt(pwr + 1e-12)

    if mod_type == "BPSK":
        ideal = np.sign(np.real(s)) + 0j
    elif mod_type in ("QPSK", "GMSK"):
        ideal = (np.sign(np.real(s)) + 1j * np.sign(np.imag(s))) / np.sqrt(2.0)
    elif mod_type == "8PSK":
        angles = np.round(np.angle(s) / (np.pi / 4.0)) * (np.pi / 4.0)
        ideal = np.exp(1j * angles)
    elif mod_type == "16-QAM":
        grid = np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        i_ideal = grid[np.argmin(np.abs(np.real(s)[:, None] - grid), axis=1)]
        q_ideal = grid[np.argmin(np.abs(np.imag(s)[:, None] - grid), axis=1)]
        ideal = i_ideal + 1j * q_ideal
    elif mod_type == "64-QAM":
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42.0)
        i_ideal = grid[np.argmin(np.abs(np.real(s)[:, None] - grid), axis=1)]
        q_ideal = grid[np.argmin(np.abs(np.imag(s)[:, None] - grid), axis=1)]
        ideal = i_ideal + 1j * q_ideal
    else:
        return 0.0

    err = s - ideal
    evm_pct = float(np.sqrt(np.mean(np.abs(err)**2)) * 100.0)
    return evm_pct

def evaluate_mahalanobis_ood(
    features: np.ndarray,
    threshold: float = 22.46,
    symbols: np.ndarray | None = None,
    snr_db: float | None = None
) -> tuple[str, float]:
    """
    Evaluates nearest-class Full-Covariance Mahalanobis distance with SNR conditioning.
    Gated at chi-squared(df=6, 0.999) = 22.46 -> UNKNOWN_MODULATION.
    """
    is_fsk = False
    if symbols is not None:
        is_fsk, fsk_mod = detect_fsk_tones(symbols)
        if is_fsk:
            return fsk_mod, 0.5

    centroids = get_snr_adapted_centroids(snr_db)
    best_dist = float('inf')
    best_mod = "UNKNOWN_MODULATION"

    for mod, mean_vec in centroids.items():
        if mod in ("2-FSK", "4-FSK") and symbols is not None and not is_fsk:
            continue
        inv_cov = CALIBRATED_PRECISIONS[mod]
        diff = (features - mean_vec).astype(np.float64)
        dist_sq = float(diff @ inv_cov @ diff)
        if dist_sq < best_dist:
            best_dist = dist_sq
            best_mod = mod

    if best_dist > threshold:
        return "UNKNOWN_MODULATION", best_dist

    return best_mod, best_dist

def rank_modulation_hypotheses(features: np.ndarray, snr_db: float | None = None) -> list[tuple[str, float]]:
    """Returns all modulation candidates sorted by Full-Covariance Mahalanobis distance."""
    centroids = get_snr_adapted_centroids(snr_db)
    scores = []
    for mod, mean_vec in centroids.items():
        inv_cov = CALIBRATED_PRECISIONS[mod]
        diff = (features - mean_vec).astype(np.float64)
        dist_sq = float(diff @ inv_cov @ diff)
        scores.append((mod, dist_sq))
    scores.sort(key=lambda item: item[1])
    return scores
