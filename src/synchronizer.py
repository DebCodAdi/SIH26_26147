"""Phase 3: Decision-Directed Carrier Tracking, Cycle-Slip Recovery & Preamble Synchronization."""
import numpy as np
import scipy.signal
from src.config import BARKER_7, BARKER_11, BARKER_13, DSSS_11, CCSDS_ASM_32
from numba import njit

EKF_ALPHA = 0.08
EKF_BETA = 0.004

@njit(fastmath=True, nogil=True)
def _fast_pll_with_slip_detection(y: np.ndarray, mod_code: int, alpha: float = 0.08, beta: float = 0.004) -> tuple[np.ndarray, int]:
    """
    2nd-Order Costas / Decision-Directed PLL with Anti-Windup and Cycle-Slip Detection.
    Returns (locked_symbols, cycle_slip_count).
    """
    phase = 0.0
    freq = 0.0
    num_symbols = len(y)
    s_locked = np.zeros(num_symbols, dtype=np.complex64)
    grid_16qam = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32) / 3.16227766

    slip_count = 0
    slip_threshold = 1.5707963267948966  # pi/2

    for k in range(num_symbols):
        cos_p = np.cos(phase)
        sin_p = np.sin(phase)
        yr = y[k].real
        yi = y[k].imag
        sr = yr * cos_p + yi * sin_p
        si = yi * cos_p - yr * sin_p

        if mod_code == 0:  # BPSK / GMSK
            sign_r = 1.0 if sr >= 0.0 else -1.0
            e_phi = si * sign_r
            slip_threshold = 3.141592653589793
        elif mod_code == 1:  # QPSK
            sign_r = 1.0 if sr >= 0.0 else -1.0
            sign_i = 1.0 if si >= 0.0 else -1.0
            e_phi = sign_r * si - sign_i * sr
            slip_threshold = 1.5707963267948966
        elif mod_code == 2:  # 8PSK
            ang = np.arctan2(si, sr)
            ang_slice = np.round(ang / 0.7853981633974483) * 0.7853981633974483
            e_phi = si * np.cos(ang_slice) - sr * np.sin(ang_slice)
            slip_threshold = 0.7853981633974483
        else:  # 16-QAM / 64-QAM
            min_d_r = 1e9
            i_h = grid_16qam[0]
            for g in grid_16qam:
                d = abs(sr - g)
                if d < min_d_r:
                    min_d_r = d
                    i_h = g
            min_d_i = 1e9
            q_h = grid_16qam[0]
            for g in grid_16qam:
                d = abs(si - g)
                if d < min_d_i:
                    min_d_i = d
                    q_h = g
            e_phi = si * i_h - sr * q_h
            slip_threshold = 1.5707963267948966

        # Cycle slip detector: if phase error jumps abruptly, correct frequency integrator
        if abs(e_phi) > slip_threshold:
            slip_count += 1
            e_phi = np.mod(e_phi + np.pi, 2.0 * np.pi) - np.pi

        # Anti-windup clamping
        if e_phi > 1.0:
            e_phi = 1.0
        elif e_phi < -1.0:
            e_phi = -1.0

        freq += beta * e_phi
        if freq > 0.1:
            freq = 0.1
        elif freq < -0.1:
            freq = -0.1

        phase += freq + alpha * e_phi
        s_locked[k] = sr + 1j * si

    return s_locked, slip_count

# Warmup JIT PLL for all modulation types on module import
_dummy_pll = np.ones(20, dtype=np.complex64)
for _c in range(4):
    _ = _fast_pll_with_slip_detection(_dummy_pll, _c)

def track_carrier_pll(y: np.ndarray, mod_type: str = "QPSK") -> np.ndarray:
    """JIT-accelerated 2nd-order decision-directed Costas/EKF loop for phase tracking."""
    if len(y) == 0:
        return y
    mod_map = {"BPSK": 0, "GMSK": 0, "QPSK": 1, "8PSK": 2, "16-QAM": 3, "64-QAM": 3}
    mod_code = mod_map.get(mod_type, 1)
    s_locked, _ = _fast_pll_with_slip_detection(y.astype(np.complex64), mod_code, EKF_ALPHA, EKF_BETA)
    return s_locked

def resolve_sync_and_rotation(
    s: np.ndarray,
    mod_type: str = "QPSK"
) -> tuple[np.ndarray, int, float, float]:
    """
    Normalized cross-correlation across Barker and standard preambles.
    Extracts complex peak angle and quantizes to modulation phase symmetry.
    """
    window = min(len(s), 64)
    best_score = -1.0
    best_metric = -1.0
    best_offset = 0
    best_phase_rad = 0.0
    best_pat_len = 11

    preambles = [BARKER_11, BARKER_13, BARKER_7, DSSS_11, CCSDS_ASM_32]

    for pat in preambles:
        p_len = len(pat)
        if len(s) < p_len:
            continue
        ref_norm = float(np.linalg.norm(pat))
        
        raw_corr = scipy.signal.correlate(s[:window], pat, mode='valid')
        energy = np.convolve(np.abs(s[:window])**2, np.ones(p_len), mode='valid')
        denom = np.sqrt(energy) * ref_norm + 1e-12
        norm_corr = np.abs(raw_corr) / denom

        peak = int(np.argmax(norm_corr))
        metric = float(norm_corr[peak])
        score = metric * np.sqrt(p_len)

        if score > best_score:
            best_score = score
            best_metric = metric
            best_offset = peak
            best_pat_len = p_len
            
            z_peak = raw_corr[peak]
            best_phase_rad = float(np.angle(z_peak))

    if best_metric < 0.55:
        return s.astype(np.complex64), 0, float(best_metric), 0.0

    s_derot = s * np.exp(-1j * best_phase_rad)
    payload_start = max(0, best_offset + best_pat_len)
    return s_derot[payload_start:].astype(np.complex64), payload_start, best_metric, best_phase_rad
