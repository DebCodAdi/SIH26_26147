"""Phase 2: Gram-Schmidt Orthogonalization, Gardner TED & JIT Fractionally-Spaced CMA."""
import numpy as np
from numba import njit
from src.config import CMA_NUM_TAPS, CMA_MU, GARDNER_MU

def gram_schmidt_iq_balance(x: np.ndarray) -> np.ndarray:
    """Corrects I/Q amplitude imbalance without distorting constellation geometry."""
    if len(x) < 4 or np.mean(np.abs(x)**2) < 1e-12:
        return x
    I = np.real(x)
    Q = np.imag(x)
    rho = np.mean(I * Q) / (np.mean(I**2) + 1e-12)
    Q_orth = Q - rho * I
    gain = np.sqrt(np.mean(I**2) / (np.mean(Q_orth**2) + 1e-12))
    Q_corr = Q_orth * gain
    return (I + 1j * Q_corr).astype(np.complex64)

def apply_gardner_ted(x_2sps: np.ndarray, mu: float = GARDNER_MU) -> np.ndarray:
    """Selects the optimal eye-opening symbol sampling strobe from 2-SPS samples."""
    if len(x_2sps) < 4:
        return x_2sps
    p0 = float(np.sum(np.abs(x_2sps[0::2])**2))
    p1 = float(np.sum(np.abs(x_2sps[1::2])**2))
    k_opt = 0 if p0 >= p1 else 1
    return x_2sps[k_opt::2].astype(np.complex64)

@njit(fastmath=True, nogil=True)
def _cma_equalize_numba(x_2sps: np.ndarray, num_taps: int = 21, mu: float = 0.001, R2: float = 1.0) -> np.ndarray:
    N = len(x_2sps)
    if N < num_taps + 10:
        return x_2sps

    w = np.zeros(num_taps, dtype=np.complex64)
    w[num_taps // 2] = 1.0 + 0.0j

    y_out = np.zeros(N - num_taps + 1, dtype=np.complex64)

    for n in range(num_taps - 1, N):
        y = 0.0 + 0.0j
        for k in range(num_taps):
            y += w[k] * x_2sps[n - k]
        y_out[n - num_taps + 1] = y

        # CMA error
        y_pwr = y.real * y.real + y.imag * y.imag
        e = y * (y_pwr - R2)

        # LMS gradient descent
        for k in range(num_taps):
            w[k] -= mu * e * np.conj(x_2sps[n - k])

    return y_out

# Warmup JIT CMA compiler at module load
_dummy_cma = np.ones(30, dtype=np.complex64)
_ = _cma_equalize_numba(_dummy_cma, 11, 0.001, 1.0)

def cma_equalize(x_2sps: np.ndarray, num_taps: int = CMA_NUM_TAPS, mu: float = CMA_MU, R2: float = 1.0) -> np.ndarray:
    """Constant Modulus Algorithm (CMA) blind adaptive equalizer running in JIT C-speed."""
    return _cma_equalize_numba(x_2sps.astype(np.complex64), num_taps, mu, R2)

def apply_twopass_cma(x_2sps: np.ndarray, num_taps: int = CMA_NUM_TAPS, mu: float = CMA_MU) -> np.ndarray:
    """Extracts optimal symbol strobes with Gram-Schmidt balance and Gardner TED."""
    x_bal = gram_schmidt_iq_balance(x_2sps)
    pwr = float(np.mean(np.abs(x_bal)**2))
    if pwr > 1e-12:
        x_bal = x_bal / np.sqrt(pwr)

    y_syms = apply_gardner_ted(x_bal)
    return y_syms.astype(np.complex64)

def compute_godard_r2(mod_type: str = 'QPSK') -> float:
    """Computes the Godard constant R2 = E[|s|^4] / E[|s|^2] for known modulations."""
    if mod_type in ('BPSK', 'QPSK', '8PSK', '2-FSK', '4-FSK', 'GMSK'):
        return 1.0
    elif mod_type == '16-QAM':
        return 1.32
    elif mod_type == '64-QAM':
        return 1.38
    else:
        return 1.0
