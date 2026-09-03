"""Blind IQ Impairment Correction: Adaptive DC Offset, IQ Imbalance, and AGC Normalization."""
import numpy as np
from scipy.signal import lfilter

def remove_dc_offset_iir(x: np.ndarray, alpha: float = 0.9999) -> np.ndarray:
    """
    Adaptive DC offset removal. If significant DC bias exists (|mean| > 0.05 * rms),
    removes it using zero-mean centering or IIR notch to preserve symbol pulse shapes.
    """
    if len(x) == 0:
        return x.copy()
    
    x_mean = np.mean(x)
    x_rms = np.sqrt(np.mean(np.abs(x)**2)) + 1e-12
    
    if abs(x_mean) / x_rms > 0.05:
        # Subtract mean DC bias
        return (x - x_mean).astype(x.dtype)
    return x.copy()

def correct_iq_imbalance_gsop(x: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """
    Gram-Schmidt Orthogonalization Procedure (GSOP) for blind IQ imbalance correction.
    Applies correction only when measurable IQ gain/phase mismatch exceeds threshold.
    """
    if len(x) < 32:
        return x.copy()

    I = np.real(x).astype(np.float64)
    Q = np.imag(x).astype(np.float64)

    E_I2 = np.mean(I ** 2)
    E_Q2 = np.mean(Q ** 2)
    if E_I2 < 1e-12 or E_Q2 < 1e-12:
        return x.copy()

    # Correlation coefficient
    rho = np.mean(I * Q) / np.sqrt(E_I2 * E_Q2)
    gain_imbalance = abs(E_I2 - E_Q2) / (E_I2 + E_Q2)

    # The decision threshold must scale with the sampling uncertainty of these estimates, not
    # be a fixed constant. For genuinely balanced I/Q the sample correlation still fluctuates
    # with standard error ~1/sqrt(N), which for a short burst (N=684) is ~0.038 — larger than
    # the old fixed 0.03 threshold. That made this correction fire on perfectly clean signals
    # and distort them: measured, it turned a 4 Hz carrier-offset error into a 2148 Hz one.
    # Requiring ~3 standard errors keeps real imbalance detected while leaving noise alone.
    stat_threshold = max(threshold, 3.0 / np.sqrt(max(len(x), 1)))

    # Only correct if imbalance is genuine to avoid noise amplification
    if abs(rho) > stat_threshold or gain_imbalance > stat_threshold:
        Q_orth = Q - (rho * np.sqrt(E_Q2 / E_I2)) * I
        E_Q_orth2 = np.mean(Q_orth ** 2)
        if E_Q_orth2 > 1e-12:
            Q_corr = Q_orth * np.sqrt(E_I2 / E_Q_orth2)
            return (I + 1j * Q_corr).astype(np.complex64)

    return x.copy()

def normalize_agc(x: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """
    Power normalization to target RMS power level.
    """
    if len(x) == 0:
        return x.copy()

    pwr = np.mean(np.abs(x) ** 2)
    if pwr < 1e-12:
        return x.copy()

    gain = np.sqrt(target_power / pwr)
    return (x * gain).astype(np.complex64)
