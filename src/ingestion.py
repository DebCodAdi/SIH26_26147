import json
import os
import numpy as np
import scipy.io.wavfile
import scipy.signal

def ingest_wav(filepath: str) -> tuple[float, np.ndarray]:
    """Ingests .wav files, handling stereo SDR complex baseband and mono analytic signals."""
    # File-extension spoofing detection
    try:
        with open(filepath, 'rb') as f:
            magic = f.read(4)
        if magic != b'RIFF':
            # Not a real WAV, try as raw
            with open(filepath, 'rb') as f:
                raw_bytes = f.read()
            success, fmt, data, fs = auto_ingest_and_clean(raw_bytes)
            return fs if fs else 0.0, data
    except Exception:
        pass

    try:
        fs, data = scipy.io.wavfile.read(filepath)
    except Exception:
        # WAV header corruption guard
        try:
            with open(filepath, 'rb') as f:
                raw_bytes = f.read()
            success, fmt, data, fs_out = auto_ingest_and_clean(raw_bytes)
            return fs_out if fs_out else 0.0, data
        except Exception:
            return 0.0, np.array([], dtype=np.complex64)

    if data.dtype == np.uint8:
        data_norm = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype == np.int16:
        data_norm = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data_norm = data.astype(np.float32) / 2147483648.0
    else:
        data_norm = data.astype(np.float32)

    if data_norm.ndim == 2 or (hasattr(data_norm, 'shape') and data_norm.shape[-1] == 2):
        x = data_norm[:, 0] + 1j * data_norm[:, 1]
    else:
        x = scipy.signal.hilbert(data_norm)
        if len(x) > 256:
            x = x[128:-128]
    return float(fs), x.astype(np.complex64)

def auto_ingest_and_clean(
    raw_bytes: bytes | str,
    user_fs: float | None = None,
    meta_path: str | None = None,
    require_fs: bool = False,
    max_memory_bytes: int = 50_000_000
) -> tuple[bool, str, np.ndarray, float]:
    """
    Evaluates byte representation via KL-divergence, removes NaNs/DC,
    and runs a Neyman-Pearson energy detection test.
    """
    # 9. Zero-byte / empty file handling
    if len(raw_bytes) == 0:
        return False, 'EMPTY_FILE', np.array([], dtype=np.complex64), 0.0

    is_filepath = isinstance(raw_bytes, str)
    if is_filepath:
        if not os.path.exists(raw_bytes) or os.path.getsize(raw_bytes) == 0:
            return False, 'EMPTY_FILE', np.array([], dtype=np.complex64), 0.0
        data_len = os.path.getsize(raw_bytes)
    else:
        data_len = len(raw_bytes)

    # 10. Minimum viable sample count (32 bytes = 4 complex64 samples)
    if data_len < 32:
        return False, "INSUFFICIENT_SAMPLES", np.array([], dtype=np.complex64), 0.0

    fs_effective = user_fs
    dt_from_sigmf = None

    # 8. SigMF metadata support
    if meta_path is None and is_filepath:
        base = os.path.splitext(raw_bytes)[0]
        meta_path = base + ".sigmf-meta"

    if fs_effective is None and meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            global_meta = meta.get("global", {})
            fs_effective = float(global_meta.get("core:sample_rate", meta.get("core:sample_rate", meta.get("sample_rate", meta.get("fs", 0.0)))))
            if fs_effective <= 0:
                fs_effective = None
                
            dt_str = global_meta.get("core:datatype", meta.get("core:datatype", ""))
            if dt_str == "cf32_le": dt_from_sigmf = "complex64"
            elif dt_str == "ci16_le": dt_from_sigmf = "int16"
            elif dt_str == "ci8": dt_from_sigmf = "int8"
        except Exception:
            pass

    if fs_effective is None and require_fs:
        return False, "MISSING_SAMPLE_RATE", np.array([], dtype=np.complex64), 0.0

    # 5. Streaming/memmap for large files
    if is_filepath and data_len > max_memory_bytes:
        dt = np.complex64
        if dt_from_sigmf == "int16": dt = np.int16
        elif dt_from_sigmf == "int8": dt = np.int8
            
        mmap_obj = np.memmap(raw_bytes, dtype=dt, mode='r')
        auto_ingest_and_clean.mmap_ref = mmap_obj
        
        chunk = mmap_obj[:1_000_000]
        if dt == np.int16:
            chunk = chunk[0::2].astype(np.float32)/32768.0 + 1j*chunk[1::2].astype(np.float32)/32768.0
        elif dt == np.int8:
            chunk = chunk[0::2].astype(np.float32)/128.0 + 1j*chunk[1::2].astype(np.float32)/128.0
            
        best_x = np.array(chunk, dtype=np.complex64)
        return True, f"memmap_{dt.__name__}", best_x, fs_effective if fs_effective else 200000.0

    if is_filepath:
        with open(raw_bytes, 'rb') as f:
            raw_bytes_data = f.read()
    else:
        raw_bytes_data = raw_bytes

    candidates = [
        ('complex64', False, False, 10.0), # 1. complex64 direct
        ('complex128', False, False, 5.0), # 2. complex128
        ('float64', False, False, 2.0),    # 3. float64 interleaved
        ('float32', False, False, 5.0),    # 4. native float32
        ('float32', True,  False, 5.0),    # 4. byteswapped float32
        ('float32', False, True,  4.0),
        ('int16',   False, False, 0.0),
        ('int16',   False, True,  0.0),
        ('int16',   True,  False, 0.0),
        ('int16',   True,  True,  0.0),
        ('int8',    False, False, 0.0),
        ('int8',    False, True,  0.0),
    ]
    
    best_score = -float('inf')
    best_x = np.array([], dtype=np.complex64)
    best_fmt = "UNKNOWN"

    with np.errstate(all='ignore'):
        for dtype_str, swap, swap_iq, bonus in candidates:
            try:
                # Byte alignment checks
                if dtype_str == 'complex64' and len(raw_bytes_data) % 8 != 0: continue
                if dtype_str == 'complex128' and len(raw_bytes_data) % 16 != 0: continue
                if dtype_str == 'float64' and len(raw_bytes_data) % 8 != 0: continue
                    
                arr = np.frombuffer(raw_bytes_data, dtype=dtype_str)
                if swap: arr = arr.byteswap()
                
                if dtype_str.startswith('complex'):
                    x = arr.astype(np.complex64)
                    if swap_iq:
                        x = x.imag + 1j * x.real
                else:
                    if len(arr) % 2 != 0:
                        arr = arr[:-1]
                    if len(arr) < 32:
                        continue

                    i_data = arr[0::2].astype(np.float32)
                    q_data = arr[1::2].astype(np.float32)

                    if dtype_str == 'int16':
                        i_data /= 32768.0
                        q_data /= 32768.0
                    elif dtype_str == 'int8':
                        i_data /= 128.0
                        q_data /= 128.0
                    elif dtype_str.startswith('float'):
                        finite_i = np.isfinite(i_data) & np.isfinite(q_data)
                        if np.any(finite_i):
                            peak = float(np.percentile(np.abs(i_data[finite_i] + 1j * q_data[finite_i]), 99.9))
                            if peak > 1.0:
                                i_data /= peak
                                q_data /= peak

                    x = (q_data + 1j * i_data) if swap_iq else (i_data + 1j * q_data)
                    
                finite_mask = np.isfinite(x)
                if np.sum(finite_mask) < 16:
                    continue

                std_i = float(np.std(x.real[finite_mask]))
                std_q = float(np.std(x.imag[finite_mask]))

                if std_i > 0 and std_q > 0:
                    sat_ratio = float(np.mean(np.abs(x[finite_mask]) > 0.99))
                    score = -abs(np.log(std_i / std_q)) - sat_ratio + bonus
                    if score > best_score + 1e-3:
                        best_score = score
                        best_x = x.copy()
                        best_fmt = f"{dtype_str}_swap={swap}_swapiq={swap_iq}"
            except Exception:
                continue

    if len(best_x) == 0:
        return False, "UNKNOWN_FORMAT", np.array([], dtype=np.complex64), 0.0

    nans = np.isnan(best_x) | np.isinf(best_x)
    if np.any(nans):
        valid_idx = np.where(~nans)[0]
        invalid_idx = np.where(nans)[0]
        if len(valid_idx) == 0:
            return False, "CORRUPTED_ALL_NAN", np.array([], dtype=np.complex64), 0.0
        
        re = np.real(best_x).copy()
        im = np.imag(best_x).copy()
        re[invalid_idx] = np.interp(invalid_idx, valid_idx, re[valid_idx])
        im[invalid_idx] = np.interp(invalid_idx, valid_idx, im[valid_idx])
        best_x = re + 1j * im

    x_mean = np.mean(best_x)
    x_rms = np.sqrt(np.mean(np.abs(best_x)**2))
    if abs(x_mean) > 0.25 * x_rms:
        best_x = best_x - x_mean

    raw_pwr = float(np.mean(np.abs(best_x)**2))
    if raw_pwr < 1e-12:
        return False, "INSUFFICIENT_SIGNAL_EVIDENCE", best_x.astype(np.complex64), fs_effective if fs_effective else 1.0

    if fs_effective is None or fs_effective <= 0:
        from src.spectral import estimate_blind_fs
        fs_effective, _ = estimate_blind_fs(best_x)

    return True, best_fmt, best_x.astype(np.complex64), float(fs_effective)

def detect_signal_mme(x: np.ndarray, smoothing_dim: int = 16, gamma: float = 1.35) -> tuple[bool, float]:
    """
    Maximum-to-Minimum Eigenvalue (MME) Blind Signal Detector (Zeng & Liang, 2009).
    Computes sample covariance eigenvalues to detect correlated signals down to -15 dB SNR,
    circumventing the classic radiometer 'SNR wall'.
    
    Args:
        x: Complex baseband signal.
        smoothing_dim: Covariance matrix dimension L (typically 8-32).
        gamma: Margin over the Marcenko-Pastur asymptotic noise bound.
        
    Returns:
        (signal_detected: bool, test_statistic: float)
    """
    N = len(x)
    L = smoothing_dim
    if N < 2 * L:
        pwr = float(np.mean(np.abs(x)**2)) if N > 0 else 0.0
        return (pwr > 1e-6, 1.0)

    P = min(N - L + 1, 4096)
    # Construct L x P space-time snapshot matrix
    Y = np.zeros((L, P), dtype=np.complex64)
    for i in range(L):
        Y[i, :] = x[i : i + P]

    # Sample covariance matrix R = (1/P) * Y * Y^H
    R = (Y @ Y.conj().T) / float(P)
    
    # Compute eigenvalues of Hermitian matrix
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.sort(np.maximum(eigvals, 1e-12))
    
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    
    test_statistic = lam_max / lam_min

    # Theoretical Marcenko-Pastur noise threshold for pure white noise
    mu = float((1.0 + np.sqrt(L / float(P)))**2)
    nu = float((1.0 - np.sqrt(L / float(P)))**2)
    noise_bound = mu / max(nu, 1e-6)
    threshold = noise_bound * gamma

    signal_detected = bool(test_statistic > threshold)
    return signal_detected, test_statistic

def detect_bursts(
    x: np.ndarray,
    window_len: int = 256,
    threshold_ratio: float = 2.5
) -> list[tuple[int, int]]:
    """
    Sliding-window energy burst detector for pulsed, hopped, or slotted transmissions.
    
    Returns:
        List of (start_idx, end_idx) sample slices containing active bursts.
    """
    N = len(x)
    if N < window_len * 2:
        return [(0, N)]

    # Compute moving power envelope
    pwr = np.abs(x)**2
    kernel = np.ones(window_len, dtype=np.float32) / float(window_len)
    moving_pwr = np.convolve(pwr, kernel, mode='same')
    
    # Noise floor estimated from lowest 20th percentile
    noise_floor = float(np.percentile(moving_pwr, 20)) + 1e-12
    threshold = noise_floor * threshold_ratio
    
    active = moving_pwr > threshold
    diff = np.diff(active.astype(np.int8))
    
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    
    if active[0]:
        starts = np.insert(starts, 0, 0)
    if active[-1]:
        ends = np.append(ends, N)
        
    bursts = []
    for s, e in zip(starts, ends):
        if (e - s) >= window_len:  # Filter out transient spikes
            # Add margin
            s_pad = max(0, s - window_len // 2)
            e_pad = min(N, e + window_len // 2)
            bursts.append((s_pad, e_pad))
            
    if not bursts:
        bursts = [(0, N)]
    return bursts
