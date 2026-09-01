"""Phase 4: Blind Interleaver Parameter Estimation & Multi-Mode Deinterleavers."""
import numpy as np

def estimate_interleaver_depth(bits: np.ndarray, max_depth: int = 64) -> list[int]:
    """
    Estimates candidate interleaver block depths via auto-mutual-information / lag correlation.
    Returns list of candidate depths, with standard depths [1, 4, 8, 16] included.
    """
    candidates = [1, 4, 8, 16]
    if len(bits) < 128:
        return candidates

    b_zero = bits.astype(float) - np.mean(bits)
    n = len(b_zero)
    scores = {}

    for d in range(2, min(max_depth + 1, n // 4)):
        corr = float(np.abs(np.mean(b_zero[:-d] * b_zero[d:])))
        scores[d] = corr

    sorted_depths = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    top_detected = [d for d in sorted_depths[:3] if d not in candidates]
    return candidates + top_detected

def interleave_block(data: np.ndarray, m: int) -> np.ndarray:
    """Matrix block interleaver across m parallel paths with zero padding."""
    if m <= 1 or len(data) < m:
        return data
    pad = (m - (len(data) % m)) % m
    padded = np.pad(data, (0, pad), mode='constant') if pad > 0 else data
    return padded.reshape((-1, m)).T.flatten()

def deinterleave_block(data: np.ndarray, m: int) -> np.ndarray:
    """Matrix block deinterleaver restoring m parallel paths."""
    if m <= 1 or len(data) < m:
        return data
    pad = (m - (len(data) % m)) % m
    padded = np.pad(data, (0, pad), mode='constant') if pad > 0 else data
    return padded.reshape((m, -1)).T.flatten()

def deinterleave_convolutional(data: np.ndarray, I: int = 4, M: int = 16) -> np.ndarray:
    """
    Forney-type convolutional deinterleaver with I branches and step M.
    Skips end-to-end pipeline latency D = I * (I - 1) * M.
    """
    if len(data) < I * (I - 1) * M:
        return data

    delays = [(I - 1 - k) * M for k in range(I)]
    fifos = [np.zeros(d, dtype=data.dtype) if d > 0 else None for d in delays]
    
    out = []
    for idx, sym in enumerate(data):
        branch = idx % I
        fifo = fifos[branch]
        if fifo is not None and len(fifo) > 0:
            out_sym = fifo[0]
            fifo[:-1] = fifo[1:]
            fifo[-1] = sym
            out.append(out_sym)
        else:
            out.append(sym)

    out_arr = np.array(out, dtype=data.dtype)
    latency = I * (I - 1) * M
    if len(out_arr) > latency:
        return out_arr[latency:]
    return out_arr
