"""Global constants, polynomials, Galois fields, and structural parameters."""
import numpy as np
import galois

# Preambles (Barker & Synchronization Sequences)
BARKER_7  = np.array([1, 1, 1, -1, -1, 1, -1], dtype=np.float32)
BARKER_11 = np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1], dtype=np.float32)
BARKER_13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float32)
DSSS_11   = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1], dtype=np.float32)
CCSDS_ASM_32 = np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8)).astype(np.float32) * 2.0 - 1.0

# NASA Standard Convolutional Code: K=7, Rate 1/2, Generators (171, 133) octal
CONV_K = 7
CONV_MEMORY = np.array([CONV_K - 1])  # Memory order M = 6 (64 trellis states)
CONV_POLYS = np.array([[0o171, 0o133]])
CONV_FLUSH_BITS = 6

# Reed-Solomon Codec: RS(255, 223) over GF(2^8)
RS_N = 255
RS_K = 223
RS_T = (RS_N - RS_K) // 2  # Corrects up to 16 byte errors
RS_CODEC = galois.ReedSolomon(RS_N, RS_K, c=1)

# IEEE 802.11n Standard LDPC Code: N=648, K=324, Rate 1/2 (Z=27, 12x24 sub-blocks)
LDPC_N = 648
LDPC_K = 324
LDPC_RATE = 0.5
LDPC_Z = 27

# Prototype shifting matrix for 802.11n Rate 1/2 (12 x 24)
LDPC_H_PROTO = np.array([
    [ 0, -1, -1, -1,  0,  0, -1, -1,  0, -1, -1,  0,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [22,  0, -1, -1, 17, -1,  0,  0, 12, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 6, -1,  0, -1, 10, -1, -1, -1, 24, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 2, -1, -1,  0, 20, -1, -1, -1, 25,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1, -1],
    [23, -1, -1, -1,  3, -1, -1, -1,  0, -1,  9, 11, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1, -1],
    [24, -1, 23,  1, 17, -1,  3, -1, 10, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1, -1],
    [25, -1, -1, -1,  8, -1, -1, -1,  7, 18, -1, -1,  0, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1, -1],
    [13, 24, -1, -1,  0, -1,  8, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1, -1],
    [ 7, 20, -1, 16, 22, 10, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1, -1],
    [11, -1, -1, -1, 19, -1, -1, -1, 13, -1,  3, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0, -1],
    [25, -1,  8, -1, 23, 18, -1, 14,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
    [ 3, -1, -1, -1, 16, -1, -1,  2, 25,  5, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0]
], dtype=int)

def build_ldpc_h_matrix(proto: np.ndarray = LDPC_H_PROTO, z: int = LDPC_Z) -> np.ndarray:
    """Expands prototype matrix into full parity-check matrix H of size (12*Z, 24*Z)."""
    rows, cols = proto.shape
    h = np.zeros((rows * z, cols * z), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            shift = proto[r, c]
            if shift >= 0:
                h[r * z : (r + 1) * z, c * z : (c + 1) * z] = np.roll(np.eye(z, dtype=np.uint8), shift, axis=1)
    return h

LDPC_H_MATRIX = build_ldpc_h_matrix()

# Constellation Maps (Normalized Unit Power)
CONSTELLATION_BPSK = np.array([-1.0, 1.0], dtype=np.complex64)
CONSTELLATION_QPSK = np.array([
    ( 1.0 + 1j) / np.sqrt(2.0),
    (-1.0 + 1j) / np.sqrt(2.0),
    ( 1.0 - 1j) / np.sqrt(2.0),
    (-1.0 - 1j) / np.sqrt(2.0)
], dtype=np.complex64)

# Equalizer & Carrier Tracking Loop Defaults
CMA_NUM_TAPS = 21
CMA_MU = 0.002
EKF_ALPHA = 0.05
EKF_BETA = 0.001
GARDNER_MU = 0.01
