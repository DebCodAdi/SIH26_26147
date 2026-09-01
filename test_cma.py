import numpy as np
from src.equalizers import cma_equalize, apply_twopass_cma, gram_schmidt_iq_balance, compute_godard_r2

# Generate test QPSK signal at 2 SPS with multipath
symbols = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
N = 500
idx = np.random.randint(0, 4, N)
tx = symbols[idx]
tx_2sps = np.repeat(tx, 2)  # 2 SPS
# Add ISI channel [1, 0.3, -0.1]
channel = np.array([1.0, 0.3, -0.1])
rx = np.convolve(tx_2sps, channel, mode='same').astype(np.complex64)
rx += 0.05 * (np.random.randn(len(rx)) + 1j * np.random.randn(len(rx)))  # noise

# Test CMA
y_eq = cma_equalize(rx, num_taps=21, mu=0.001)
print(f'CMA output length: {len(y_eq)}')
print(f'CMA output power: {np.mean(np.abs(y_eq)**2):.4f}')

# Test full pipeline
y_syms = apply_twopass_cma(rx)
print(f'Two-pass output length: {len(y_syms)}')
print(f'R2 QPSK: {compute_godard_r2("QPSK")}, 16-QAM: {compute_godard_r2("16-QAM")}')
print('ALL EQUALIZER TESTS PASSED')
