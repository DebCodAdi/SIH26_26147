"""Synthetic Testbench Vector Generator for PS 26147 Blind SDR Interceptor."""
import os
import zlib
import numpy as np
import scipy.signal
import commpy.channelcoding.convcode as cc
from src.config import (
    BARKER_11, BARKER_13, BARKER_7,
    CONV_MEMORY, CONV_POLYS,
    RS_CODEC,
    LDPC_H_MATRIX
)
from src.deinterleaver import interleave_block
from src.fec_decoders import encode_reed_solomon, encode_ldpc, encode_convolutional_viterbi

def generate_test_vector(
    mod_type: str = "QPSK",
    fec_type: str = "NONE",
    interleaver_depth: int = 1,
    payload_text: bytes = b"PS26147_TEST_PAYLOAD_PACKET_9999",
    fs: float = 200000.0,
    baud_rate: float = 50000.0,
    cfo_hz: float = 350.0,
    snr_db: float = 25.0,
    multipath: bool = False
) -> tuple[np.ndarray, bytes]:
    """
    Generates a realistic synthetic SDR capture with specified modulation, FEC,
    interleaving, carrier frequency offset (CFO), and optional multipath / AWGN.
    Returns (iq_samples_complex64, ground_truth_payload).
    """
    # 1. Add CRC-32 to Payload
    crc = zlib.crc32(payload_text).to_bytes(4, byteorder='big')
    data_frame = payload_text + crc

    # 2. Apply Forward Error Correction (FEC) & Interleaving
    if fec_type in ["RS", "RS(255,223)", "RS255"]:
        data_arr = np.frombuffer(data_frame, dtype=np.uint8)
        encoded_bytes = encode_reed_solomon(data_arr, RS_CODEC)
        if interleaver_depth > 1:
            encoded_bytes = interleave_block(encoded_bytes, interleaver_depth)
        raw_bits = np.unpackbits(encoded_bytes)

    elif fec_type in ["CONV", "CONV_K7_R1/2", "VITERBI"]:
        raw_bits_uncoded = np.unpackbits(np.frombuffer(data_frame, dtype=np.uint8))
        raw_bits = encode_convolutional_viterbi(raw_bits_uncoded)

    elif fec_type in ["CONV+RS", "CONV+RS(255,223)", "CONCAT"]:
        data_arr = np.frombuffer(data_frame, dtype=np.uint8)
        encoded_bytes = encode_reed_solomon(data_arr, RS_CODEC)
        if interleaver_depth > 1:
            encoded_bytes = interleave_block(encoded_bytes, interleaver_depth)
        rs_bits = np.unpackbits(encoded_bytes)
        raw_bits = encode_convolutional_viterbi(rs_bits)

    elif fec_type in ["LDPC", "LDPC_N648_K324"]:
        raw_bits_uncoded = np.unpackbits(np.frombuffer(data_frame, dtype=np.uint8))
        raw_bits = encode_ldpc(raw_bits_uncoded, LDPC_H_MATRIX)

    else:  # Uncoded
        raw_bits = np.unpackbits(np.frombuffer(data_frame, dtype=np.uint8))

    # 3. Modulation Mapping
    if mod_type == "BPSK":
        syms = (2.0 * raw_bits.astype(np.float32) - 1.0).astype(np.complex64)
        preamble = (BARKER_11 + 0j).astype(np.complex64)
        postamble = np.zeros(24, dtype=np.complex64)
        tx_symbols = np.concatenate([preamble, syms, postamble])

    elif mod_type == "QPSK":
        if len(raw_bits) % 2 != 0:
            raw_bits = np.append(raw_bits, 0)
        b_mat = raw_bits.reshape((-1, 2))
        syms = ((2.0 * b_mat[:, 0] - 1.0) + 1j * (2.0 * b_mat[:, 1] - 1.0)) / np.sqrt(2.0)
        preamble = (BARKER_11 + 0j).astype(np.complex64)
        postamble = np.zeros(24, dtype=np.complex64)
        tx_symbols = np.concatenate([preamble, syms.astype(np.complex64), postamble])

    elif mod_type == "8PSK":
        pad_len = (3 - (len(raw_bits) % 3)) % 3
        if pad_len > 0:
            raw_bits = np.append(raw_bits, np.zeros(pad_len, dtype=int))
        b_mat = raw_bits.reshape((-1, 3))
        gray_map = {
            (0,0,0): 0, (0,0,1): 1, (0,1,1): 2, (0,1,0): 3,
            (1,1,0): 4, (1,1,1): 5, (1,0,1): 6, (1,0,0): 7
        }
        phases = np.array([gray_map[tuple(row)] * (np.pi / 4.0) for row in b_mat])
        syms = np.exp(1j * phases).astype(np.complex64)
        preamble = (BARKER_11 + 0j).astype(np.complex64)
        postamble = np.zeros(24, dtype=np.complex64)
        tx_symbols = np.concatenate([preamble, syms, postamble])

    elif mod_type == "16-QAM":
        pad_len = (4 - (len(raw_bits) % 4)) % 4
        if pad_len > 0:
            raw_bits = np.append(raw_bits, np.zeros(pad_len, dtype=int))
        b_mat = raw_bits.reshape((-1, 4))
        amp_map = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
        i_syms = np.array([amp_map[(row[0], row[1])] for row in b_mat])
        q_syms = np.array([amp_map[(row[2], row[3])] for row in b_mat])
        syms = ((i_syms + 1j * q_syms) / np.sqrt(10.0)).astype(np.complex64)
        preamble = (BARKER_11 + 0j).astype(np.complex64)
        postamble = np.zeros(24, dtype=np.complex64)
        tx_symbols = np.concatenate([preamble, syms, postamble])

    elif mod_type in ["2-FSK", "4-FSK"]:
        # Continuous phase FSK waveform synthesis
        sps = fs / baud_rate
        num_tones = 4 if mod_type == "4-FSK" else 2
        bits_per_sym = 2 if num_tones == 4 else 1
        pad_len = (bits_per_sym - (len(raw_bits) % bits_per_sym)) % bits_per_sym
        if pad_len > 0:
            raw_bits = np.append(raw_bits, np.zeros(pad_len, dtype=int))
        sym_indices = np.packbits(raw_bits.reshape((-1, 8 // bits_per_sym)), axis=-1)
        # Frequency deviations
        f_dev = baud_rate / 2.0
        # Instantaneous phase integration
        sym_levels = 2.0 * (raw_bits if num_tones == 2 else raw_bits.reshape((-1, 2))[:, 0] * 2 + raw_bits.reshape((-1, 2))[:, 1]) - (num_tones - 1)
        freq_series = np.repeat(sym_levels * f_dev, int(sps))
        phase_series = 2.0 * np.pi * np.cumsum(freq_series) / fs
        tx_wave = np.exp(1j * phase_series).astype(np.complex64)
        # Return FSK waveform directly
        n = np.arange(len(tx_wave))
        tx_cfo = tx_wave * np.exp(1j * 2.0 * np.pi * cfo_hz * n / fs)
        noise_pwr = 10.0 ** (-snr_db / 10.0)
        noise = (np.random.randn(len(tx_cfo)) + 1j * np.random.randn(len(tx_cfo))) * np.sqrt(noise_pwr / 2.0)
        return (tx_cfo + noise).astype(np.complex64), payload_text

    # 4. Pulse Shaping (RRC / Nyquist)
    sps = fs / baud_rate
    up_samples = np.zeros(int(len(tx_symbols) * sps), dtype=np.complex64)
    up_samples[::int(sps)] = tx_symbols
    h_rrc = scipy.signal.firwin(31, 1.0 / sps, window='hamming')
    tx_wave = scipy.signal.convolve(up_samples, h_rrc, mode='same')

    # 5. Channel Impairments: CFO, Multipath, AWGN
    n = np.arange(len(tx_wave))
    tx_cfo = tx_wave * np.exp(1j * 2.0 * np.pi * cfo_hz * n / fs)

    if multipath:
        h_chan = np.array([1.0, 0.25 * np.exp(1j * np.pi / 4.0), 0.1 * np.exp(-1j * np.pi / 3.0)], dtype=np.complex64)
        tx_impaired = scipy.signal.convolve(tx_cfo, h_chan, mode='same')
    else:
        tx_impaired = tx_cfo

    noise_pwr = 10.0 ** (-snr_db / 10.0)
    noise = (np.random.randn(len(tx_impaired)) + 1j * np.random.randn(len(tx_impaired))) * np.sqrt(noise_pwr / 2.0)
    rx_wave = (tx_impaired + noise).astype(np.complex64)

    return rx_wave, payload_text
