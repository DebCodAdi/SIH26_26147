"""Phase 3: Multi-Mode Linear Demappers (BPSK, QPSK, 8PSK, 16-QAM, 64-QAM) & FM Discriminator."""
import numpy as np

def demodulate_fsk(x_bb: np.ndarray, sps: float, num_tones: int = 2) -> np.ndarray:
    """Quadrature FM discriminator demodulation for 2-FSK and 4-FSK."""
    if len(x_bb) < 2:
        return np.array([], dtype=np.uint8)

    inst_phase = np.unwrap(np.angle(x_bb))
    inst_freq = np.diff(inst_phase)
    int_sps = max(1, int(round(sps)))
    symbols = inst_freq[int_sps // 2 :: int_sps]
    symbols = symbols - np.mean(symbols)
    std_s = float(np.std(symbols))
    if std_s > 0:
        symbols /= std_s

    if num_tones == 2:
        return (symbols > 0).astype(np.uint8)
    else:  # 4-FSK (2 bits per symbol)
        bits = []
        for s in symbols:
            if s < -1.0:   bits.extend([0, 0])
            elif s < 0.0:  bits.extend([0, 1])
            elif s < 1.0:  bits.extend([1, 1])
            else:          bits.extend([1, 0])
        return np.array(bits, dtype=np.uint8)

def compute_llr(symbols: np.ndarray, mod_type: str = "QPSK", noise_var: float = 0.1) -> np.ndarray:
    """
    Computes soft log-likelihood ratios LLR = ln(P(b=0)/P(b=1)) for soft-decision Viterbi & LDPC.
    Positive LLR -> bit 0, Negative LLR -> bit 1.
    """
    if len(symbols) == 0:
        return np.zeros(0, dtype=np.float32)

    s_norm = symbols / (np.sqrt(np.mean(np.abs(symbols)**2)) + 1e-12)
    nv = max(1e-4, noise_var)

    if mod_type in ("BPSK", "GMSK"):
        llr = 2.0 * np.real(s_norm) / nv
        return llr.astype(np.float32)

    elif mod_type == "QPSK":
        llr_i = 2.0 * np.real(s_norm) / nv
        llr_q = 2.0 * np.imag(s_norm) / nv
        return np.column_stack([llr_i, llr_q]).flatten().astype(np.float32)

    elif mod_type == "16-QAM":
        i_val = np.real(s_norm)
        q_val = np.imag(s_norm)
        # Approximate Max-Log LLR for 16-QAM
        llr_i0 = 2.0 * i_val / nv
        llr_i1 = (2.0 / np.sqrt(5.0) - np.abs(i_val)) / nv
        llr_q0 = 2.0 * q_val / nv
        llr_q1 = (2.0 / np.sqrt(5.0) - np.abs(q_val)) / nv
        return np.column_stack([llr_i0, llr_i1, llr_q0, llr_q1]).flatten().astype(np.float32)

    else:
        # Default linear projection
        llr_i = 2.0 * np.real(s_norm) / nv
        llr_q = 2.0 * np.imag(s_norm) / nv
        return np.column_stack([llr_i, llr_q]).flatten().astype(np.float32)

def demap_linear(symbols: np.ndarray, mod_type: str = "QPSK") -> list[tuple[str, np.ndarray]]:
    """Multi-hypothesis linear coordinate demapper supporting BPSK, QPSK, 8PSK, 16-QAM, and 64-QAM."""
    modes = []
    if len(symbols) == 0:
        return modes

    if mod_type in ("BPSK", "GMSK"):
        # Mode 1: Standard Real slicing
        bits_re = (np.real(symbols) > 0).astype(np.uint8)
        modes.append(("BPSK_Standard", bits_re))
        # Mode 2: Inverted Real slicing (180 deg phase ambiguity)
        modes.append(("BPSK_Inverted", (1 - bits_re).astype(np.uint8)))
        # Mode 3: Imag slicing (90 deg ambiguity)
        bits_im = (np.imag(symbols) > 0).astype(np.uint8)
        modes.append(("BPSK_Imag_Std", bits_im))
        modes.append(("BPSK_Imag_Inv", (1 - bits_im).astype(np.uint8)))

    elif mod_type == "8PSK":
        gray_table = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0]
        ], dtype=np.uint8)
        for k in range(8):
            rot_syms = symbols * np.exp(1j * k * (np.pi / 4.0))
            angles = np.mod(np.angle(rot_syms) + np.pi / 8.0, 2.0 * np.pi)
            sectors = (angles / (np.pi / 4.0)).astype(int) % 8
            bits_8psk = gray_table[sectors].flatten()
            modes.append((f"8PSK_Rot{k*45}", bits_8psk))
            
            # Natural binary 8PSK
            nat_bits = []
            for s in sectors:
                nat_bits.extend([(s >> 2) & 1, (s >> 1) & 1, s & 1])
            modes.append((f"8PSK_Nat_Rot{k*45}", np.array(nat_bits, dtype=np.uint8)))

    elif mod_type == "16-QAM":
        def slice_4pam_axis(vals, thresh):
            """Gray-coded 4-PAM: (b0,b1) where b0=sign, b1=outer."""
            out = []
            for v in vals:
                if v < -thresh:   out.extend([0, 0])
                elif v < 0.0:     out.extend([0, 1])
                elif v < thresh:  out.extend([1, 1])
                else:             out.extend([1, 0])
            return out

        def slice_4pam_natural(vals, thresh):
            """Natural binary 4-PAM: 0, 1, 2, 3."""
            out = []
            for v in vals:
                if v < -thresh:   out.extend([0, 0])
                elif v < 0.0:     out.extend([0, 1])
                elif v < thresh:  out.extend([1, 0])
                else:             out.extend([1, 1])
            return out

        for k in range(4):
            rot_syms = symbols * np.exp(1j * k * (np.pi / 2.0))
            i_vals = np.real(rot_syms)
            q_vals = np.imag(rot_syms)
            i_rms = float(np.sqrt(np.mean(i_vals**2))) + 1e-12
            q_rms = float(np.sqrt(np.mean(q_vals**2))) + 1e-12
            i_thresh = i_rms * (2.0 / np.sqrt(5.0))
            q_thresh = q_rms * (2.0 / np.sqrt(5.0))

            i_bits = slice_4pam_axis(i_vals, i_thresh)
            q_bits = slice_4pam_axis(q_vals, q_thresh)
            i_arr = np.array(i_bits, dtype=np.uint8).reshape((-1, 2))
            q_arr = np.array(q_bits, dtype=np.uint8).reshape((-1, 2))
            bits_16qam = np.hstack([i_arr, q_arr]).flatten().astype(np.uint8)
            modes.append((f"16QAM_Rot{k*90}", bits_16qam))

            # Natural 16-QAM
            in_bits = slice_4pam_natural(i_vals, i_thresh)
            qn_bits = slice_4pam_natural(q_vals, q_thresh)
            in_arr = np.array(in_bits, dtype=np.uint8).reshape((-1, 2))
            qn_arr = np.array(qn_bits, dtype=np.uint8).reshape((-1, 2))
            bits_16qam_nat = np.hstack([in_arr, qn_arr]).flatten().astype(np.uint8)
            modes.append((f"16QAM_Nat_Rot{k*90}", bits_16qam_nat))

    elif mod_type == "64-QAM":
        def slice_8pam_axis(vals, scale):
            # 8-PAM Gray coding: 3 bits per axis (b0, b1, b2)
            out = []
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) * scale
            gray_map = [
                [0,0,0], [0,0,1], [0,1,1], [0,1,0],
                [1,1,0], [1,1,1], [1,0,1], [1,0,0]
            ]
            for v in vals:
                idx = int(np.argmin(np.abs(v - levels)))
                out.extend(gray_map[idx])
            return out

        def slice_8pam_natural(vals, scale):
            # 8-PAM Natural coding: 0..7 binary
            out = []
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) * scale
            for v in vals:
                idx = int(np.clip(int(np.argmin(np.abs(v - levels))), 0, 7))
                out.extend([(idx >> 2) & 1, (idx >> 1) & 1, idx & 1])
            return out

        for k in range(4):
            rot_syms = symbols * np.exp(1j * k * (np.pi / 2.0))
            scale = float(np.sqrt(np.mean(np.abs(rot_syms)**2)) / np.sqrt(42.0)) + 1e-12
            i_bits = slice_8pam_axis(np.real(rot_syms), scale)
            q_bits = slice_8pam_axis(np.imag(rot_syms), scale)
            i_arr = np.array(i_bits, dtype=np.uint8).reshape((-1, 3))
            q_arr = np.array(q_bits, dtype=np.uint8).reshape((-1, 3))
            bits_64qam = np.hstack([i_arr, q_arr]).flatten().astype(np.uint8)
            modes.append((f"64QAM_Rot{k*90}", bits_64qam))

            in_bits = slice_8pam_natural(np.real(rot_syms), scale)
            qn_bits = slice_8pam_natural(np.imag(rot_syms), scale)
            in_arr = np.array(in_bits, dtype=np.uint8).reshape((-1, 3))
            qn_arr = np.array(qn_bits, dtype=np.uint8).reshape((-1, 3))
            bits_64qam_nat = np.hstack([in_arr, qn_arr]).flatten().astype(np.uint8)
            modes.append((f"64QAM_Nat_Rot{k*90}", bits_64qam_nat))

    else:  # QPSK / Default Multi-ary
        # Mode A: Standard Diagonal (I > 0, Q > 0)
        bits_diag = np.column_stack([np.real(symbols) > 0, np.imag(symbols) > 0]).flatten().astype(np.uint8)
        modes.append(("Linear_Diagonal", bits_diag))

        # Mode B: Axis-Aligned 45-deg Rotated Demap
        rot_syms = symbols * np.exp(1j * np.pi / 4.0)
        bits_axis = np.column_stack([np.real(rot_syms) > 0, np.imag(rot_syms) > 0]).flatten().astype(np.uint8)
        modes.append(("Linear_AxisRotated", bits_axis))

        # Mode C & D: Phase Ambiguity Quadrants (90 and 270 deg)
        rot_90 = symbols * np.exp(1j * np.pi / 2.0)
        bits_90 = np.column_stack([np.real(rot_90) > 0, np.imag(rot_90) > 0]).flatten().astype(np.uint8)
        modes.append(("Linear_Rot90", bits_90))

        rot_270 = symbols * np.exp(-1j * np.pi / 2.0)
        bits_270 = np.column_stack([np.real(rot_270) > 0, np.imag(rot_270) > 0]).flatten().astype(np.uint8)
        modes.append(("Linear_Rot270", bits_270))

        # Natural Binary QPSK Quadrants
        for k in range(4):
            r_sym = symbols * np.exp(1j * k * (np.pi / 2.0))
            ang = np.mod(np.angle(r_sym), 2.0 * np.pi)
            sec = (ang / (np.pi / 2.0)).astype(int) % 4
            nat_bits = []
            for s in sec:
                nat_bits.extend([(s >> 1) & 1, s & 1])
            modes.append((f"QPSK_Nat_Rot{k*90}", np.array(nat_bits, dtype=np.uint8)))

    return modes
