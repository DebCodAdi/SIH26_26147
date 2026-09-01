import re
import sys

with open('src/pipeline.py', 'r') as f:
    content = f.read()

# 1. Add imports
content = content.replace(
    "from src.blackboard import BlackboardArbiter\n",
    "from src.blackboard import BlackboardArbiter\nfrom src.iq_correction import remove_dc_offset_iir, correct_iq_imbalance_gsop, normalize_agc\n"
)

# 2. Add estimate_snr_m2m4
snr_code = """
def estimate_snr_m2m4(x: np.ndarray) -> float:
    M2 = np.mean(np.abs(x)**2)
    M4 = np.mean(np.abs(x)**4)
    k = M4 / (M2**2 + 1e-12)
    if k > 1.95:  # pure noise
        return -10.0
    snr_linear = max(1.0 / (k - 1.0 + 1e-6), 0.01)
    return float(10 * np.log10(snr_linear))
"""
content = content.replace(
    "def estimate_snr_kurtosis(x: np.ndarray) -> float:\n",
    snr_code.lstrip() + "\ndef estimate_snr_kurtosis(x: np.ndarray) -> float:\n"
)

# 3. Replace the pipeline logic from Edge Case 1 down to Blackboard multi-hypothesis
start_idx = content.find("    # Edge Case 1: Corrupted NaNs")
if start_idx == -1:
    print("Could not find start point")
    sys.exit(1)

end_idx = content.find("    # Spectrum calculation for telemetry/GUI")
if end_idx == -1:
    print("Could not find end point")
    sys.exit(1)

new_logic = """    # Check for NaN/Inf in cleaned signal (ingestion already repairs them)
    nans_remaining = int(np.sum(~np.isfinite(x_clean)))
    if nans_remaining > len(x_clean) * 0.5:
        return {"status": "CORRUPTED_EXCESSIVE_NAN", "stage": "INGESTION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}
        
    if not valid or len(x_clean) < 64:
        return {"status": "INSUFFICIENT_SAMPLES", "stage": "INGESTION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    try:
        x_clean = remove_dc_offset_iir(x_clean)
        x_clean = correct_iq_imbalance_gsop(x_clean)
        x_clean = normalize_agc(x_clean)
    except Exception as e:
        return {"status": f"ERROR_IQ_CORRECTION: {e}", "stage": "INGESTION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    snr_db = estimate_snr_m2m4(x_clean)
    if snr_db < -5.0:
        return {
            "status": "NO_SIGNAL_DETECTED",
            "stage": "SPECTRAL_ANALYSIS",
            "format": fmt,
            "fs_hz": float(fs),
            "cfo_hz": 0.0,
            "baud_rate": 0.0,
            "sps": 1.0,
            "modulation": "NOISE",
            "crc_valid": False,
            "payload": b"",
            "entropy": 0.0,
            "payload_class": "NOISE",
            "fec_type": "NONE",
            "interleaver": "NONE",
            "bit_slip": 0,
            "metadata": {"snr_est_db": snr_db}
        }

    # Stage 2: Coarse CFO & Clock Recovery
    try:
        cfo_hz, rs, sps, x_bb = estimate_cfo_and_baud(x_clean, fs=fs)
        if rs < 100 or rs > fs / 2:
            # Baud rate unreasonable, try alternative estimation
            rs = 1000.0
            sps = fs / rs
    except Exception as e:
        return {"status": f"ERROR_SPECTRAL: {e}", "stage": "SPECTRAL", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}
        
    # Stage 3: Equalization & Fine Timing
    try:
        if sps <= 1.25 or len(x_clean) < 512:
            pwr = float(np.mean(np.abs(x_bb)**2))
            y_syms = (x_bb / np.sqrt(pwr + 1e-12)).astype(np.complex64)
        else:
            x_2sps = resample_to_2sps(x_bb, sps)
            y_syms = apply_twopass_cma(x_2sps)
            if len(y_syms) == 0:
                y_syms = x_2sps[::2]
    except Exception as e:
        return {"status": f"ERROR_EQUALIZATION: {e}", "stage": "EQUALIZATION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    # Stage 4: Modulation Classification
    try:
        features = extract_features(y_syms)
        mod_type, ood_dist = evaluate_mahalanobis_ood(features, symbols=x_bb)
    except Exception as e:
        return {"status": f"ERROR_CLASSIFICATION: {e}", "stage": "CLASSIFICATION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    if mod_type == "UNKNOWN_MODULATION" or ood_dist > 22.46:
        return {
            "status": "UNKNOWN_MODULATION",
            "stage": "CLASSIFICATION",
            "format": fmt,
            "fs_hz": float(fs),
            "cfo_hz": float(cfo_hz),
            "baud_rate": float(rs),
            "sps": float(sps),
            "modulation": "UNKNOWN_MODULATION",
            "ood_dist": float(ood_dist),
            "crc_valid": False,
            "payload": b"",
            "entropy": 0.0,
            "payload_class": "UNKNOWN",
            "fec_type": "NONE",
            "interleaver": "NONE",
            "bit_slip": 0,
            "metadata": {"ood_dist": float(ood_dist)}
        }

    # Stage 5: Synchronization & Carrier PLL
    try:
        fsk_bits = None
        payload_raw = np.array([], dtype=np.complex64)
        if mod_type in ["2-FSK", "4-FSK"]:
            num_tones = 4 if mod_type == "4-FSK" else 2
            fsk_bits = demodulate_fsk(x_bb, sps=sps, num_tones=num_tones)
            payload_raw, start_idx, sync_metric, phase_rad = resolve_sync_and_rotation(y_syms, mod_type="QPSK")
            s_locked = track_carrier_pll(payload_raw, mod_type="QPSK")
            payload_syms = s_locked
        else:
            payload_raw, start_idx, sync_metric, phase_rad = resolve_sync_and_rotation(y_syms, mod_type=mod_type)
            s_locked = track_carrier_pll(payload_raw, mod_type=mod_type)
            payload_syms = s_locked
    except Exception as e:
        return {"status": f"ERROR_SYNCHRONIZATION: {e}", "stage": "SYNCHRONIZATION", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

    # Stage 6: Blackboard Multi-Hypothesis Arbitration
    try:
        ranked_candidates = [m[0] for m in rank_modulation_hypotheses(features)]
        arbiter = BlackboardArbiter()
        results = arbiter.evaluate_stream(
            symbols=payload_syms if len(payload_syms) > 0 else s_locked,
            mod_type=mod_type,
            fsk_bits=fsk_bits,
            pre_pll_symbols=payload_raw,
            ranked_mods=ranked_candidates
        )
    except Exception as e:
        return {"status": f"ERROR_ARBITER: {e}", "stage": "ARBITER", "format": fmt, "fs_hz": float(fs), "crc_valid": False, "payload": b""}

"""

new_content = content[:start_idx] + new_logic + content[end_idx:]

with open('src/pipeline.py', 'w') as f:
    f.write(new_content)

print("Pipeline updated.")
