"""
Low-SNR failure attribution (Phase 1) and oracle ablation (Phase 2) for PS 26147.

MEASUREMENT ONLY. Imports src/ and tests/ but modifies nothing.

Phase 1 walks every corpus file at or below a cutoff SNR through the production receive chain,
comparing each stage's output against ground truth, and attributes the failure to the FIRST
stage that goes out of tolerance. That ordering matters: a wrong symbol rate corrupts every
later stage, so attributing such a file to "demodulation" would be misleading.

Phase 2 re-runs the same files substituting ground-truth values for individual parameters, to
separate "we estimated the parameters wrongly" from "the signal is not recoverable even with
perfect knowledge".

    python -m tools.lowsnr_analysis --max-snr 15
    python -m tools.lowsnr_analysis --max-snr 15 --phase 2 --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGES = [
    "1_burst_detection",
    "2_snr_estimation",
    "3_modulation_classification",
    "4_sample_rate",
    "5_sps_estimation",
    "6_baud_estimation",
    "7_coarse_cfo",
    "8_fine_cfo_phase",
    "10_timing_recovery",
    "12_demodulation",
    "13_fec_decoding",
    "14_crc_validation",
    "OK_decoded",
]

PSK_QAM = {"BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM"}
BITS_PER_SYM = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16-QAM": 4, "64-QAM": 6,
                "2-FSK": 1, "4-FSK": 2}


def load_corpus(path: str, max_snr: float) -> list[dict]:
    with open(path) as fh:
        m = json.load(fh)
    return [f for f in m["files"] if f["snr_db"] <= max_snr]


def aligned_ber(bits: np.ndarray, truth: np.ndarray) -> float:
    """Alignment-invariant BER (best over all lags)."""
    if bits is None or len(bits) < len(truth) or len(truth) == 0:
        return 1.0
    d = 2.0 * bits.astype(np.float32) - 1.0
    t = 2.0 * truth.astype(np.float32) - 1.0
    c = np.correlate(d, t, mode="valid")
    return float((len(truth) - float(np.max(c))) / 2.0 / len(truth))


def aligned_ser(bits: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Symbol error rate: fraction of k-bit groups containing >=1 bit error, best alignment."""
    if bits is None or len(bits) < len(truth) or len(truth) == 0 or k <= 0:
        return 1.0
    d = 2.0 * bits.astype(np.float32) - 1.0
    t = 2.0 * truth.astype(np.float32) - 1.0
    c = np.correlate(d, t, mode="valid")
    lag = int(np.argmax(c))
    seg = bits[lag:lag + len(truth)]
    n = (len(truth) // k) * k
    if n == 0:
        return 1.0
    err = (seg[:n] != truth[:n]).reshape(-1, k).any(axis=1)
    return float(np.mean(err))


def receive(rx, fs, spec, truth_bits, *, oracle=frozenset()):
    """
    Runs the production receive chain with optional ground-truth substitutions.
    Returns a dict of stage outputs plus BER/SER/CRC.
    """
    from src.iq_correction import remove_dc_offset_iir, correct_iq_imbalance_gsop, normalize_agc
    from src.ingestion import detect_signal_mme
    from src.spectral import estimate_cfo_and_baud, resample_to_2sps
    from src.equalizers import apply_twopass_cma, gram_schmidt_iq_balance, apply_gardner_ted
    from src.classifier import extract_features, evaluate_mahalanobis_ood
    from src.synchronizer import resolve_sync_and_rotation, track_carrier_pll
    from src.demodulators import demap_linear, demodulate_fsk
    from src.blackboard import BlackboardArbiter

    out = {}
    x = normalize_agc(correct_iq_imbalance_gsop(remove_dc_offset_iir(rx)))
    present, mme = detect_signal_mme(x)
    out["detected"] = bool(present)
    out["mme"] = float(mme)

    diag: dict = {}
    cfo_e, rs_e, sps_e, _ = estimate_cfo_and_baud(x, fs=fs, diag=diag)
    out["est_cfo"], out["est_baud"], out["est_sps"] = cfo_e, rs_e, sps_e
    out["baud_conf"] = float(diag.get("baud_confidence", 0.0))

    cfo = spec["cfo_hz"] if "cfo" in oracle else cfo_e
    sps = spec["sps"] if "baud" in oracle else sps_e

    n = np.arange(len(x))
    x_bb = (x * np.exp(-1j * 2.0 * np.pi * cfo * n / fs)).astype(np.complex64)

    true_mod = spec["modulation"]
    k_bits = BITS_PER_SYM.get(true_mod, 2)

    # ---- symbol extraction -------------------------------------------------------------
    if sps <= 1.25 or len(x) < 512:
        pwr = float(np.mean(np.abs(x_bb) ** 2))
        cand_syms = [(x_bb / np.sqrt(pwr + 1e-12)).astype(np.complex64)]
    else:
        x2 = resample_to_2sps(x_bb, sps)
        if "timing" in oracle:
            # Upper bound on timing recovery: try both 2-SPS strobes, keep the best.
            bal = gram_schmidt_iq_balance(x2)
            p = float(np.mean(np.abs(bal) ** 2))
            bal = bal / np.sqrt(p) if p > 1e-12 else bal
            cand_syms = [bal[0::2].astype(np.complex64), bal[1::2].astype(np.complex64)]
        else:
            y = apply_twopass_cma(x2)
            cand_syms = [y if len(y) > 0 else x2[::2]]

    # ---- classification ----------------------------------------------------------------
    feats = extract_features(cand_syms[0])
    pred_mod, _ = evaluate_mahalanobis_ood(feats, symbols=x_bb)
    out["pred_mod"] = pred_mod
    mod_used = true_mod if "mod" in oracle else (pred_mod if pred_mod != "UNKNOWN_MODULATION" else "QPSK")
    out["mod_used"] = mod_used

    # ---- demodulation ------------------------------------------------------------------
    best_ber, best_bits = 1.0, None
    if mod_used in ("2-FSK", "4-FSK", "GMSK"):
        tones = 4 if mod_used == "4-FSK" else 2
        bits = demodulate_fsk(x_bb, sps=sps, num_tones=tones)
        best_ber, best_bits = aligned_ber(bits, truth_bits), bits
    else:
        for sym in cand_syms:
            pre, _si, _mt, _ph = resolve_sync_and_rotation(sym, mod_type=mod_used)
            src = pre if len(pre) > 0 else sym
            lock = track_carrier_pll(src, mod_type=mod_used)
            for stream in (lock, src, sym):
                for _lbl, bits in demap_linear(stream, mod_type=mod_used):
                    b = aligned_ber(bits, truth_bits)
                    if b < best_ber:
                        best_ber, best_bits = b, bits
    out["ber"] = best_ber
    out["ser"] = aligned_ser(best_bits, truth_bits, k_bits) if best_bits is not None else 1.0

    # ---- FEC + CRC via the real arbiter -------------------------------------------------
    payload = bytes.fromhex(spec["payload_hex"])
    arb = BlackboardArbiter()
    crc_ok = False
    try:
        streams = cand_syms if mod_used not in ("2-FSK", "4-FSK", "GMSK") else []
        fsk_bits = best_bits if mod_used in ("2-FSK", "4-FSK", "GMSK") else None
        for sym in (streams or [cand_syms[0]]):
            pre, _si, _mt, _ph = resolve_sync_and_rotation(sym, mod_type=mod_used)
            src = pre if len(pre) > 0 else sym
            lock = track_carrier_pll(src, mod_type=mod_used)
            res = arb.evaluate_stream(symbols=lock, mod_type=mod_used, fsk_bits=fsk_bits,
                                      pre_pll_symbols=src, ranked_mods=None,
                                      aux_symbol_streams=[sym])
            if res and res[0].crc_valid and res[0].payload == payload:
                crc_ok = True
                break
    except Exception:
        pass
    out["crc"] = crc_ok
    return out


def attribute(spec, obs) -> str:
    """First stage that goes out of tolerance, in pipeline order."""
    if obs["crc"]:
        return "OK_decoded"
    if not obs["detected"]:
        return "1_burst_detection"

    # Sample rate is supplied to the pipeline for this corpus, so stage 4 cannot fail here.
    baud_rel = abs(obs["est_baud"] - spec["baud_hz"]) / max(spec["baud_hz"], 1e-9)
    if baud_rel > 0.01:
        return "6_baud_estimation"
    if abs(obs["est_sps"] - spec["sps"]) > 0.1:
        return "5_sps_estimation"

    # A carrier error matters when it rotates the constellation appreciably across the burst.
    # Tolerance: a quarter cycle over the record length.
    rec_s = spec.get("_record_s", 0.0)
    cfo_tol = 0.25 / rec_s if rec_s > 0 else 50.0
    if abs(obs["est_cfo"] - spec["cfo_hz"]) > cfo_tol:
        return "7_coarse_cfo"

    if obs["pred_mod"] != spec["modulation"]:
        return "3_modulation_classification"
    if obs["ber"] > 0.2:
        return "10_timing_recovery"
    if obs["ber"] > 0.01:
        return "12_demodulation"
    if spec["fec"] != "NONE":
        return "13_fec_decoding"
    return "14_crc_validation"


def build_one(spec):
    from tests.testbench_gen import generate_test_vector
    truth: dict = {}
    payload = bytes.fromhex(spec["payload_hex"])
    rx, _ = generate_test_vector(
        mod_type=spec["modulation"], fec_type=spec["fec"],
        interleaver_depth=spec["interleaver_depth"], payload_text=payload,
        fs=spec["fs_hz"], baud_rate=spec["baud_hz"], cfo_hz=spec["cfo_hz"],
        snr_db=spec["snr_db"], seed=spec["file_seed"], truth_out=truth)
    tb = truth.get("coded_bits")
    if tb is None:   # FSK path returns before truth_out is populated
        tb = np.zeros(0, dtype=np.uint8)
    return rx, tb


def phase1_worker(spec):
    rx, tb = build_one(spec)
    spec = dict(spec)
    spec["_record_s"] = len(rx) / spec["fs_hz"]
    obs = receive(rx, spec["fs_hz"], spec, tb)
    return {"index": spec["index"], "snr": spec["snr_db"], "mod": spec["modulation"],
            "fec": spec["fec"], "stage": attribute(spec, obs),
            "ber": obs["ber"], "ser": obs["ser"], "crc": obs["crc"],
            "baud_conf": obs["baud_conf"], "has_truth": len(tb) > 0}


ORACLES = [
    ("A_automatic", frozenset()),
    ("B_oracle_mod", frozenset({"mod"})),
    ("C_oracle_baud_sps", frozenset({"baud"})),
    ("D_oracle_cfo", frozenset({"cfo"})),
    ("E_oracle_timing", frozenset({"timing"})),
    ("F_mod_plus_sync", frozenset({"mod", "cfo", "timing"})),
    ("G_all_params", frozenset({"mod", "baud", "cfo", "timing"})),
]


def phase2_worker(spec):
    rx, tb = build_one(spec)
    spec2 = dict(spec)
    spec2["_record_s"] = len(rx) / spec["fs_hz"]
    row = {"index": spec["index"], "snr": spec["snr_db"], "mod": spec["modulation"],
           "has_truth": len(tb) > 0}
    for name, orc in ORACLES:
        o = receive(rx, spec["fs_hz"], spec2, tb, oracle=orc)
        row[name] = {"ber": o["ber"], "ser": o["ser"], "crc": o["crc"]}
    return row


def snr_band(s: float) -> str:
    if s <= 5.0:
        return "0-5 dB"
    if s <= 10.0:
        return "5-10 dB"
    return "10-15 dB"


def report_phase1(rows):
    N = len(rows)
    print("=" * 78)
    print("  PHASE 1 - LOW-SNR FAILURE ATTRIBUTION")
    print("=" * 78)
    print(f"  Files analysed (SNR <= cutoff): {N}")
    print("  Attribution = FIRST stage out of tolerance, in pipeline order.")
    print("  Tolerances: baud 1% relative | SPS 0.1 | CFO a quarter cycle over the record")
    print("              | AMC exact | BER>0.2 timing | BER>0.01 demodulation")
    print()
    print("  Failure stage totals")
    print("  " + "-" * 60)
    c = Counter(r["stage"] for r in rows)
    for st in STAGES:
        if c.get(st):
            print(f"   {st:<32}{c[st]:>5}{100.0*c[st]/N:>9.1f}%")
    print("  " + "-" * 60)
    print()
    print("  Failure matrix: stage x SNR band")
    bands = ["0-5 dB", "5-10 dB", "10-15 dB"]
    print("  " + "-" * 60)
    print(f"   {'stage':<32}" + "".join(f"{b:>10}" for b in bands))
    print("  " + "-" * 60)
    grid = defaultdict(Counter)
    for r in rows:
        grid[r["stage"]][snr_band(r["snr"])] += 1
    for st in STAGES:
        if st in grid:
            print(f"   {st:<32}" + "".join(f"{grid[st].get(b,0):>10}" for b in bands))
    print("  " + "-" * 60)
    print()
    print("  Failure matrix: stage x modulation")
    mods = sorted({r["mod"] for r in rows})
    print("  " + "-" * (32 + 10 * len(mods)))
    print(f"   {'stage':<32}" + "".join(f"{m:>10}" for m in mods))
    print("  " + "-" * (32 + 10 * len(mods)))
    gm = defaultdict(Counter)
    for r in rows:
        gm[r["stage"]][r["mod"]] += 1
    for st in STAGES:
        if st in gm:
            print(f"   {st:<32}" + "".join(f"{gm[st].get(m,0):>10}" for m in mods))
    print("  " + "-" * (32 + 10 * len(mods)))
    print()
    print("  Baud confidence (Finding 3 metric) vs outcome")
    for conf in (0.0, 0.5, 1.0):
        sub = [r for r in rows if abs(r["baud_conf"] - conf) < 1e-9]
        if sub:
            dec = sum(1 for r in sub if r["crc"])
            print(f"   conf={conf:.1f}: {len(sub):>4} files, decoded {dec:>3} "
                  f"({100.0*dec/len(sub):5.1f}%)")
    print("=" * 78)


def report_phase2(rows):
    print("=" * 78)
    print("  PHASE 2 - ORACLE ABLATION (what is actually limiting low-SNR performance)")
    print("=" * 78)
    usable = [r for r in rows if r["has_truth"]]
    print(f"  Files: {len(rows)}   with bit-level ground truth: {len(usable)}")
    print("  (the FSK generator path returns before ground-truth bits are recorded,")
    print("   so BER/SER are reported over the linear-modulation subset only)")
    print()
    bands = ["0-5 dB", "5-10 dB", "10-15 dB"]
    for band in bands:
        sub = [r for r in usable if snr_band(r["snr"]) == band]
        sub_all = [r for r in rows if snr_band(r["snr"]) == band]
        if not sub_all:
            continue
        print(f"  {band}   (n={len(sub_all)}, with-truth n={len(sub)})")
        print("  " + "-" * 68)
        print(f"   {'configuration':<22}{'med BER':>10}{'med SER':>10}{'CRC decode':>14}")
        print("  " + "-" * 68)
        for name, _ in ORACLES:
            bers = [r[name]["ber"] for r in sub]
            sers = [r[name]["ser"] for r in sub]
            dec = sum(1 for r in sub_all if r[name]["crc"])
            mb = f"{np.median(bers):.4f}" if bers else "n/a"
            ms = f"{np.median(sers):.4f}" if sers else "n/a"
            print(f"   {name:<22}{mb:>10}{ms:>10}"
                  f"{f'{dec}/{len(sub_all)} ({100.0*dec/len(sub_all):.0f}%)':>14}")
        print("  " + "-" * 68)
        print()
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="benchmarks/corpus_manifest.json")
    ap.add_argument("--max-snr", type=float, default=15.0)
    ap.add_argument("--phase", type=int, default=1, choices=(1, 2))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    corpus = load_corpus(args.manifest, args.max_snr)
    worker = phase1_worker if args.phase == 1 else phase2_worker
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(worker, corpus, chunksize=1)
    else:
        rows = [worker(s) for s in corpus]
    rows.sort(key=lambda r: r["index"])

    if args.phase == 1:
        report_phase1(rows)
    else:
        report_phase2(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
