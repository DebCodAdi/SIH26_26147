"""
Full-pipeline evaluation harness for PS 26147.

Generates a reproducible synthetic corpus with tests/testbench_gen.py, runs the complete
production pipeline (src/pipeline.run_full_pipeline) over every file, and prints a plain-text
report that is diffable between runs.

This is a MEASUREMENT TOOL ONLY. It does not modify or tune anything in src/.

    python -m tools.evaluate --n 20                     # quick check
    python -m tools.evaluate --n 200 --workers 8        # full run
    python -m tools.evaluate --n 200 --out results.csv  # per-file rows for inspection
    python -m tools.evaluate --seed 26147               # fixed corpus, reproducible

Corpus construction notes (limits of the generator, not of this harness):
  * tests/testbench_gen.py applies interleaving ONLY for the RS and CONCAT FEC paths; the
    interleaver_depth argument is ignored for NONE, CONV and LDPC. Ground truth records the
    interleaver that was actually applied, not the one requested.
  * The generator implements BLOCK interleaving only (interleave_block). It has no
    convolutional interleaver, so no CONV-interleaved files can be produced. Those rows are
    absent from the corpus rather than being faked.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULATIONS = ["BPSK", "QPSK", "8PSK", "16-QAM", "2-FSK", "4-FSK"]
SNRS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
FEC_TYPES = ["NONE", "CONV", "RS", "CONCAT", "LDPC"]
SPS_VALUES = [2, 4, 8]
FS_CHOICES = [200000.0, 96000.0, 250000.0, 48000.0]
CFO_FRACTIONS = [0.0, 0.003, -0.003, 0.01, -0.01]
PAYLOAD_LENGTHS = [24, 31, 40, 53]

# FEC label as the generator understands it -> label the pipeline reports on a successful lock.
FEC_EXPECTED_LABEL = {
    "NONE": "NONE",
    "CONV": "CONV_K7_R1/2",
    "RS": "RS(255,223)",
    "CONCAT": "CONV+RS(255,223)",
    "LDPC": "LDPC_N648_K324",
}
# Interleaving is only actually applied by the generator for these FEC paths.
FEC_SUPPORTING_INTERLEAVE = {"RS", "CONCAT"}


def normalise_interleaver(label: str) -> str:
    """Collapses the pipeline's interleaver labels to a family: NONE / BLOCK / OTHER."""
    if not label or label == "NONE":
        return "NONE"
    if label.startswith("BLOCK_M") or label.startswith("BIT_BLOCK"):
        return "BLOCK"
    return "OTHER"


def build_corpus(n: int, seed: int) -> list[dict]:
    """Deterministically builds the corpus specification. Same seed => same corpus."""
    rng = np.random.default_rng(seed)
    corpus = []
    for i in range(n):
        # Modulation cycles so every class is represented evenly regardless of n.
        mod = MODULATIONS[i % len(MODULATIONS)]
        snr = float(rng.choice(SNRS))
        fec = str(rng.choice(FEC_TYPES))
        sps = int(rng.choice(SPS_VALUES))
        fs = float(rng.choice(FS_CHOICES))
        baud = fs / sps
        cfo = float(rng.choice(CFO_FRACTIONS)) * baud
        payload_len = int(rng.choice(PAYLOAD_LENGTHS))
        requested_depth = int(rng.choice([1, 4, 8]))

        # Record the interleaver that will ACTUALLY be applied by the generator.
        if fec in FEC_SUPPORTING_INTERLEAVE and requested_depth > 1:
            true_interleaver = "BLOCK"
            depth = requested_depth
        else:
            true_interleaver = "NONE"
            depth = 1

        payload = bytes(int(v) for v in rng.integers(32, 127, size=payload_len))
        corpus.append({
            "index": i,
            "modulation": mod,
            "snr_db": snr,
            "fec": fec,
            "interleaver": true_interleaver,
            "interleaver_depth": depth,
            "fs_hz": fs,
            "baud_hz": baud,
            "sps": float(sps),
            "cfo_hz": cfo,
            "payload_hex": payload.hex(),
            "file_seed": int(rng.integers(0, 2**31 - 1)),
        })
    return corpus


def evaluate_one(spec: dict) -> dict:
    """Generates one capture, runs the full pipeline, returns per-file measurements."""
    from tests.testbench_gen import generate_test_vector
    from src.pipeline import run_full_pipeline

    result = {
        "index": spec["index"],
        "true_mod": spec["modulation"],
        "snr_db": spec["snr_db"],
        "true_fec": spec["fec"],
        "true_interleaver": spec["interleaver"],
        "true_fs": spec["fs_hz"],
        "true_baud": spec["baud_hz"],
        "true_sps": spec["sps"],
        "true_cfo": spec["cfo_hz"],
        "pred_mod": "", "pred_fec": "", "pred_interleaver": "",
        "est_fs": float("nan"), "est_baud": float("nan"),
        "est_sps": float("nan"), "est_cfo": float("nan"),
        "crc_valid": False, "payload_match": False,
        "latency_s": float("nan"), "status": "", "exception": "",
    }

    payload = bytes.fromhex(spec["payload_hex"])
    path = None
    t0 = time.perf_counter()
    try:
        rx, _ = generate_test_vector(
            mod_type=spec["modulation"], fec_type=spec["fec"],
            interleaver_depth=spec["interleaver_depth"], payload_text=payload,
            fs=spec["fs_hz"], baud_rate=spec["baud_hz"], cfo_hz=spec["cfo_hz"],
            snr_db=spec["snr_db"], seed=spec["file_seed"],
        )
        with tempfile.NamedTemporaryFile(suffix=".iq", delete=False) as fh:
            path = fh.name
            fh.write(np.asarray(rx, dtype=np.complex64).tobytes())

        res = run_full_pipeline(path, user_fs=spec["fs_hz"])
        result.update({
            "pred_mod": str(res.get("modulation", "")),
            "pred_fec": str(res.get("fec_type", "")),
            "pred_interleaver": str(res.get("interleaver", "")),
            "est_fs": float(res.get("fs_hz", float("nan"))),
            "est_baud": float(res.get("baud_rate", float("nan"))),
            "est_sps": float(res.get("sps", float("nan"))),
            "est_cfo": float(res.get("cfo_hz", float("nan"))),
            "crc_valid": bool(res.get("crc_valid", False)),
            "payload_match": bool(res.get("payload", b"") == payload),
            "status": str(res.get("status", "")),
        })
    except Exception:
        # A failure on one file must never abort the run; record and continue.
        result["exception"] = traceback.format_exc(limit=3).strip().replace("\n", " | ")
        result["status"] = "EXCEPTION"
    finally:
        result["latency_s"] = time.perf_counter() - t0
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
    return result


def pct(num: int, den: int) -> str:
    return f"{(100.0 * num / den) if den else 0.0:5.1f}%"


def fmt_err(values: list[float]) -> tuple[str, str]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return "n/a", "n/a"
    return f"{np.median(arr):.2f}", f"{np.percentile(arr, 90):.2f}"


def print_report(rows: list[dict], wall_s: float, seed: int, n: int) -> None:
    N = len(rows)
    ok_amc = sum(1 for r in rows if r["pred_mod"] == r["true_mod"])
    ok_fec = sum(1 for r in rows
                 if r["pred_fec"] == FEC_EXPECTED_LABEL.get(r["true_fec"], "?"))
    ok_crc = sum(1 for r in rows if r["crc_valid"] and r["payload_match"])
    n_exc = sum(1 for r in rows if r["exception"])
    lat = [r["latency_s"] for r in rows if np.isfinite(r["latency_s"])]

    amc_p = 100.0 * ok_amc / N if N else 0.0
    fec_p = 100.0 * ok_fec / N if N else 0.0
    crc_p = 100.0 * ok_crc / N if N else 0.0
    combined = (amc_p + fec_p + crc_p) / 3.0

    print("=" * 78)
    print(f"  PS 26147 PIPELINE EVALUATION REPORT   (seed={seed}, n={n})")
    print("=" * 78)
    print()
    print("1. Overall")
    print(f"   Modulation Classification (AMC)     {ok_amc:>4}/{N:<5} {pct(ok_amc, N)}")
    print(f"   FEC Code Identification             {ok_fec:>4}/{N:<5} {pct(ok_fec, N)}")
    print(f"   CRC-32 Locked Decodes               {ok_crc:>4}/{N:<5} {pct(ok_crc, N)}")
    print(f"   Pipeline Crashes / Exceptions       {n_exc:>4}/{N:<5} {pct(n_exc, N)}")
    print(f"   Combined Score                        --      {combined:5.1f}%")
    print()
    print("   Combined Score = unweighted mean of the three rates above")
    print("   (AMC + FEC identification + CRC-locked decode) / 3.")
    print("   Crashes are counted as failures in all three, never excluded.")
    print("   FEC identification is only reported by the pipeline on a CRC lock, so it")
    print("   cannot exceed the decode rate except on genuinely uncoded (NONE) files.")
    print()
    print(f"   Total Files Tested    : {N}")
    print(f"   Total Wall-Clock Time : {wall_s:.2f} s")
    print(f"   Median Latency / File : {np.median(lat) * 1000.0 if lat else 0.0:.1f} ms")
    print()

    print("2. Per-Modulation Classification Breakdown")
    print("   " + "-" * 62)
    print(f"   {'Modulation':<12}{'Correct/Total':>16}{'Accuracy':>12}{'Rating':>10}")
    print("   " + "-" * 62)
    for mod in MODULATIONS:
        sub = [r for r in rows if r["true_mod"] == mod]
        if not sub:
            continue
        c = sum(1 for r in sub if r["pred_mod"] == mod)
        acc = 100.0 * c / len(sub)
        rating = "[OK]" if acc >= 85.0 else ("[~~]" if acc >= 60.0 else "[XX]")
        print(f"   {mod:<12}{f'{c}/{len(sub)}':>16}{acc:>11.1f}%{rating:>10}")
    print("   " + "-" * 62)
    print("   Rating: [OK] >=85%   [~~] 60-85%   [XX] <60%")
    print()

    print("3. Per-SNR Breakdown")
    print("   " + "-" * 68)
    print(f"   {'SNR':>6}{'AMC':>12}{'Decode':>12}{'med Bauderr':>16}{'med CFOerr':>15}")
    print("   " + "-" * 68)
    for snr in SNRS:
        sub = [r for r in rows if r["snr_db"] == snr]
        if not sub:
            continue
        a = sum(1 for r in sub if r["pred_mod"] == r["true_mod"])
        d = sum(1 for r in sub if r["crc_valid"] and r["payload_match"])
        be = [abs(r["est_baud"] - r["true_baud"]) for r in sub]
        ce = [abs(r["est_cfo"] - r["true_cfo"]) for r in sub]
        bm, _ = fmt_err(be)
        cm, _ = fmt_err(ce)
        print(f"   {snr:>5.0f}{pct(a, len(sub)):>12}{pct(d, len(sub)):>12}"
              f"{bm + ' Hz':>16}{cm + ' Hz':>15}")
    print("   " + "-" * 68)
    print()

    print("4. Confusion Matrix  (rows = true, columns = predicted)")
    preds = sorted({r["pred_mod"] for r in rows if r["pred_mod"]})
    counts: dict = defaultdict(Counter)
    for r in rows:
        counts[r["true_mod"]][r["pred_mod"] or "(none)"] += 1
    col_labels = preds + (["(none)"] if any(not r["pred_mod"] for r in rows) else [])
    header = f"   {'true \\ pred':<12}" + "".join(f"{c:>10}" for c in col_labels)
    print("   " + "-" * (len(header) - 3))
    print(header)
    print("   " + "-" * (len(header) - 3))
    for mod in MODULATIONS:
        if mod not in counts:
            continue
        line = f"   {mod:<12}" + "".join(f"{counts[mod].get(c, 0):>10}" for c in col_labels)
        print(line)
    print("   " + "-" * (len(header) - 3))
    print()

    print("5. Parameter Accuracy  (absolute error)")
    print("   " + "-" * 58)
    print(f"   {'Parameter':<14}{'Median':>16}{'90th pct':>16}")
    print("   " + "-" * 58)
    for label, key_est, key_true, unit in (
        ("Fs", "est_fs", "true_fs", "Hz"),
        ("Baud", "est_baud", "true_baud", "Hz"),
        ("SPS", "est_sps", "true_sps", ""),
        ("CFO", "est_cfo", "true_cfo", "Hz"),
    ):
        errs = [abs(r[key_est] - r[key_true]) for r in rows]
        med, p90 = fmt_err(errs)
        suffix = f" {unit}" if unit else ""
        print(f"   {label:<14}{med + suffix:>16}{p90 + suffix:>16}")
    print("   " + "-" * 58)
    print()
    print("   Fs is supplied to the pipeline (user_fs) for every file in this corpus,")
    print("   so its error reflects pass-through fidelity, not blind estimation.")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="PS 26147 full-pipeline evaluation harness")
    ap.add_argument("--n", type=int, default=200, help="number of corpus files")
    ap.add_argument("--seed", type=int, default=26147, help="master seed (reproducible corpus)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", type=str, default=None, help="write per-file results to this CSV")
    ap.add_argument("--manifest", type=str, default="benchmarks/corpus_manifest.json")
    args = ap.parse_args()

    corpus = build_corpus(args.n, args.seed)
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump({"seed": args.seed, "n": args.n, "files": corpus}, fh, indent=2)

    t0 = time.perf_counter()
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers) as pool:
            rows = pool.map(evaluate_one, corpus, chunksize=1)
    else:
        rows = [evaluate_one(spec) for spec in corpus]
    wall = time.perf_counter() - t0

    rows.sort(key=lambda r: r["index"])   # order-independent of worker scheduling

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print_report(rows, wall, args.seed, args.n)
    if args.out:
        print(f"\nPer-file results written to {args.out}")
    print(f"Corpus manifest written to {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
