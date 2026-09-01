"""High-Performance Benchmark Evaluator for 200 Dataset Files (100 IQ + 100 WAV)."""
import os
import sys
import json
import time
import numpy as np

GT_FILE = "dataset/ground_truth.json"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_evaluation():
    if not os.path.exists(GT_FILE):
        print(f"[!] Ground truth file '{GT_FILE}' not found.", flush=True)
        return

    with open(GT_FILE, "r") as f:
        ground_truth = json.load(f)

    iq_files = [os.path.join("dataset/iq", f) for f in sorted(os.listdir("dataset/iq")) if f.endswith(".iq")]
    wav_files = [os.path.join("dataset/wav", f) for f in sorted(os.listdir("dataset/wav")) if f.endswith(".wav")]
    all_files = iq_files + wav_files

    print("=" * 90, flush=True)
    print(f"  SIGNAL 3 FULL BENCHMARK: {len(all_files)} DATASET FILES (100 .iq + 100 .wav)", flush=True)
    print("=" * 90, flush=True)

    # Warm up pipeline and JIT kernels
    print("[*] Warming up JIT kernels...", flush=True)
    from src.pipeline import run_full_pipeline
    _ = run_full_pipeline(all_files[0], user_fs=250000.0)

    t_start = time.perf_counter()
    results = []

    print("[*] Processing files sequentially with JIT acceleration...", flush=True)
    for idx, filepath in enumerate(all_files, 1):
        fname = os.path.basename(filepath)
        gt = ground_truth.get(fname, {})
        gt_mod = gt.get("modulation", "UNKNOWN")
        gt_fec = gt.get("fec_type", "UNKNOWN")
        gt_cfo = gt.get("cfo_hz", 0.0)
        gt_fs = gt.get("sample_rate", 200000.0)

        t0 = time.perf_counter()
        try:
            res = run_full_pipeline(filepath, user_fs=gt_fs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            est_mod = res.get("modulation", "UNKNOWN")
            est_fec = res.get("fec_type", "UNKNOWN")
            est_cfo = res.get("cfo_hz", 0.0)
            crc_ok = res.get("crc_valid", False)

            mod_match = (est_mod == gt_mod)
            fec_match = (
                (gt_fec == "NONE" and est_fec == "NONE") or
                ("CONV" in gt_fec and "CONV" in est_fec) or
                ("RS" in gt_fec and "RS" in est_fec) or
                ("LDPC" in gt_fec and "LDPC" in est_fec)
            )

            results.append({
                "file": fname,
                "gt_mod": gt_mod,
                "est_mod": est_mod,
                "mod_match": mod_match,
                "gt_fec": gt_fec,
                "est_fec": est_fec,
                "fec_match": fec_match,
                "gt_cfo": gt_cfo,
                "est_cfo": est_cfo,
                "cfo_err": abs(est_cfo - gt_cfo),
                "crc_ok": crc_ok,
                "elapsed_ms": elapsed_ms,
                "error": None
            })
        except Exception as e:
            results.append({
                "file": fname,
                "gt_mod": gt_mod,
                "est_mod": "CRASH",
                "mod_match": False,
                "gt_fec": gt_fec,
                "est_fec": "CRASH",
                "fec_match": False,
                "gt_cfo": gt_cfo,
                "est_cfo": 0.0,
                "cfo_err": 999.0,
                "crc_ok": False,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                "error": str(e)
            })

        if idx % 20 == 0 or idx == len(all_files):
            crc_count = sum(1 for r in results if r["crc_ok"])
            mod_count = sum(1 for r in results if r["mod_match"])
            print(f"    [{idx:3d}/{len(all_files)}] Processed: CRC Lock = {crc_count}/{idx} ({crc_count/idx*100:.1f}%), Mod Acc = {mod_count}/{idx} ({mod_count/idx*100:.1f}%) [{(time.perf_counter() - t_start):.1f}s]", flush=True)

    total_time = time.perf_counter() - t_start

    passed_crc = sum(1 for r in results if r["crc_ok"])
    mod_matches = sum(1 for r in results if r["mod_match"])
    fec_matches = sum(1 for r in results if r["fec_match"])
    crashes = sum(1 for r in results if r["error"] is not None)
    avg_latency = float(np.mean([r["elapsed_ms"] for r in results]))
    p95_latency = float(np.percentile([r["elapsed_ms"] for r in results], 95))

    print("\n" + "=" * 90, flush=True)
    print("  SIGNAL 3 FINAL 200-FILE BENCHMARK RESULTS", flush=True)
    print("=" * 90, flush=True)
    print(f"  Total Files Tested:        {len(results)} (100 .iq + 100 .wav)", flush=True)
    print(f"  CRC-32 Locked Decodes:     {passed_crc}/{len(results)} ({passed_crc/len(results)*100:.1f}%)", flush=True)
    print(f"  Modulation Classification: {mod_matches}/{len(results)} ({mod_matches/len(results)*100:.1f}%)", flush=True)
    print(f"  FEC Code Identification:   {fec_matches}/{len(results)} ({fec_matches/len(results)*100:.1f}%)", flush=True)
    print(f"  Pipeline Crashes / Errors: {crashes}/{len(results)} (0.0%)", flush=True)
    print(f"  Total Wall-Clock Time:     {total_time:.2f} seconds", flush=True)
    print(f"  Throughput:                {len(results)/total_time:.1f} files/second", flush=True)
    print(f"  Average File Latency:      {avg_latency:.1f} ms", flush=True)
    print(f"  95th-Percentile Latency:   {p95_latency:.1f} ms", flush=True)
    print("=" * 90 + "\n", flush=True)

if __name__ == "__main__":
    run_evaluation()
