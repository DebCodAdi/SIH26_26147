"""Batch Dataset Verifier for PS 26147 Blind SDR Interceptor.

Evaluates test files in dataset/iq/ and dataset/wav/ against dataset/ground_truth.json
and prints formatted side-by-side ground truth vs model result tables.
"""
import os
import sys
import json
import argparse
from src.pipeline import run_full_pipeline

def verify_dataset(
    folder: str = "dataset/iq",
    gt_file: str = "dataset/ground_truth.json",
    limit: int = 100,
    verbose: bool = True
):
    if not os.path.exists(gt_file):
        print(f"[!] Ground truth file '{gt_file}' not found. Please run 'python generate_dataset.py' first.")
        return

    with open(gt_file, "r") as f:
        ground_truth = json.load(f)

    if not os.path.exists(folder):
        print(f"[!] Folder '{folder}' not found.")
        return

    files = sorted([f for f in os.listdir(folder) if f.endswith(('.iq', '.wav', '.raw', '.cf32'))])
    if limit:
        files = files[:limit]

    print("=" * 85)
    print(f"  PS 26147 BATCH TESTBENCH: Evaluating {len(files)} files in '{folder}'")
    print("=" * 85)

    passed_crc = 0
    mod_matches = 0
    fec_matches = 0
    total = len(files)

    for idx, fname in enumerate(files, 1):
        fpath = os.path.join(folder, fname)
        gt = ground_truth.get(fname, {})
        gt_mod = gt.get("modulation", "UNKNOWN")
        gt_fec = gt.get("fec_type", "UNKNOWN")
        gt_cfo = gt.get("cfo_hz", 0.0)
        gt_fs = gt.get("sample_rate", 200000.0)
        gt_hex = gt.get("payload_hex", "")
        gt_ascii = gt.get("payload_ascii", "")

        try:
            res = run_full_pipeline(fpath, user_fs=gt_fs)
            est_mod = res.get("modulation", "UNKNOWN")
            est_fec = res.get("fec_type", "UNKNOWN")
            est_cfo = res.get("cfo_hz", 0.0)
            crc_ok = res.get("crc_valid", False)
            res_payload = res.get("payload", b"")
            res_hex = res_payload.hex() if isinstance(res_payload, bytes) else ""

            mod_match = (est_mod == gt_mod)
            if mod_match: mod_matches += 1

            if crc_ok:
                passed_crc += 1
                status_str = "[PASS - CRC LOCKED]"
            else:
                status_str = "[FAIL - NO CRC]"

            if verbose:
                print(f"\n--- File {idx}/{total}: {fname} {status_str} ---")
                print(f"  Ground Truth: Mod={gt_mod:<8} FEC={gt_fec:<16} CFO={gt_cfo:>7.1f}Hz  Fs={gt_fs/1e3:.0f}kHz")
                print(f"  Model Result: Mod={est_mod:<8} FEC={est_fec:<16} CFO={est_cfo:>7.1f}Hz  CRC={crc_ok}")
                if gt_ascii:
                    print(f"  GT Payload:   {gt_ascii[:65]}...")
                if res.get("payload_ascii"):
                    print(f"  Est Payload:  {res.get('payload_ascii')[:65]}...")

        except Exception as e:
            print(f"[!] Error on {fname}: {e}")

    print("\n" + "=" * 85)
    print(f"  BATCH VERIFICATION SUMMARY ({total} Files Tested)")
    print("=" * 85)
    print(f"  CRC-32 Locked Decodes:  {passed_crc}/{total} ({passed_crc/total*100:.1f}%)")
    print(f"  Modulation Matches:     {mod_matches}/{total} ({mod_matches/total*100:.1f}%)")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify generated .iq and .wav dataset.")
    parser.add_argument("--folder", "-d", type=str, default="dataset/iq", help="Folder containing .iq or .wav files")
    parser.add_argument("--limit", "-n", type=int, default=10, help="Number of files to test (default: 10)")
    parser.add_argument("--all", action="store_true", help="Test all files in the folder")
    args = parser.parse_args()

    num = 100 if args.all else args.limit
    verify_dataset(folder=args.folder, limit=num)
