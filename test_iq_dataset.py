"""Benchmark evaluator on c:/Users/adity/Downloads/iq_test_dataset."""
import os
import sys
import json
from src.pipeline import run_full_pipeline

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

manifest_path = "c:/Users/adity/Downloads/iq_test_dataset/manifest.json"
with open(manifest_path, "r") as f:
    manifest = json.load(f)

print("=" * 100)
print("  BENCHMARK ON iq_test_dataset (15 CHALLENGE FILES)")
print("=" * 100)

for item in manifest:
    fname = item["filename"]
    fpath = os.path.join("c:/Users/adity/Downloads/iq_test_dataset", fname)
    gt_fs = item["sample_rate_hz"]
    gt_mod = item["modulation"]
    gt_baud = item["baud_rate"]
    gt_sps = item["sps"]
    
    res = run_full_pipeline(fpath)
    est_fs = res.get("fs_hz", 0.0)
    est_mod = res.get("modulation", "UNKNOWN")
    est_sps = res.get("sps", 0.0)
    status = res.get("status", "UNKNOWN")
    payload = res.get("payload_ascii", "")
    
    print(f"{fname:30s} | GT Fs: {gt_fs:6d} | Est Fs: {est_fs:8.1f} | GT Mod: {gt_mod:6s} | Est Mod: {est_mod:6s} | Status: {status}")
    if payload:
        print(f"   Payload: {payload[:50]}")
