"""Dataset2 Challenge Case Evaluator with fuzzy filename matching."""
import os
import sys
import json
import glob
import numpy as np

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from src.pipeline import run_full_pipeline

with open("dataset2/expected_results.json", "r") as f:
    expected = json.load(f)

print("=" * 80)
print("EVALUATING DATASET2 CHALLENGE CASES")
print("=" * 80)

passed = 0
failed = 0
skipped = 0

def find_file_on_disk(expected_name, directory="dataset2"):
    """Find a file on disk, handling Windows download renames like (1) suffix."""
    exact_path = os.path.join(directory, expected_name)
    if os.path.exists(exact_path):
        return exact_path
    
    # Try fuzzy matching: strip extension, search for files containing the base name
    base_name = os.path.splitext(expected_name)[0]
    ext = os.path.splitext(expected_name)[1]
    
    for f in os.listdir(directory):
        f_lower = f.lower()
        base_lower = base_name.lower()
        if base_lower in f_lower and f_lower.endswith(ext.lower()):
            return os.path.join(directory, f)
    
    return None

for filename, exp in expected.items():
    filepath = find_file_on_disk(filename)
    if filepath is None:
        print(f"\n[SKIP] {filename} does not exist on disk.")
        skipped += 1
        continue

    actual_fname = os.path.basename(filepath)
    try:
        res = run_full_pipeline(filepath, user_fs=200000.0)
        
        exp_behavior = exp.get("expected_behavior", "")
        got_status = res.get("status", "")
        
        # If no explicit expected_behavior but there's an expected_message with exact match,
        # default expected behavior to FULL_DECODE_SUCCESS
        if not exp_behavior and exp.get("expected_message") and exp.get("expected_exact_match"):
            exp_behavior = "FULL_DECODE_SUCCESS"
        
        # Check behavior match
        behavior_pass = (exp_behavior == got_status) or (not exp_behavior)
        
        # Check modulation match (if specified)
        exp_mod = exp.get("expected_modulation")
        got_mod = res.get("modulation")
        mod_pass = (exp_mod is None) or (got_mod == exp_mod)
        
        # Check message match (if specified)
        exp_msg = exp.get("expected_message")
        got_payload = res.get("payload", b"")
        if isinstance(got_payload, bytes):
            got_msg = got_payload.decode("utf-8", errors="replace") if got_payload else None
        else:
            got_msg = str(got_payload) if got_payload else None
        
        msg_pass = True
        if exp_msg is not None:
            msg_pass = (got_msg == exp_msg)
        elif exp_msg is None and exp.get("expected_exact_match") is not None:
            pass
        
        # Check interleaver (if specified)
        exp_interleaver = exp.get("expected_interleaver")
        got_interleaver = res.get("interleaver", "NONE")
        interleaver_pass = (exp_interleaver is None) or (exp_interleaver.lower() in got_interleaver.lower())
        
        # Overall pass/fail
        case_pass = behavior_pass and mod_pass and msg_pass and interleaver_pass
        
        status_str = "PASS" if case_pass else "FAIL"
        if case_pass:
            passed += 1
        else:
            failed += 1
        
        print(f"\n--- {filename} --- [{status_str}]")
        if actual_fname != filename:
            print(f"  Disk File:         {actual_fname}")
        print(f"  Description:       {exp.get('description')}")
        print(f"  Expected Behavior: {exp_behavior}")
        print(f"  Pipeline Status:   {got_status} {'OK' if behavior_pass else 'MISMATCH'}")
        print(f"  Detected Mod:      {got_mod} (Expected: {exp_mod}) {'OK' if mod_pass else 'MISMATCH'}")
        print(f"  Detected FEC:      {res.get('fec_type')} (Expected: {exp.get('expected_fec')})")
        print(f"  Detected Interl:   {got_interleaver} (Expected: {exp_interleaver}) {'OK' if interleaver_pass else 'MISMATCH'}")
        print(f"  CRC Valid:         {res.get('crc_valid')}")
        print(f"  Payload:           {got_payload}")
        if exp_msg:
            print(f"  Expected Message:  {exp_msg} {'OK' if msg_pass else 'MISMATCH'}")
    except Exception as e:
        failed += 1
        print(f"\n--- {filename} --- [CRASH/EXCEPTION]: {e}")

print("\n" + "=" * 80)
print(f"DATASET2 RESULTS: {passed} PASSED, {failed} FAILED, {skipped} SKIPPED out of {len(expected)} total")
print("=" * 80)
