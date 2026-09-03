"""
PS 26147 demonstration script.

Generates three known-truth captures, runs each through the full production pipeline
(src/pipeline.run_full_pipeline), and prints a TRUE vs ESTIMATED comparison table for every
parameter the problem statement asks the system to extract.

    python demo/run_demo.py            # run once
    python demo/run_demo.py --runs 5   # repeat, to show it is stable and not a lucky seed

Exit status is 0 only if every case in every run passes, so it can be used as a smoke test.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import scipy.io.wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import run_full_pipeline           # noqa: E402
from tests.testbench_gen import generate_test_vector  # noqa: E402

GREEN, RED, YELLOW, BOLD, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[1m", "\033[0m"


def build_cases(tmpdir: str, seed_offset: int) -> list[dict]:
    """Builds the three demo captures on disk and returns their ground truth."""
    cases = []

    # ---- Case 1: BPSK, uncoded, 30 dB, raw complex64 .iq ----
    payload = b"SIH26147_BPSK_DEMO_PAYLOAD_0001"
    truth: dict = {}
    rx, _ = generate_test_vector(
        mod_type="BPSK", fec_type="NONE", payload_text=payload,
        fs=200000.0, baud_rate=50000.0, cfo_hz=150.0, snr_db=30.0,
        seed=100 + seed_offset, truth_out=truth,
    )
    path = os.path.join(tmpdir, "case1_bpsk.iq")
    rx.astype(np.complex64).tofile(path)
    cases.append({
        "name": "Case 1 — BPSK uncoded, 30 dB, .iq (complex64)",
        "path": path, "user_fs": 200000.0,
        "truth": {"fs": 200000.0, "baud": 50000.0, "sps": 4.0, "cfo": 150.0,
                  "mod": "BPSK", "fec": "NONE", "interleaver": "NONE",
                  "crc": True, "payload": payload},
    })

    # ---- Case 2: QPSK + convolutional K=7 r1/2 (Viterbi), 30 dB, raw .iq ----
    payload = b"SIH26147_QPSK_VITERBI_DEMO_0002"
    rx, _ = generate_test_vector(
        mod_type="QPSK", fec_type="CONV", payload_text=payload,
        fs=200000.0, baud_rate=50000.0, cfo_hz=150.0, snr_db=30.0,
        seed=200 + seed_offset,
    )
    path = os.path.join(tmpdir, "case2_qpsk_conv.iq")
    rx.astype(np.complex64).tofile(path)
    cases.append({
        "name": "Case 2 — QPSK + convolutional K=7 r1/2 (Viterbi), 30 dB, .iq",
        "path": path, "user_fs": 200000.0,
        "truth": {"fs": 200000.0, "baud": 50000.0, "sps": 4.0, "cfo": 150.0,
                  "mod": "QPSK", "fec": "CONV_K7_R1/2", "interleaver": "NONE",
                  "crc": True, "payload": payload},
    })

    # ---- Case 3: QPSK, uncoded, 30 dB, stereo .wav (I=left, Q=right) ----
    # A stereo WAV carrying I and Q is the standard way SDR captures are stored as audio;
    # ingest_wav() reads Fs directly from the RIFF header, so Fs here is exact, not inferred.
    payload = b"SIH26147_WAV_DEMO_PAYLOAD_0003"
    fs_wav = 96000.0
    baud_wav = 24000.0
    rx, _ = generate_test_vector(
        mod_type="QPSK", fec_type="NONE", payload_text=payload,
        fs=fs_wav, baud_rate=baud_wav, cfo_hz=100.0, snr_db=30.0,
        seed=300 + seed_offset,
    )
    stereo = np.column_stack([np.real(rx), np.imag(rx)])
    stereo = (stereo / (np.max(np.abs(stereo)) + 1e-9) * 30000.0).astype(np.int16)
    path = os.path.join(tmpdir, "case3_qpsk.wav")
    scipy.io.wavfile.write(path, int(fs_wav), stereo)
    cases.append({
        "name": "Case 3 — QPSK uncoded, 30 dB, .wav (stereo I/Q, Fs from RIFF header)",
        "path": path, "user_fs": None,
        "truth": {"fs": fs_wav, "baud": baud_wav, "sps": fs_wav / baud_wav, "cfo": 100.0,
                  "mod": "QPSK", "fec": "NONE", "interleaver": "NONE",
                  "crc": True, "payload": payload},
    })
    return cases


def check(label: str, true_val, est_val, ok: bool, note: str = "") -> tuple[str, bool]:
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    row = f"  {label:<14}{str(true_val):<26}{str(est_val):<26}{mark}"
    if note:
        row += f"  {YELLOW}{note}{RESET}"
    return row, ok


def run_case(case: dict) -> bool:
    """Runs one capture and prints its TRUE vs ESTIMATED table. Returns pass/fail."""
    t = case["truth"]
    res = run_full_pipeline(case["path"], user_fs=case["user_fs"])

    print(f"\n{BOLD}{case['name']}{RESET}")
    print(f"  file: {os.path.basename(case['path'])}   status: {res.get('status')}")
    print(f"  {'PARAMETER':<14}{'TRUE':<26}{'ESTIMATED':<26}RESULT")
    print("  " + "-" * 74)

    est_payload = res.get("payload", b"")
    est_sps = float(res.get("sps", 0.0))
    est_baud = float(res.get("baud_rate", 0.0))
    est_fs = float(res.get("fs_hz", 0.0))
    est_cfo = float(res.get("cfo_hz", 0.0))

    rows, oks = [], []
    for row, ok in [
        check("Sampling Fs", f"{t['fs']:.0f} Hz", f"{est_fs:.0f} Hz",
              abs(est_fs - t["fs"]) < 1.0,
              "from RIFF header" if case["path"].endswith(".wav") else "operator-supplied"),
        check("Baud rate", f"{t['baud']:.0f} Bd", f"{est_baud:.1f} Bd",
              abs(est_baud - t["baud"]) / t["baud"] < 0.02),
        check("SPS", f"{t['sps']:.2f}", f"{est_sps:.2f}",
              abs(est_sps - t["sps"]) < 0.1),
        check("CFO", f"{t['cfo']:.1f} Hz", f"{est_cfo:.2f} Hz",
              abs(est_cfo - t["cfo"]) < 50.0),
        check("Modulation", t["mod"], res.get("modulation", "?"),
              res.get("modulation") == t["mod"]),
        check("FEC", t["fec"], res.get("fec_type", "?"),
              res.get("fec_type") == t["fec"]),
        check("Interleaver", t["interleaver"], res.get("interleaver", "?"),
              res.get("interleaver") == t["interleaver"]),
        check("CRC-32", t["crc"], res.get("crc_valid", False),
              bool(res.get("crc_valid")) == t["crc"]),
        check("Payload", t["payload"].decode(errors="replace")[:24],
              (est_payload.decode(errors="replace")[:24] if isinstance(est_payload, bytes) else "?"),
              est_payload == t["payload"]),
    ]:
        rows.append(row)
        oks.append(ok)

    print("\n".join(rows))
    passed = all(oks)
    print(f"  => {GREEN + 'CASE PASSED' + RESET if passed else RED + 'CASE FAILED' + RESET}"
          f"  ({sum(oks)}/{len(oks)} parameters correct)")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="PS 26147 demo")
    parser.add_argument("--runs", type=int, default=1, help="repeat the whole demo N times")
    args = parser.parse_args()

    print(f"{BOLD}PS 26147 — Automated analysis of .IQ/.wav files with signal parameter extraction{RESET}")
    print("Each capture below has known ground truth; the pipeline is given only the file.")

    all_runs_ok = True
    for run_idx in range(1, args.runs + 1):
        if args.runs > 1:
            print(f"\n{BOLD}{'=' * 78}\n RUN {run_idx} of {args.runs}\n{'=' * 78}{RESET}")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Vary the noise realisation per run so repeated runs are a genuine stability
            # check rather than the same capture replayed.
            cases = build_cases(tmpdir, seed_offset=run_idx)
            results = [run_case(c) for c in cases]
        ok = all(results)
        all_runs_ok &= ok
        print(f"\n  RUN {run_idx}: {sum(results)}/{len(results)} cases passed "
              f"{GREEN + 'OK' + RESET if ok else RED + 'FAILED' + RESET}")

    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}OVERALL: {(GREEN + 'ALL RUNS PASSED') if all_runs_ok else (RED + 'SOME RUNS FAILED')}{RESET}")
    return 0 if all_runs_ok else 1


if __name__ == "__main__":
    sys.exit(main())
