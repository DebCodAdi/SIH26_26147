"""
Receiver benchmark harness for PS 26147.

Measures the physical-layer receive chain against known ground truth over a statistically
meaningful matrix of (modulation x SNR x trials), reporting:

  * pre-FEC bit error rate (BER) of the demodulated stream vs. the true transmitted bits
  * automatic modulation classification (AMC) accuracy
  * carrier frequency offset (CFO) and baud-rate estimation error
  * end-to-end decode success (CRC lock with a byte-exact payload match)

BER is the primary signal here. EVM computed over a whole burst is a misleading optimisation
target because it includes the zero-valued postamble and PLL acquisition transients; BER
against the true bit sequence is unambiguous. For reference, uncoded coherent BER at 20 dB
Es/N0 is effectively zero (<1e-8) for BPSK/QPSK, so any measured BER above ~1e-2 at high SNR
indicates a receiver defect rather than a noise limit.

Usage:
    python -m tools.benchmark_rx                    # default matrix
    python -m tools.benchmark_rx --trials 20        # more trials per cell
    python -m tools.benchmark_rx --mods QPSK,BPSK --snrs 30,20,10
    python -m tools.benchmark_rx --json out.json    # machine-readable, for CI regression gating
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.classifier import extract_features, evaluate_mahalanobis_ood
from src.demodulators import demap_linear
from src.equalizers import apply_twopass_cma
from src.pipeline import run_full_pipeline
from src.spectral import estimate_cfo_and_baud, resample_to_2sps
from src.synchronizer import resolve_sync_and_rotation, track_carrier_pll
from tests.testbench_gen import generate_test_vector

DEFAULT_MODS = ["BPSK", "QPSK", "8PSK", "16-QAM"]
DEFAULT_SNRS = [30.0, 20.0, 15.0, 10.0, 5.0]


def best_aligned_ber(demapped: np.ndarray, truth_bits: np.ndarray) -> float:
    """
    Minimum bit error rate of `demapped` against `truth_bits` over all time alignments.

    The demapped stream contains a preamble and an unknown bit offset, so the comparison is
    alignment-invariant: mapping both to +-1 turns Hamming distance into a correlation, and the
    peak correlation over all lags gives the best-aligned error count in one pass.
    """
    if len(demapped) < len(truth_bits) or len(truth_bits) == 0:
        return 1.0
    d = 2.0 * demapped.astype(np.float32) - 1.0
    t = 2.0 * truth_bits.astype(np.float32) - 1.0
    corr = np.correlate(d, t, mode="valid")
    best = float(np.max(corr))
    n = len(truth_bits)
    errors = (n - best) / 2.0
    return float(errors / n)


def receive_chain_bits(rx_wave: np.ndarray, fs: float, mod_type: str) -> tuple[list[np.ndarray], dict]:
    """
    Runs the production receive chain and returns every candidate demapped bit stream
    (one per rotation/mapping hypothesis) plus estimated parameters.

    This deliberately calls the same functions src/pipeline.py uses, so the benchmark measures
    the shipped code rather than a reimplementation.
    """
    cfo_hz, rs, sps, x_bb = estimate_cfo_and_baud(rx_wave, fs=fs)

    if sps <= 1.25 or len(rx_wave) < 512:
        pwr = float(np.mean(np.abs(x_bb) ** 2))
        y_syms = (x_bb / np.sqrt(pwr + 1e-12)).astype(np.complex64)
    else:
        x_2sps = resample_to_2sps(x_bb, sps)
        y_syms = apply_twopass_cma(x_2sps)
        if len(y_syms) == 0:
            y_syms = x_2sps[::2]

    features = extract_features(y_syms)
    predicted_mod, _ = evaluate_mahalanobis_ood(features, symbols=x_bb)

    payload_raw, _, _, _ = resolve_sync_and_rotation(y_syms, mod_type=mod_type)
    source = payload_raw if len(payload_raw) > 0 else y_syms
    locked = track_carrier_pll(source, mod_type=mod_type)

    candidates = []
    for symbols in (locked, source, y_syms):
        if len(symbols) == 0:
            continue
        for _label, bits in demap_linear(symbols, mod_type=mod_type):
            candidates.append(bits)

    return candidates, {
        "cfo_hz": float(cfo_hz),
        "baud_rate": float(rs),
        "sps": float(sps),
        "predicted_mod": predicted_mod,
    }


def run_cell(mod: str, snr_db: float, trials: int, cfo_hz: float, fs: float,
             baud_rate: float, multipath: bool, base_seed: int) -> dict:
    """Runs all trials for one (modulation, SNR) cell and aggregates the metrics."""
    bers, cfo_errs, baud_errs = [], [], []
    amc_hits = 0
    decode_hits = 0

    for trial in range(trials):
        seed = base_seed + trial
        truth: dict = {}
        rx_wave, payload = generate_test_vector(
            mod_type=mod, fec_type="NONE", payload_text=b"BENCHMARK_RX_PAYLOAD_0123456789",
            fs=fs, baud_rate=baud_rate, cfo_hz=cfo_hz, snr_db=snr_db,
            multipath=multipath, seed=seed, truth_out=truth,
        )

        try:
            candidates, est = receive_chain_bits(rx_wave, fs, mod)
        except Exception:
            bers.append(1.0)
            continue

        truth_bits = truth.get("coded_bits", np.zeros(0, dtype=np.uint8))
        trial_ber = min((best_aligned_ber(c, truth_bits) for c in candidates), default=1.0)
        bers.append(trial_ber)
        cfo_errs.append(abs(est["cfo_hz"] - cfo_hz))
        baud_errs.append(abs(est["baud_rate"] - baud_rate))
        amc_hits += int(est["predicted_mod"] == mod)

        # End-to-end decode through the full public pipeline entry point.
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".iq", delete=False) as fh:
            path = fh.name
            fh.write(rx_wave.tobytes())
        try:
            res = run_full_pipeline(path, user_fs=fs)
            decode_hits += int(bool(res.get("crc_valid")) and res.get("payload") == payload)
        except Exception:
            pass
        finally:
            os.unlink(path)

    return {
        "modulation": mod,
        "snr_db": snr_db,
        "trials": trials,
        "ber_median": float(np.median(bers)) if bers else 1.0,
        "ber_best": float(np.min(bers)) if bers else 1.0,
        "amc_accuracy": amc_hits / max(1, trials),
        "decode_success": decode_hits / max(1, trials),
        "cfo_err_median_hz": float(np.median(cfo_errs)) if cfo_errs else float("nan"),
        "baud_err_median_hz": float(np.median(baud_errs)) if baud_errs else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PS 26147 receiver benchmark harness")
    parser.add_argument("--mods", type=str, default=",".join(DEFAULT_MODS))
    parser.add_argument("--snrs", type=str, default=",".join(str(s) for s in DEFAULT_SNRS))
    parser.add_argument("--trials", type=int, default=8, help="trials per (mod, SNR) cell")
    parser.add_argument("--cfo", type=float, default=150.0)
    parser.add_argument("--fs", type=float, default=200000.0)
    parser.add_argument("--baud", type=float, default=50000.0)
    parser.add_argument("--multipath", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--json", type=str, default=None, help="write results to this JSON file")
    args = parser.parse_args()

    mods = [m.strip() for m in args.mods.split(",") if m.strip()]
    snrs = [float(s) for s in args.snrs.split(",") if s.strip()]

    print(f"Receiver benchmark: {len(mods)} modulations x {len(snrs)} SNRs x {args.trials} trials")
    print(f"Fs={args.fs:.0f} Hz  baud={args.baud:.0f}  CFO={args.cfo:.0f} Hz  multipath={args.multipath}\n")
    header = f"{'Mod':<8}{'SNR':>6}{'BER(med)':>11}{'BER(best)':>11}{'AMC':>7}{'Decode':>8}{'CFOerr':>10}{'Bauderr':>10}"
    print(header)
    print("-" * len(header))

    results = []
    for mod in mods:
        for snr in snrs:
            cell = run_cell(mod, snr, args.trials, args.cfo, args.fs, args.baud,
                            args.multipath, args.seed)
            results.append(cell)
            print(f"{cell['modulation']:<8}{cell['snr_db']:>6.0f}"
                  f"{cell['ber_median']:>11.4f}{cell['ber_best']:>11.4f}"
                  f"{cell['amc_accuracy']*100:>6.0f}%{cell['decode_success']*100:>7.0f}%"
                  f"{cell['cfo_err_median_hz']:>10.1f}{cell['baud_err_median_hz']:>10.1f}")

    high_snr = [r for r in results if r["snr_db"] >= 20.0]
    if high_snr:
        print(f"\nHigh-SNR (>=20 dB) summary — this is where a correct receiver should be near-perfect:")
        print(f"  median BER      : {np.median([r['ber_median'] for r in high_snr]):.4f}   (target < 0.001)")
        print(f"  AMC accuracy    : {np.mean([r['amc_accuracy'] for r in high_snr])*100:.1f}%   (target > 95%)")
        print(f"  decode success  : {np.mean([r['decode_success'] for r in high_snr])*100:.1f}%   (target > 95%)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
