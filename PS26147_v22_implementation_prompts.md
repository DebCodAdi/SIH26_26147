# PS 26147 — Implementation Prompts (v22, Checkpointed)

Derived from the v22 Master Execution Plan. Each phase is self-contained: a
build prompt, an automated in-code checkpoint, and a manual STOP gate — same
discipline as v20. **Do not proceed to the next phase until both checkpoints
pass.**

v22's DSP core (EKF carrier tracking, Gardner TED, CMA equalizer, GF(2) blind
code-length solver, Mahalanobis OOD gate) is a real upgrade over v20. But as
written, several pieces are specified in prose/module-docstrings and never
actually wired into `main.py` or the GUI — so a phase that just re-implements
the file as given would ship a pipeline that silently ignores its own
classifier and FEC search. Each phase below calls out the specific fix where
that happens, inline, so it doesn't get lost.

---

## Phase 0 — Dependency & Contract Smoke Test

**Prompt:**
```
Verify the environment and catch internal spec inconsistencies before
writing pipeline code:
- scikit-commpy: Trellis(K=7, polys=[0o171,0o133]) round-trips a known bit
  sequence through conv_encode -> viterbi_decode.
- galois.ReedSolomon(255,223,c=1) round-trips a known byte sequence.
- An LDPC library for IEEE 802.11n (N=648,K=324,R=1/2) round-trips a known
  bit sequence through encode -> belief-propagation decode. NOTE: v22's
  fec_decoders.py has no decode_ldpc() despite config.py defining LDPC_N/K
  and the FEC dispatch table listing an LDPC branch — this checkpoint must
  fail loudly (ModuleNotFoundError equivalent, not a silent skip) until
  that function exists, so the gap surfaces here rather than at Phase 4.
- np.frombuffer, np.packbits(bitorder='big'), scipy.signal.welch,
  scipy.signal.correlate all behave as expected on toy inputs.

Also reconcile two documented inconsistencies before Phase 1 starts:
1. sync_and_derotate_barker sets payload_start = peak_idx + 11 in the code,
   but the v22 prose spec says peak_idx + 1. Barker-11 is 11 symbols long,
   so +11 (skip the whole preamble) is correct; the prose is wrong. Confirm
   the code's +11 is what ships, don't "fix" the code to match the prose.
2. classifier.py's docstring says "Mahalanobis OOD-gated Random Forest
   classifier" but the actual code is pure Mahalanobis-distance
   nearest-class-mean — there is no RandomForest anywhere in the module,
   and nothing trains/serializes class_means/class_cov_invs. Decide now:
   either add the RF layer the docstring promises, or fix the docstring.
   Phase 3 below assumes you keep it Mahalanobis-only and builds the
   missing training step for that.
```

**In-code checkpoint — `checkpoint_0.py`:**
Assert each round-trip above exactly (LDPC: zero post-decode residual).
Assert `decode_ldpc` exists and is importable — fail the checkpoint with a
clear message if not, rather than skipping it. Write pass/fail per
component to `checkpoint_0.json`.

**Manual checkpoint:**
STOP — confirm all entries pass, and confirm you've made an explicit
decision (not a default-by-omission) on the Mahalanobis-vs-RandomForest
question above, since it changes what Phase 3 needs to train.

---

## Phase 1 — Ingestion, Format Sniffing & Signal Gating

**Prompt:**
```
Implement src/ingestion.py exactly per v22 spec:
- blind_ingest_and_clean(raw_bytes, p_fa=1e-4): candidate format sweep
  (float32 / int16 / int16-byteswapped / int8), scored by
  -|ln(std_I/std_Q)| - P(|x|>0.99), NaN/Inf spline repair on real and
  imaginary parts independently, DC removal, Neyman-Pearson energy
  detector (gamma = 1 + sqrt(2/N)*3.719) gating INSUFFICIENT_SIGNAL_EVIDENCE.
- ingest_wav(filepath) for stereo-complex and mono-Hilbert-analytic .wav.

FIX (carried over from v20, not solved here either — make it explicit
rather than silent): the format sniffer recovers *shape* (I/Q order,
dtype, byte order) but never recovers *sample rate*. Fs is currently a
hardcoded 200000.0 in main.py, applied to every file regardless of what
it actually was captured at. Add a required Fs parameter to
blind_ingest_and_clean's caller contract: read it from a sidecar
<filename>.meta.json, or accept --sample-rate on the CLI, and FAIL LOUD
(a distinct status, not a silent 200 kHz assumption) if neither is given.
This is a correctness bug in v22 as shipped, not a stylistic gap: every
downstream Hz-denominated estimate (CFO, baud rate) is wrong by
whatever ratio true_Fs/200000.0 differs.
```

**In-code checkpoint — `checkpoint_1.py`:**
Per v22: format-sniffer resolution test (int16 interleaved), NaN-repair
test, Neyman-Pearson noise-rejection test — all as already given. ADD: a
test that ingestion raises/returns a distinct `MISSING_SAMPLE_RATE` status
when no Fs source is provided, and correctly reads a `.meta.json` sidecar
when one is.

**Manual checkpoint:**
STOP — feed a pure-noise file and a real synthetic capture through by eye;
confirm the noise file is gated and the real one is not. Confirm the
missing-Fs case fails loud, not silently defaulting to 200 kHz.

---

## Phase 2 — Front-End Correction: CFO, IQ Balance, Equalization, Clock Recovery

**Prompt:**
```
Implement src/spectral.py and src/equalizer.py per v22 spec:
- estimate_coarse_cfo: centered Welch PSD + parabolic peak interpolation.
- recover_baud_rate: dual-path SCD (envelope-squared for linear mods,
  instantaneous-frequency-derivative for FSK), pick path by peak-to-mean
  SNR.
- blind_front_end_adapt: Gram-Schmidt IQ orthogonalization/gain
  correction, then 21-tap T/2 FSE-CMA equalizer (mu=0.002, center-spike
  init), operating on 2-SPS input.

Resampling note: main.py resamples x_bb to 2 SPS using the *estimated*
SPS from recover_baud_rate before handing off to the equalizer — this
is correct in v22's main.py, keep it, but add a sanity bound: if
estimated SPS < 1.5 or > 50, treat it as a failed clock-recovery
(new status BAUD_RECOVERY_FAILED) rather than resampling against a
clearly-wrong rate and feeding garbage into the CMA loop.
```

**In-code checkpoint — `checkpoint_2.py`:**
Per v22: Gram-Schmidt orthogonality-recovery test, CFO precision test
(<25 Hz error on a known offset). ADD: CMA eye-opening test on the
testbench's 3-tap multipath channel (assert post-equalization EVM is
lower than pre-equalization EVM, not just that the loop runs), and the
new SPS sanity-bound test.

**Manual checkpoint:**
STOP — plot the constellation before/after CMA equalization on a
multipath-impaired synthetic file. Does the eye visibly open?

---

## Phase 3 — Modulation Classification & Carrier/Timing Synchronization

**Prompt:**
```
Implement src/classifier.py and src/synchronization.py per v22 spec:
- extract_features: 6D [|C40|,|C42|,|C60|, std(|x|), diff_phase_std,
  peak_count] on unit-power-normalized baseband.
- evaluate_mahalanobis_ood: nearest-class Mahalanobis distance, gate at
  chi-squared(df=6, 0.999)=22.46 -> UNKNOWN_MODULATION.
- track_carrier_ekf: 2nd-order EKF PLL with per-modulation phase
  detectors (BPSK/QPSK/8PSK/16-QAM).
- sync_and_derotate_barker: M-fold rotation search against Barker-11,
  payload_start = peak_idx + 11.

REQUIRED FIX — this is the most important one in the whole plan: v22's
main.py never uses the classifier's output. It calls extract_features(),
throws the result away, and then hardcodes track_carrier_ekf(...,
mod_type="QPSK") and sync_and_derotate_barker(..., mod_type="QPSK")
unconditionally. That defeats the entire premise of a *blind* classifier
— every non-QPSK file in the testbench would be forced through the wrong
phase detector and rotation search and silently fail or mis-decode.

Fix: evaluate_mahalanobis_ood must actually run, its winning class must
be threaded into both track_carrier_ekf and sync_and_derotate_barker as
mod_type, and UNKNOWN_MODULATION must short-circuit the pipeline (return
a structured status, do not fall through to QPSK as a default).

Also build what's referenced but never assembled in v22: a training/
calibration step that produces class_means/class_cov_invs from labeled
synthetic features (one mean vector + covariance per modulation, from
testbench_gen.py output across seeds/SNRs), and serializes them
(e.g. classifier_params.npz) for evaluate_mahalanobis_ood to load. Without
this the function has no way to run at all — the class distributions
don't exist anywhere in v22 as delivered.
```

**In-code checkpoint — `checkpoint_3.py`:**
Per v22: EKF lock-under-drift test, Barker rotation-resolution test. ADD:
- A test that feeds each of the 4+ trained modulations through the full
  classify -> sync path and asserts the *correct* mod_type is threaded
  through (not hardcoded QPSK) by checking sync_and_derotate_barker was
  called with the classifier's actual output, not a constant.
- A test that an out-of-distribution / untrained modulation produces
  UNKNOWN_MODULATION and the pipeline halts there — does not proceed to
  EKF/Barker sync at all.

**Manual checkpoint:**
STOP — run one file from each trained modulation class through the CLI
end to end and confirm the reported modulation in telemetry matches
ground truth for all of them, not just QPSK. This is the single check
that catches the hardcoding bug if the fix above was missed.

---

## Phase 4 — Blind Interleaver ID & Multi-FEC Decoding

**Prompt:**
```
Implement src/deinterleaver.py and src/fec_decoders.py per v22 spec:
- estimate_interleaver_depth: auto-mutual-information peak search, d in
  [2,64].
- deinterleave_block / deinterleave_convolutional (FIFO-register form,
  D=I(I-1)M latency skip).
- demap_8psk_soft / demap_16qam_soft: max-log / symmetric-Gray LLRs.
- detect_blind_code_length: GF(2) Gauss-Jordan rank-drop search for
  block-code (n,k).
- decode_viterbi (6-bit flush strip), decode_reed_solomon (per-block
  try/except, <=16 byte errors).

REQUIRED FIX (LDPC): config.py defines LDPC_N=648/LDPC_K=324/LDPC_RATE=0.5
and the FEC dispatch table in the v22 prose lists an LDPC branch
(belief propagation -> integer parity check -> systematic bit extraction),
but fec_decoders.py has no decode_ldpc() at all. Implement it now:
log-domain sum-product belief propagation over the 802.11n base H matrix,
cast to int8, verify (c_int @ H_int.T) % 2 == 0, extract c_dec[:324].
This was caught at Phase 0 but must be resolved here, not deferred again.

REQUIRED FIX (dispatch): main.py never calls estimate_interleaver_depth,
deinterleave_block/convolutional, decode_viterbi, or decode_reed_solomon
at all — it goes straight from hard-slicing payload_syms.real to CRC
extraction, meaning FEC and interleaving are entirely bypassed in the
executable pipeline despite being fully implemented as library functions.
Wire the actual hypothesis dispatch main.py is missing:
  1. Try each interleaver hypothesis (none / block M in {4,8,16} /
     convolutional I=4,M=16) from estimate_interleaver_depth's candidates.
  2. For each, try each FEC branch (standalone conv / concatenated
     RS+conv / RS-only / LDPC) per the demapping each requires.
  3. Accept the first (interleaver, FEC) combination whose decode
     succeeds structurally (Viterbi re-encode residual near-zero, RS
     decode succeeds on all blocks, or LDPC parity checks to zero).
  4. If none succeed, return FEC_TYPE_MISMATCH — do not fall through to
     treating raw hard-sliced bits as if they were already FEC-decoded.
This is the core "blind" capability the PS asks for; as shipped, v22's
main.py silently skips it entirely.
```

**In-code checkpoint — `checkpoint_4.py`:**
Per v22: GF(2) rank-solver test, Viterbi flush-strip test, RS
error-correction test. ADD:
- decode_ldpc round-trip test (assert zero residual on a noiseless
  codeword, matching the Phase 0 assertion).
- End-to-end dispatch test: generate one file per (interleaver type x FEC
  type) combination, run through the fixed main.py path, assert the
  correct (interleaver, FEC) hypothesis wins for every combination.
- Deliberately route a concatenated-type file through the standalone-conv
  branch only; assert it reports FEC_TYPE_MISMATCH rather than a wrong
  low-confidence decode.

**Manual checkpoint:**
STOP — the single most important gate in the plan, same as v20. Confirm
every interleaver/FEC combination in the testbench matrix has actually
been run (not assumed), and the deliberate wrong-branch test fails
cleanly.

---

## Phase 5 — Blind CRC Extraction & Entropy Classification

**Prompt:**
```
Implement src/frame_validator.py per v22 spec:
- blind_extract_and_classify_payload: sweep L in [5, L_max], big-endian
  CRC-32 match, Shannon entropy on the recovered payload, threshold 7.95
  bits/byte separating VALID_PAYLOAD_PLAINTEXT from
  VALID_PAYLOAD_ENCRYPTED.

Note the entropy classifier is a nice addition over v20 but is purely
informational — do not let "VALID_PAYLOAD_ENCRYPTED" be treated as a
pipeline failure. A CRC-valid encrypted payload is still a correct blind
decode; encryption of the *content* is out of scope for this PS (same
conclusion as v20's critique response, item 37).

Run the full corrected pipeline (Phases 1-5, with Phase 3/4's dispatch
fixes in place) against every file in the testbench, diff against
ground truth field by field. Report X/N passed.
```

**In-code checkpoint — `checkpoint_5.py`:**
Per v22: blind CRC + plaintext-entropy test, encrypted-entropy
classification test. ADD: full testbench run, per-file PASS/FAIL,
"X/N passed" summary — this doesn't exist anywhere in v22 as delivered
(checkpoint_5.py only unit-tests the function in isolation, never runs
the assembled pipeline end to end the way v20's Phase 5 checkpoint did).

**Manual checkpoint:**
STOP — do not proceed to the GUI until every testbench case passes or
the failure is understood. Spot-check one VALID_PAYLOAD_ENCRYPTED and one
VALID_PAYLOAD_PLAINTEXT case by eye.

---

## Phase 6 — GUI (Real Pipeline, Not Mock Data)

**Prompt:**
```
Wrap the now-verified pipeline in the PyQt6/PyQtGraph dashboard per v22
spec: QThread worker, spectrum waterfall, constellation scatter,
telemetry HUD, payload console, 60 FPS target for the visual refresh
specifically.

REQUIRED FIX: v22's src/gui/app.py DspWorkerThread does not call any real
pipeline code — it generates np.random mock PSD/constellation/telemetry
data in an infinite loop. This directly violates the standing rule (true
in both v20 and v22's stated design goals) that the GUI's load-file path
and the CLI must call the exact same function, not parallel
implementations. As shipped, the v22 GUI is a pure visual mockup and
would demo successfully while decoding nothing.

Fix: replace DspWorkerThread's run() loop with a call into the Phase
1-5 pipeline (the same function main.py's process_file uses, refactored
into a single run_full_pipeline(filepath) -> dict entry point, mirroring
v20's pattern), executed once per loaded file rather than looping mock
data forever. Route its structured result dict into spectrum_signal /
constellation_signal / telemetry_signal / payload_signal from real
intermediate arrays (x_bb for spectrum, payload_syms for constellation),
not synthetic ones.
```

**In-code checkpoint:**
Re-run the testbench through the GUI's load-file path end-to-end; assert
results are identical to the Phase 5 checkpoint's CLI output for the same
files. This checkpoint cannot pass while DspWorkerThread still generates
mock data — that's the point of it.

**Manual checkpoint:**
STOP — visually confirm the waterfall/constellation/HUD/console update
from a real loaded file (not a random-noise animation), and that decoded
ASCII/hex in the console matches a known testbench payload by eye.

---

## Phase 7 — Ad-Hoc Verification for Files With No Ground Truth

**Prompt:**
```
v22 has no equivalent of this phase at all — there is no run_full_pipeline
single-entry function, no status/stage_reached structured result, and no
tool for verifying an arbitrary real capture you don't have an answer key
for. Port this wholesale from the v20 master checkpoint doc:

- run_full_pipeline(filepath, sample_rate=None) -> dict with status,
  stage_reached, and every intermediate field (modulation, cfo_hz,
  baud_rate, interleaver_type/params, fec_type, crc_valid, payload_hex/
  ascii), never raising, always returning a diagnosable structured result.
  This is also what Phase 6's GUI fix above needs to call.
- compare_to_ground_truth(result, ground_truth, tolerances) for partial
  or full known-answer comparison, with tolerances sourced from
  checkpoint_2.json/checkpoint_3.json's measured estimation error, not
  hardcoded.
- The trust-ordering manual procedure: CRC pass is primary evidence: run
  your own known-message file first if a real capture fails, before
  concluding the file (rather than the pipeline) is bad; never treat
  readable-looking payload_ascii as evidence on its own if crc_valid is
  False.
```

**In-code checkpoint — `checkpoint_master.py`:**
Same two-tier structure as v20: tier 1 (known ground truth, from
testbench_gen.py output) asserts overall_pass across the full matrix using
the corrected Phase 3/4 dispatch; tier 2 (no ground truth) asserts
internal consistency only (e.g. crc_valid=True implies nonzero
payload_len_bytes).

**Manual checkpoint:**
STOP — confirm `run_full_pipeline` is the one function both `main.py` and
the GUI's file-load handler call (verify by inspection, not just by
matching output) — this closes the loop on the Phase 6 fix.

---

## Addendum — second verification pass (found after the first draft shipped)

Two more real gaps, one of them severe enough to invalidate several of the
checkpoints above if not fixed first:

**A. `tests/testbench_gen.py` doesn't generate the matrix it claims to.**
`generate_test_vector`'s FEC branch only implements `"CONV"` and `"RS"` —
requesting `"RS+CONV"` or `"LDPC"` silently falls through to
`encoded_bits = raw_bits` (no FEC at all) while still labeling the file
with the requested type. Modulation mapping only implements `"BPSK"` and
`"QPSK"` — requesting `"8PSK"` or `"16-QAM"` silently produces BPSK symbols
under that label. This means every checkpoint in Phases 3-5 above that
says "generate a file of type X, verify the pipeline reports X" is
checking against **wrong ground truth** for anything outside
{BPSK,QPSK}×{Conv,RS}, and would pass or fail meaninglessly.

**Fix — insert this as Phase 1.5, before Phase 3's checkpoint can be
trusted:** extend `generate_test_vector` to actually implement 8PSK/16-QAM
symbol mapping (Gray-coded, matching `CONSTELLATION_8PSK`/`_16QAM` in
config.py) and the RS+Conv concatenated chain (RS encode -> byte
interleave -> conv encode) and LDPC chain (zero-pad to 324-bit multiples
-> LDPC encode), matching what v20's generator already did — this is a
regression relative to v20, not new scope. Do not trust any Phase 3/4/5
checkpoint result for non-BPSK/QPSK or non-Conv/RS cases until this is
fixed and the generator is itself spot-checked (decode a generated
16-QAM/LDPC file's ground-truth bits back through the encoder and confirm
they match, the same round-trip discipline Phase 0 already demands
elsewhere).

**B. `apply_gardner_ted` is implemented but orphaned — never called from
`main.py`.** The pipeline goes resample-to-2SPS -> CMA equalize -> EKF
carrier track with no timing-error correction step in between, despite
the testbench deliberately injecting non-integer SPS (4.37) specifically
to exercise timing recovery. CMA assumes a roughly-correct timing grid
going in; without Gardner TED running first, fractional timing offset has
nothing correcting it.

**Fix:** insert `apply_gardner_ted` between `blind_front_end_adapt`'s
output and `track_carrier_ekf`'s input in `main.py` (Phase 2's fix
section), and add a Phase 2 checkpoint asserting EVM is measurably lower
with Gardner TED in the chain than without it, on a file with injected
non-integer SPS specifically.

**Lower-severity, flagged but not blocking:**
- The format sniffer's `swap_iq=True` code path exists but no candidate
  tuple ever sets it — genuine Q-before-I captures are never tried despite
  the function appearing to handle that case.
- The `C60` cumulant formula (`m60 - 15·m20·m40 + 30·m20³`) isn't checked
  against a reference (Swami & Sadler-style formulas); add a Phase 3
  checkpoint that verifies computed cumulants against published
  theoretical values for ideal noiseless BPSK/QPSK, since a subtly wrong
  formula degrades classifier accuracy silently rather than crashing.
- `detect_blind_code_length`'s brute-force n∈[10,256] GF(2) rank search is
  roughly O(n³) per file — fine at demo scale, worth a timeout/early-exit
  guard if judges supply an unusually long capture.

---

## Closing note — what changed relative to a literal v22 build

Eight real gaps were found across two verification passes and folded into
the phases above rather than left implicit: missing LDPC decoder, missing
Fs handling, missing classifier training/calibration step, classifier
output not wired to sync (hardcoded QPSK), FEC/interleaver dispatch never
called from `main.py`, GUI running on mock data instead of the real
pipeline, a testbench generator that silently mislabels 4 of its 6
claimed FEC/modulation combinations, and an implemented-but-uncalled
timing-recovery stage. None of these are difficulty-of-the-underlying-DSP
issues — they're wiring and generator-fidelity gaps between well-built
individual functions and the places that are supposed to call them
correctly. Building v22 exactly as delivered would produce a demo that
looks complete, passes checkpoints against wrong ground truth, and isn't
actually complete.

**Is this now perfect? No — and it's worth being direct about why "perfect"
isn't the right bar.** After these two passes I'm fairly confident the
*wiring* is now specified correctly. What's still genuinely open, and was
already flagged as irreducible in the earlier 40-point critique against
v20, remains irreducible here too: real train/test domain shift (every
classifier here trains on synthetic data only), and the fact that blind
FEC+interleaver identification on truly adversarial or out-of-matrix
inputs is a best-effort search, not a guarantee. Those aren't bugs to fix;
they're honest limits of the problem, and the status/stage_reached
architecture exists specifically so the system reports "I don't know"
there instead of a confident wrong answer. A third pass might still find
something — that's the nature of a spec this size — but I'd stop chasing
"perfect" and instead build Phase 0 through 2 first and let real runs
surface anything left.
