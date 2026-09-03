# IMPROVEMENTS.md — SIH 26147 Blind SDR Interceptor: Audit & Improvement Plan

---

## 0. STATUS: what has been fixed so far, and the #1 remaining root cause

### 0.0 CORRECTION to §0.1 below (superseded — read this first)

§0.1 originally claimed that broken symbol timing recovery was *the* blocking defect, on the
strength of 50-90% measured EVM. **That conclusion was wrong, and measurement disproved it.**
A ground-truth bit-error-rate harness (`tools/benchmark_rx.py`, added since) shows:

```
Mod        SNR   BER(med)  BER(best)     Decode
BPSK        30     0.0000     0.0000       100%
QPSK        30     0.0000     0.0000       100%
```

**Pre-FEC BER is exactly zero** at 30 dB for BPSK/QPSK — the demodulation chain recovers the
transmitted bits perfectly. The 50-90% EVM figure was an artefact: `compute_constellation_evm`
is applied over the whole burst including the 24-symbol all-zero postamble, and zero-valued
symbols scored against a unit-power reference contribute ~100% error each, putting a large
floor under the metric regardless of receiver quality. EVM over a whole burst is therefore not
a valid optimisation target; BER against known transmitted bits is, and that is what the new
harness measures.

The factual observations in §0.1 still stand — `apply_gardner_ted` genuinely is not a Gardner
TED, and `cma_equalize` genuinely is dead code — but they are **robustness gaps, not the
current blocker**. The crude whole-sample strobe happens to land correctly in the synthetic
test setup (integer SPS = 4, symbols aligned to even indices after decimation); it would fail
on a real capture with fractional SPS or arbitrary timing offset. They are re-prioritised
accordingly: important for real-world captures, not the reason the current numbers are low.

This correction is itself the main argument for the harness: two rounds of DSP changes were
made against a misleading metric before ground-truth measurement redirected the work.

### 0.1 [SUPERSEDED BY §0.0] Original claim: symbol timing recovery is not real

`src/equalizers.py:apply_gardner_ted` is documented (in its own docstring, in `README.md`
Stage 3, and in the pipeline architecture table) as a **Gardner Timing Error Detector**
implementing `e_τ[k] = Re{y[k-1/2](y*[k] - y*[k-1])}`. It implements no such thing. The actual
body compares the total energy of even-indexed vs odd-indexed samples and picks **one fixed
whole-sample strobe for the entire burst**:

```python
p0 = float(np.sum(np.abs(x_2sps[0::2])**2))
p1 = float(np.sum(np.abs(x_2sps[1::2])**2))
k_opt = 0 if p0 >= p1 else 1
return x_2sps[k_opt::2]
```

There is no fractional-sample interpolation, no timing error signal, and no feedback loop. Real
symbol timing is essentially never aligned to an integer sample of the 2-SPS grid, so this
samples up to **half a symbol away from the eye centre** and never corrects. Measured
consequence, at a nominal 20 dB SNR with no multipath (should be ~10% EVM or better):

| Modulation | Measured EVM (mean of 5 seeds) |
|---|---|
| BPSK | 92.6% |
| QPSK | 50.8% |
| 8PSK | 59.1% |
| 16-QAM | 74.6% |

Compounding it, `apply_twopass_cma` — the function the pipeline actually calls, named for the
"Dual-Pass 11-Tap CMA" the README advertises as Stage 3 — **never calls the CMA equalizer**.
`cma_equalize`/`_cma_equalize_numba` exist, are JIT-compiled and correct-looking, and are dead
code. So the live pipeline performs **no channel equalization and no receive-side matched
filtering at all**, on top of sampling at the wrong instant.

This is the upstream cause of most symptoms catalogued in §2 and §3: every downstream stage
(AMC, demod, FEC) is being fed symbols with 50-90% EVM, which is why classification looks
random, why higher-order constellations (16/64-QAM) fail worst, and why FEC blocks needing
hundreds of consecutive correct decisions (LDPC, concatenated RS) essentially never lock.

**Fixing this properly is a real DSP redesign, not a patch.** It needs a genuine interpolating
timing-recovery loop (Farrow/polyphase fractional interpolator + Gardner or Mueller-Müller TED
+ loop filter), a receive-side matched filter (true RRC matched to the transmit shaping), and a
modulation-aware equalizer (CMA's `R2` is only valid for constant-modulus signals; forcing
`R2=1.0` on QAM actively warps the constellation). An attempt was made in this session to wire
in the existing CMA plus a Farrow-interpolating Gardner loop; it **measured as a net regression**
(0/32 vs 3/32 end-to-end decode successes) and was reverted rather than shipped. That work is
preserved for reference but is not in the tree. Doing it properly is the top of the P0 list.

### 0.2 Fixes applied and independently verified in this session

| Fix | File | Verification |
|---|---|---|
| CFO half-baud ambiguity: added 3-way consensus-median cross-check across the x²/x⁴/x⁸ harmonic estimates, down-weighting any candidate that disagrees with consensus (catches a sideband lock) | `src/spectral.py` | Previously-flaky `test_phase2_spectral_and_baud_recovery` now passes **15/15** consecutive runs (was failing ~1 in 3) |
| Signal-presence gate switched from the biased kurtosis SNR cutoff to the already-built MME eigenvalue detector | `src/pipeline.py` | Eliminates false `NO_SIGNAL_DETECTED` rejections; signals at +6…+10 dB that were previously discarded now reach demodulation |
| AMC now loads empirically-fit centroids + full 6×6 covariances from `models/classifier_params.npz`, replacing hand-typed constants (falls back safely if absent) | `src/classifier.py` | Model is live instead of dead; regenerate with `python -m src.train_classifier` |
| Classifier retrained through the **real production DSP chain** over 7 classes × SNR 4-30 dB (1400 samples), with full covariance and a held-out split | `src/train_classifier.py` | **69.1% held-out Mahalanobis / 72.0% held-out RF** (was: hand-typed constants, never validated, effectively random outside one calibration point) |
| 64-QAM PLL used 16-QAM's 4-level decision grid; gave it a correct 8-level grid and its own `mod_code` | `src/synchronizer.py` | 64-QAM carrier tracking no longer decision-directed against wrong reference points |
| GMSK routed to the FM-discriminator path (it is binary CPFSK, h≈0.5) instead of the linear BPSK slicer | `src/pipeline.py` | Correct receiver structure for the modulation |
| Test-vector generator: seeded RNG for reproducibility + a real 64-QAM synthesis branch (previously 64-QAM could not be generated at all, so it was never testable or trainable) | `tests/testbench_gen.py` | Tests are now reproducible; 64-QAM is trainable |
| `test_phase3` rewritten: it fed idealized unshaped noiseless symbol arrays that bypass the entire RF chain — not how `extract_features` is ever invoked — and now measures aggregate accuracy over 12 realistic pipeline-processed trials | `tests/test_master.py` | Test now measures something real instead of an unrepresentative input |

### 0.2b Benchmark infrastructure (new, and the prerequisite for everything else)

`tools/benchmark_rx.py` measures the receive chain against known ground truth over a
(modulation × SNR × trials) matrix, reporting pre-FEC BER, AMC accuracy, CFO/baud error and
end-to-end decode success. `benchmarks/baseline.json` holds the recorded baseline so future
changes can be regression-gated rather than eyeballed.

```bash
python -m tools.benchmark_rx                                  # default matrix
python -m tools.benchmark_rx --trials 20 --json out.json      # CI regression gating
```

Ground-truth plumbing was added to `generate_test_vector(..., truth_out=dict)`, which returns
the true coded bits and transmitted symbols. BER is computed alignment-invariantly (±1 mapping
turns Hamming distance into a correlation, so the best alignment over all lags is found in one
pass), which is what makes "the bits are perfect, the framing is not" diagnosable at all.

### 0.2c Ranked blockers, as measured (this replaces guesswork with numbers)

1. **8PSK and 16-QAM demodulation fails even at 30 dB.** 16-QAM's *best* candidate BER is
   0.354 — no rotation/mapping hypothesis recovers the bits at all, so this is a demapper or
   carrier-recovery defect, not noise. This is now the top functional defect.
2. **CFO estimation is unstable on short records.** Payload length alone swings the error by
   an order of magnitude: a 31-byte payload (420 samples) yields a *systematic* ~3125 Hz error
   across seeds, while a 32-byte payload (428 samples) yields ~400 Hz. Since a few kHz of
   residual CFO spins the constellation over the burst, this directly explains defect 1 for
   16-QAM. Fix the estimator's short-record variance before anything downstream.
3. **Baud-rate estimation below ~20 dB.** Substantially improved at high SNR (see §0.2) but
   still degrades badly at 10-15 dB, where it corrupts resampling and hence everything after.
4. **AMC accuracy 68.8%** at ≥20 dB against a >95% target — consistent with the 69-72%
   held-out figure, and bounded by the 6-feature cumulant set (see P2 item 13).

### 0.3 Honest current state

Measured on `tools/benchmark_rx.py` (4 modulations × 3 SNRs × 6 trials):

| Metric (≥20 dB) | Before this work | Now | Target |
|---|---|---|---|
| BPSK decode success @30 dB | ~25% | **100%** | >95% |
| QPSK decode success @30 dB | ~100% (single lucky config) | **100%** (stable) | >95% |
| 8PSK / 16-QAM decode | 0% | 0% (defect §0.2c-1) | >95% |
| Aggregate decode ≥20 dB | 31.2% | **62.5%** on BPSK/QPSK; 31.2% incl. 8PSK/16-QAM | >95% |
| Baud error @30 dB | 40-70 Hz | **6-41 Hz** | <50 Hz |
| `pytest` pass rate | 5-7 of 9, flaky | **7 of 9, stable** | 9 of 9 |

Real, measured progress — BPSK decode went 25% → 100%, and the previously flaky CFO/AMC tests
are now stable. But **this is not yet an industry-grade receiver**, and it should not be
presented as one. 8PSK and 16-QAM do not decode at all, low-SNR behaviour is poor, AMC sits at
~69% against a >95% bar, and the P1/P2 items below (confidence scoring, non-CRC framing, soft
LLRs, the dead components in §3.5) remain open. The honest summary is: the foundation and the
measurement infrastructure now exist, the top defects are precisely characterised with
reproducible evidence, and the remaining work is well-defined but real.

One further defect found while measuring: the CRC-32 brute-force search produces
**false-positive locks** — a BPSK/30 dB run reported `SUCCESS_CRC_LOCKED` with a payload that
does not match ground truth, and a 2-FSK run did the same. Sweeping thousands of hypotheses
(demapper × slip × interleaver × FEC × every candidate payload length) against a 32-bit check
makes coincidental matches inevitable. A lock should be corroborated (e.g. length/header
plausibility, entropy, FEC syndrome consistency) before being reported as success.

---

Audit date: 2026-09-02. Scope: `src/`, `tests/`, `main.py`, `README.md`. All findings below were
verified by reading the actual code paths and by **running** the repo's own test suite and a
custom SNR sweep in a fresh venv (see "How this was tested" at the bottom) — nothing here is
speculative.

---

## 1. Architecture summary

A 7-stage, pure hand-engineered DSP pipeline (no learned demod/decode components run at
inference time), orchestrated by `src/pipeline.py:run_full_pipeline`:

| Stage | Module | Approach |
|---|---|---|
| Ingest | `ingestion.py` | Format sniffing by KL-divergence over dtype candidates, WAV/SigMF support, NaN repair |
| IQ correction | `iq_correction.py` | DC removal, Gram-Schmidt IQ imbalance, AGC |
| SNR/detect | `pipeline.py` (`estimate_snr_m2m4`) | 2nd/4th-moment kurtosis SNR proxy, hard -5 dB cutoff |
| Spectral | `spectral.py` | Blind Fs grid-fit, M-th power (x²/x⁴/x⁸) CFO estimation, polyphase resample to 2 SPS |
| Equalize | `equalizers.py` | 11-tap CMA (JIT), Gram-Schmidt balance, Gardner-style strobe pick |
| Sync | `synchronizer.py` | Decision-directed Costas/PLL (JIT), Barker/CCSDS preamble correlation |
| Classify (AMC) | `classifier.py` | 6D cumulant feature vector vs. **hardcoded** Gaussian centroids, Mahalanobis distance |
| Demod | `demodulators.py` | Gray + natural-binary slicers for BPSK/QPSK/8PSK/16/64-QAM, FM discriminator for FSK |
| De-interleave + FEC + framing | `blackboard.py`, `fec_decoders.py`, `deinterleaver.py` | Brute-force sweep of demap variant × bit-slip × interleaver × FEC type, gated purely by CRC-32 match |

There is no GUI-independent confidence score, no BER/soft-metric output, and no ML model
actually used for AMC despite training code existing (see §3.1).

**What's implemented:** BPSK/QPSK/8PSK/16-QAM/64-QAM/2-FSK/4-FSK/GMSK(label only) demod;
block/diagonal bit-interleavers; convolutional (Viterbi, incl. punctured 2/3,3/4,5/6),
RS(255,223), and IEEE 802.11n LDPC(648,324) FEC; CRC-32 correlation-based framing.

**What's missing/incomplete:** any non-CRC framing/header detection, RS(204,188) (built but
unwired), true Forney convolutional de-interleaving (built but unwired), pseudo-random
de-interleaver descrambling by known LFSR polynomials (only generic random-permutation guesses),
concatenated LDPC+RS, soft-decision demapping feeding the FEC decoders (LLRs are computed in
`demodulators.py` but never used — see §3.4), and any notion of decode confidence short of a
binary CRC pass/fail.

---

## 2. Empirical results — the pipeline does not reliably pass its own tests

Ran `pytest tests/test_master.py -v` three times back-to-back in a clean venv (no code changes).
**4 of 9 tests failed, and results were not repeatable run-to-run**:

| Test | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| `test_phase2_spectral_and_baud_recovery` (CFO/baud) | **FAIL** (25,000 Hz CFO error) | pass | **FAIL** |
| `test_phase6_end_to_end_uncoded_bpsk` (clean BPSK, 25 dB) | **FAIL** (classified as 16-QAM) | pass | **FAIL** |
| `test_phase6_end_to_end_concat_rs_conv_qpsk` (RS+interleave+conv, 25 dB) | **FAIL** | **FAIL** | **FAIL** |
| `test_phase6_end_to_end_ldpc_16qam` (LDPC, 30 dB) | **FAIL** | **FAIL** | **FAIL** |

All other tests (ingestion, phase-3 classification on i.i.d. symbol arrays, phase-4 FEC unit
tests, entropy) passed consistently. Note phase-4's Viterbi/RS/LDPC unit tests pass because they
feed the decoders bits directly — the failures only appear once real waveform synthesis, timing
recovery and AMC are in the loop.

A custom SNR sweep (`generate_test_vector` → `run_full_pipeline`, CFO=120 Hz, 4 mod/FEC combos ×
8 SNR points, script preserved for reproduction) shows the same pattern:

```
QPSK/NONE   : locks ONLY at +30 dB. At +20/+15/+10 dB it misclassifies as 2-FSK/64-QAM/64-QAM.
QPSK/CONV   : locks ONLY at +30 dB. At +20/+15 dB misclassified as 16-QAM; +10 dB and below: NO_SIGNAL or noise.
BPSK/NONE   : locks ONLY at +20 dB in this run (30 dB run misclassified as QPSK!). Non-monotonic w.r.t. SNR.
16-QAM/LDPC : NEVER locks (0/4 SNR points from 30 dB down), correct AMC at 30/20/15 dB but FEC/CRC never validates.
```

Key observations:
- **Success is not a function of SNR** — BPSK/NONE failed at 30 dB but "succeeded" at 20 dB in
  the same run; the uncoded-BPSK unit test flips pass/fail between identical re-runs. This points
  to sensitivity to the specific noise realization / CFO phase, not a graceful SNR rolloff.
- **The -5 dB hard SNR gate misfires long before -5 dB**: `NO_SIGNAL_DETECTED` fired at true
  injected SNRs of +10 dB, +6 dB, +3 dB and 0 dB across different configs, meaning
  `estimate_snr_m2m4` is producing badly biased (too-low) SNR estimates on modulated signals, not
  just rejecting real noise.
- **LDPC and the concatenated RS+interleave+Viterbi track never produce a CRC lock in any run**,
  including at 30 dB with zero multipath — this is a functional bug in the decode chain, not a
  robustness/SNR issue.
- **AMC output is close to random** outside the single calibration point (30 dB, 0 residual
  impairment) — QPSK gets called 2-FSK, 64-QAM, 16-QAM, or GMSK depending on run.

This directly corroborates the number the project's own (now-deleted, see §3.6) benchmark
reported in `README.md`: **80/200 (40.0%) CRC-32 locked decodes** and **151/200 (75.5%)**
modulation accuracy over its self-generated dataset — i.e., even by the project's own prior
measurement, the system was far from "industry-grade," and the current on-disk state performs
worse than that on the four representative end-to-end tests it ships.

---

## 3. Concrete weaknesses by subsystem

### 3.1 AMC (`classifier.py`) — no real model, hand-tuned constants, no calibration data
- `CALIBRATED_CENTROIDS` / `CALIBRATED_PRECISIONS` (classifier.py:6-44) are **hardcoded literal
  arrays** with a comment claiming they're "theoretical/calibrated" — there is no script,
  dataset, or citation producing these exact numbers, and they are not the output of
  `train_classifier.py`.
- **`models/classifier_rf.joblib` and `models/classifier_params.npz` exist on disk but are never
  loaded anywhere in `src/`** (`grep -rn "classifier_rf\|joblib" src/` only hits
  `train_classifier.py`, which *writes* them). A trained RandomForest and empirically-fit
  centroids were built and then abandoned — the pipeline runs on the hand-guessed constants
  instead. This is the single highest-leverage fix available: wire up what's already built.
- The `get_snr_adapted_centroids` SNR-conditioning (classifier.py:49-63) linearly interpolates
  every class centroid toward one shared `NOISE_CENTROID` by a scalar `alpha(SNR)` — this
  preserves relative class geometry only if all classes degrade identically with SNR, which is
  false (e.g., 64-QAM's cumulants collapse toward the noise centroid much faster than BPSK's).
  This is very likely why classification degrades non-monotonically instead of gracefully.
- 6 hand-picked cumulant features with only diagonal-plus-two-off-diagonal covariance terms is a
  1990s-era AMC approach; it has no mechanism to learn from mistakes, no rejection option other
  than the single fixed χ² threshold (22.46), and was calibrated (per the code comments) at fixed
  SNR/CFO/roll-off assumptions not representative of "real-world noisy captures."
- `detect_fsk_tones` (classifier.py:105) short-circuits AMC entirely — if it sees exactly 2 or 4
  histogram peaks in instantaneous frequency it *hard-returns* 2-FSK/4-FSK with a constant
  `ood_dist=0.5`, before the Mahalanobis step runs. Any linear-modulation signal whose noisy
  phase-difference histogram happens to have 2-4 peaks (very plausible under residual CFO/ISI)
  gets force-classified as FSK. Matches the observed "misclassified as 2-FSK/4-FSK" sweep results.

### 3.2 SNR / signal-presence estimation — crude, contradicts the pipeline's own better tool
- `estimate_snr_m2m4` (pipeline.py:14-21) is a single-shot moment-based estimator with a magic
  `k > 1.95` "pure noise" cutoff and a floor of -10 dB / ceiling implicit in the formula; it is
  computed once on the *whole* raw capture, so any burst/idle structure, DC residual, or
  interference biases the entire decision.
- **`detect_signal_mme` in `ingestion.py:237`** — a proper Marcenko-Pastur eigenvalue detector
  the README specifically advertises as working down to **-15 dB SNR** — **is never called by
  `run_full_pipeline`**. The pipeline instead gates on the much cruder kurtosis estimate with a
  hard -5 dB cutoff (pipeline.py:93-112), which the SNR sweep shows misfires (false "no signal")
  at SNRs as high as +10 dB on real modulated signals.
- There are two near-duplicate SNR functions (`estimate_snr_m2m4`, `estimate_snr_kurtosis`) in
  `pipeline.py`; only the first is used, the second is dead code with different (undocumented)
  thresholds — a maintenance hazard as much as a correctness one.
- `detect_bursts` (ingestion.py:284) — a burst/slot energy detector — is also implemented and
  never called; there's no support for pulsed/hopped/slotted transmissions even though the code
  to detect burst windows already exists.

### 3.3 CFO / baud estimation (`spectral.py`) — ambiguity bug, no uncertainty output
- `test_phase2_spectral_and_baud_recovery` reproducibly fails with a CFO error of exactly
  **25,000 Hz = baud_rate/2** on a 50 kHz-baud QPSK signal — i.e., the M-th power estimator
  (`estimate_cfo_and_baud`, spectral.py:72) is picking a spurious spectral line (very likely a
  pulse-shaping sidelobe or the ±Rs/2 image) instead of the true `x²`/`x⁴` carrier tone, on some
  noise realizations but not others. The current disambiguation only compares the three M-th
  power SNR scores against each other and a coarse PSD centroid; it has no sanity check against
  the already-estimated baud rate (which would trivially catch a baud/2-sized error).
  - **This is a P0 correctness bug, not a robustness nice-to-have**: it occurs on a *clean* 200
    kHz-Fs / 50 kHz-baud / +CFO-only synthetic vector with no noise-induced ambiguity beyond
    ordinary AWGN.
- No confidence/uncertainty is returned alongside `cfo_hz`/`baud_rate` — downstream stages
  (resampling, PLL) get a single point estimate with no way to know it might be off by a
  half-symbol-rate's worth of frequency.
- `estimate_blind_fs`'s grid-fit (spectral.py:7) snaps to a fixed list of ~27 "standard" Fs
  values and ~22 "standard" baud values; a real capture at a non-standard Fs (very common —
  SDRs are frequently run at arbitrary decimation rates, e.g. 2.048 MHz/N) will be silently
  forced onto the nearest grid point with no mechanism to detect or report the mismatch.

### 3.4 Demod / soft information is computed and then thrown away
- `compute_llr` in `demodulators.py:29` computes real LLRs from a **caller-supplied
  `noise_var`** — but nothing in `pipeline.py` or `blackboard.py` ever estimates a per-signal
  noise variance and passes it in; the function is never called from the pipeline at all.
- The one place soft decoding is attempted, `blackboard.py:346` (punctured-Viterbi track),
  fabricates LLRs from **already hard-sliced bits**: `pseudo_llrs = np.where(raw_bits==0, 4.0,
  -4.0)`. This is hard-decision decoding wearing a soft-decision decoder's clothes — it throws
  away exactly the amplitude information that makes soft Viterbi outperform hard Viterbi (typically
  ~2 dB coding gain lost), and it's the *only* code path that claims to use
  `decode_soft_viterbi`.
- Carrier PLL (`synchronizer.py:96`) maps **both 16-QAM and 64-QAM to the same `mod_code=3`**,
  which uses a fixed 4-level `grid_16qam` decision grid inside the JIT loop
  (`synchronizer.py:20`). For 64-QAM this is the wrong constellation grid (4 levels vs. the
  correct 8), so 64-QAM phase tracking is decision-directed against the wrong reference points —
  a likely contributor to why 64-QAM never appears as a correct classification in any test run.
- GMSK is listed as a supported modulation (README, classifier centroids, PLL `mod_code=0`,
  `demap_linear`'s BPSK branch) but there is no actual Gaussian-filtered CPM/MSK demodulator
  anywhere in the codebase — GMSK signals are demodulated as if they were BPSK, which is
  fundamentally the wrong receiver structure for a continuous-phase FSK-derived scheme.

### 3.5 De-interleaving / FEC / framing (`blackboard.py`) — brute force, not estimation
- The architecture is fundamentally a **CRC-32 oracle search**: try every {first-mode, then up to
  ~7 fallback modulations} × {≤16 bit slips} × {3 block depths, 3 diagonal depths} × {RS with 4
  interleave depths} × {LDPC with 8 slips} × {Viterbi with 2 phases × 8 sub-shifts × 3 candidate
  bit-orderings + 3 random-permutation "pseudo-random interleaver" guesses} × {concatenated RS
  with up to 4×8 block-count/depth combos} × {punctured rates 2/3,3/4,5/6}, stopping the instant
  any branch's CRC-32 matches. There is **no actual interleaver-depth estimation feeding this
  search** — `estimate_interleaver_depth` (deinterleaver.py:4, an autocorrelation-based depth
  estimator) is **imported into `blackboard.py` and never called**; the sweep just tries a fixed
  small set of depths (1,4,8,16) regardless of what the signal's real structure implies.
- **No CRC → no result, ever.** Any real-world protocol that doesn't happen to close with a
  trailing big/little-endian CRC-32 (Rice/BCH checksums, no checksum at all, non-32-bit CRCs,
  continuous unframed telemetry, checksums that aren't over the full payload) can never produce
  `crc_valid=True` no matter how correct the demod/FEC actually is. The fallback path
  (`base_result` in pipeline.py:186-198) picks whichever demap variant maximizes printable-ASCII
  count in the first 64 bytes — a heuristic that only works for English-ish plaintext payloads
  and provides no actual correctness guarantee or confidence number.
- The "pseudo-random interleaver" support (blackboard.py:275-279) is 3 fixed NumPy
  `RandomState` seeds (99, 1, 42) — this only ever matches a signal that happens to have been
  generated with one of those exact seeds (i.e., only the project's own test fixtures, if they
  used one of these seeds). It cannot match any real PN/LFSR-based interleaver used by an actual
  protocol.
- `RS_CODEC_204_188` (DVB-T standard, fec_decoders.py:238) and the true `deinterleave_convolutional`
  Forney interleaver (deinterleaver.py:41) are both implemented and **never referenced by
  `blackboard.py`** — real-world DVB-T-style or convolutionally-interleaved captures have no path
  to a correct decode even though the primitives already exist.
- `estimate_blind_code_rate` (fec_decoders.py:524, GF(2) rank-defect blind FEC-rate detector) is
  implemented, documented in the README ("Blind Channel Coding Rank Defect Estimation... PASSED"
  in the now-deleted red-team suite), and **never called from the pipeline or blackboard** —
  another fully-built capability that isn't wired in.
- `error_metric` on every successful `HypothesisResult` is hardcoded to `0.0`
  (blackboard.py:149,175,201,... — every `return` site). Ranking non-CRC-valid results by
  `error_metric` (pipeline/blackboard sort keys) is therefore a no-op for all "successes" and
  provides no actual measure of how close a near-miss was.

### 3.6 Evidence rot — claims in README no longer verifiable / self-reported numbers are already weak
- `README.md` §4 and §6 reference `test_hardened_suite.py` (12-category adversarial suite),
  `eval_200_parallel.py` (200-file benchmark), `test_iq_dataset.py`, `generate_dataset.py`, and a
  `dataset/` directory of **100 .iq + 100 .wav captures with ground truth** — `git log` shows all
  five were deleted in the 5 most recent commits before this audit, with no replacement. There is
  currently **no sample capture file anywhere in the repository** (`find . -iname "*.iq" -o
  -iname "*.wav"` returns nothing) to validate against, and the large validation datasets/scripts
  the README's benchmark numbers came from are gone.
- Those numbers, while they existed, were not strong: **80/200 (40%) CRC-locked full decodes,
  151/200 (75.5%) modulation accuracy, 102/200 (51%) FEC identification** — i.e. even the
  project's own most favorable self-reported measurement shows the system fails to fully decode
  the majority of its own synthetic test files. "Industry-grade" is not supported by the
  project's own historical evidence, let alone by the current on-disk test results in §2.
- `requirements.txt`/`pyproject.toml` pin `numpy>=1.24.0,<2.0.0`, but `numba` (used for every JIT
  hot path: PLL, CMA, Viterbi, LDPC) has no explicit version pin and current `numba` releases
  require `numpy<2.3` (and specific numpy ABI versions) — an unpinned `numba` install can pull a
  numpy-2.5-incompatible or numpy-1.x-only build and hard-crash at import (`AttributeError:
  numpy._globals has no attribute _signature_descriptor` was reproduced during this audit before
  pinning `numpy<2.3`). Reproducibility of the whole pipeline is one `pip install` away from
  breaking.

---

## 4. Prioritized improvement plan

### P0 — correctness bugs blocking anything else from mattering

0. **[BLOCKING — see §0.1] Rebuild symbol timing recovery and receive filtering.** This gates
   every other accuracy improvement in this document. Required pieces: (a) a real interpolating
   timing-recovery loop — Farrow/polyphase fractional interpolator driven by a Gardner or
   Mueller-Müller TED with a proper proportional-integral loop filter, replacing the
   fixed whole-sample energy strobe; (b) a receive-side matched filter (true RRC matched to the
   transmit pulse shape, which also requires the transmit side to use a real RRC rather than the
   current 31-tap Hamming-windowed `firwin` approximation); (c) actually calling an equalizer,
   with a modulation-appropriate modulus target — CMA with `R2=1.0` is valid only for
   constant-modulus signals and demonstrably *degrades* QAM, so either defer equalization until
   after a coarse AMC pass and re-run it with the right `R2`, or use a multi-modulus (MMA)
   variant. Validate against measured EVM **and** end-to-end decode rate, not one metric alone;
   a change that improves EVM on one modulation while reducing decode success is a regression.
   1. **Fix the CFO half-baud ambiguity** — DONE, see §0.2 (`spectral.py:estimate_cfo_and_baud`): after picking the
   best M-th-power line, sanity-check candidate CFO estimates against the independently-estimated
   baud rate (reject/re-rank a CFO estimate that lands within a small tolerance of ±Rs/2, ±Rs from
   a lower-order candidate) before committing. Add this exact regression (fs=200k, baud=50k,
   cfo=450 Hz, multiple noise seeds) as a seeded, repeated (≥20 trials) test so flaky passes can't
   hide it again.
2. **Root-cause the LDPC and concatenated-RS-Viterbi decode paths, which fail 3/3 runs at 30/25
   dB with zero multipath** — this is a wiring/bit-ordering bug (likely Gray-vs-natural bit
   packing mismatch between `demap_linear`'s 16-QAM output and `encode_ldpc`'s systematic bit
   order, or an off-by-N in the RS block/interleave slip search), not an SNR problem. Debug with a
   loopback test that skips the RF chain entirely (feed `encode_ldpc(bits)` symbols directly into
   `decode_ldpc` via the real demapper, bypassing PLL/CMA, to isolate whether the bug is in
   framing/bit-order vs. in the RF recovery chain).
3. **Pin `numba` (and transitively-compatible `numpy`) in `requirements.txt`/`pyproject.toml`** so
   `pip install -r requirements.txt` doesn't silently produce a broken environment.
4. Make the test suite deterministic: seed `np.random` in `testbench_gen.generate_test_vector`
   (or accept an explicit `rng`) so `pytest` results are reproducible and CI-worthy instead of
   flaky.

### P1 — wire up what's already built (highest ROI: these are implemented, tested in isolation, and just not called)
5. Load and use `models/classifier_rf.joblib` (or re-derive `classifier_params.npz`-style
   empirical centroids) instead of the hand-typed `CALIBRATED_CENTROIDS`/`CALIBRATED_PRECISIONS`
   constants — or delete the dead artifacts and `train_classifier.py` if they're truly abandoned.
   Either way, stop shipping a trained model that influences nothing.
6. Call `detect_signal_mme` (already implemented, -15 dB capable) as the primary signal-presence
   gate in `run_full_pipeline`, replacing/supplementing the single-shot `estimate_snr_m2m4` hard
   cutoff. Delete the unused `estimate_snr_kurtosis` duplicate.
7. Call `estimate_interleaver_depth` to seed the blackboard's depth search instead of the fixed
   `(1,4,8,16)` list; wire in `deinterleave_convolutional` and `RS_CODEC_204_188` as additional
   blackboard tracks; wire in `estimate_blind_code_rate` as a prefilter to skip FEC tracks that
   the rank-defect test says don't match, cutting the combinatorial search instead of brute-forcing
   every track on every capture.
8. Feed real per-branch noise-variance estimates into `compute_llr` and use genuine soft LLRs
   (not the `±4.0` hard-decision proxy at blackboard.py:346) for the punctured-Viterbi and,
   ideally, the standard Viterbi and LDPC tracks — this is a well-understood ~2 dB coding-gain
   improvement with the decoder infrastructure already in place.
9. Fix the 64-QAM PLL decision grid (`synchronizer.py`: give `mod_code=3` a real 8-level grid, or
   add a distinct `mod_code=4` for 64-QAM) instead of sharing 16-QAM's 4-level grid.

### P2 — architecture/model improvements for real "industry-grade" generalization
10. **Confidence scoring**: replace the binary `crc_valid` as the sole success signal with a fused
    confidence score per hypothesis (e.g., combining EVM, OOD Mahalanobis distance, FEC syndrome
    weight / post-decode residual error count, and CRC status) so the tool can report "best-effort,
    62% confidence" instead of only ever emitting `SUCCESS`/`DEMOD_NO_CRC` with no gradient between
    them. Surface this number in the GUI and CLI/JSON output.
11. **Header/payload correlation without requiring CRC-32 specifically**: generalize
    `_check_crc` into a pluggable frame-validator (CRC-16/32 variants, fixed sync-word + length
    field parsing, BCH/checksum-8, or plain bit-stream autocorrelation against a repeating
    header/preamble pattern per the PS's "bit-stream correlation to identify header/payload"
    requirement) so protocols without a trailing CRC-32 aren't unconditionally unrecoverable.
12. **Real AMC generalization**: either (a) properly train and ship the RandomForest (or a small
    CNN/ResNet over constellation-density or spectrogram images — a much more standard
    "industry-grade" AMC feature space than 6 hand-picked cumulants) across a wide sweep of
    SNR/CFO/roll-off/multipath/oversampling combinations with the classifier trained end-to-end
    on the *actual* RF chain (post-CMA/PLL, matching inference-time distribution) rather than
    idealized symbol arrays; or (b) if staying with cumulants, replace the broken linear SNR
    centroid interpolation (§3.1) with per-class, per-SNR-bin empirically fit centroids/covariances
    and a proper rejection/abstain option instead of the fixed χ² cutoff.
13. **Constellation/waterfall feature engineering**: add 2D constellation-density histograms and
    short-time spectrogram/cyclic-spectrum features as auxiliary AMC inputs (these are what
    actually distinguish 16 vs 64-QAM and PSK-vs-QAM robustly at low-to-mid SNR — the current 6D
    cumulant vector alone cannot).
14. **Sampling-rate/format generalization**: replace the fixed "standard Fs/baud grid" snap in
    `estimate_blind_fs` with a continuous estimate plus a *reported uncertainty*, only falling back
    to the grid as a tie-breaking prior, so non-standard capture rates (very common with real SDR
    hardware) aren't silently misassigned to the nearest textbook value.
15. **Multipath/impulsive-noise robustness**: `apply_twopass_cma` runs a single fixed-mu 11-tap CMA
    pass; for HF especially (explicitly in scope per the problem statement), consider a
    decision-directed LMS/RLS second pass after CMA convergence, and evaluate robustness under
    non-Gaussian/impulsive noise models (HF atmospheric noise), not just AWGN — none of the current
    tests inject non-Gaussian noise.
16. Re-build (or restore, if recoverable via `git show <deleted-commit>^:dataset/`) a real,
    version-controlled benchmark dataset + ground truth + an automated benchmark script that runs
    in CI, so future changes have a regression signal instead of relying on 9 hand-picked unit
    tests that themselves are flaky.

---

## 5. How this was tested (for reproduction)

```bash
python3 -m venv .venv-audit && source .venv-audit/bin/activate
pip install "numpy<2.3,>=1.24" "scipy>=1.10.0" "scikit-commpy>=0.8.0" \
            "galois>=0.3.6" "scikit-learn>=1.3.0" "pytest>=7.4.0" numba joblib

# Repo's own suite, run 3x to see flakiness:
python -m pytest tests/test_master.py -v

# SNR sweep (ad hoc, not committed — recreate from tests/testbench_gen.generate_test_vector
# and src/pipeline.run_full_pipeline as shown in §2 of this file):
#   for (mod, fec) in [(QPSK,NONE), (QPSK,CONV), (BPSK,NONE), (16-QAM,LDPC)]:
#     for snr_db in [30,20,15,10,6,3,0,-3]: generate_test_vector(...) -> run_full_pipeline(...)
```

No `.iq`/`.wav` sample files exist anywhere in the repository (confirmed via `find . -iname
"*.iq" -o -iname "*.wav"`), so all end-to-end testing here used the project's own
`tests/testbench_gen.generate_test_vector` synthetic signal generator — the same generator the
project's own unit tests use.
