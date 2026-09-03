# PS 26147 — Demo Talking Points

**Problem statement:** *Automated model for analysis of .IQ and .wav files along with signal
parameter extraction.*

---

## 1. The project in plain language

A radio receiver normally knows what it is listening to — the sample rate, the modulation, how
the data was encoded. This tool assumes **none of that**. You hand it a raw capture file
(`.iq`, `.cf32`, `.raw`, or `.wav`) and it works out, from the samples alone:

- how fast the symbols are being sent (baud rate) and how many samples there are per symbol,
- how far off the carrier frequency is (frequency offset),
- which modulation scheme was used (BPSK, QPSK, 8PSK, QAM, FSK),
- how the bits were protected against errors (convolutional/Viterbi, Reed-Solomon, LDPC),
- how the bits were shuffled (interleaving),

and then reverses all of it to recover the original message, confirming success with a CRC-32
checksum. The GUI shows the spectrum, the constellation, and a time-frequency waterfall.

The core idea is a **multi-hypothesis blackboard**: because we cannot know the modulation,
bit ordering, interleaver depth or FEC scheme in advance, the system generates many candidate
interpretations in parallel and lets the CRC-32 decide which one was right.

---

## 2. What actually works — with measured numbers

All numbers below come from `tools/benchmark_rx.py` (ground-truth bit-error-rate harness) and
`demo/run_demo.py`. They are reproducible, not estimates.

| PS requirement | Status | Evidence |
|---|---|---|
| **(i) Sampling frequency** | Exact from `.wav` RIFF header (48/96/192 kHz verified). For headerless raw `.iq` it is a prior-based guess — see gap G1 | `demo/run_demo.py` Case 3 |
| **(i) Modulation ID** | 88% at 30 dB for BPSK; ~69% aggregate ≥20 dB across 4 modulations | `benchmark_rx.py` AMC column |
| **(i) Baud rate / SPS** | Error 10-46 Hz on 50 000 Bd at 20-30 dB (≈0.05%); SPS exact to 0.01 | `benchmark_rx.py` Bauderr |
| **(i) CFO** | 0.3-1.9 Hz error at 20-30 dB on a 150 Hz offset | `benchmark_rx.py` CFOerr |
| **(i) FEC identification** | Correctly reports `CONV_K7_R1/2` end-to-end | `run_demo.py` Case 2 |
| **(ii) Demodulate PSK** | **BER exactly 0.0000**, decode 100% at 30 dB (BPSK, QPSK) | `benchmark_rx.py` |
| **(ii) Demodulate QAM** | Implemented (16/64-QAM demappers) but does **not** decode — see gap G2 | `benchmark_rx.py` 16-QAM row |
| **(ii) Demodulate FSK** | 2-FSK/4-FSK discriminator implemented; classification of FSK is strong, end-to-end decode not demonstrated | `src/demodulators.py` |
| **(iii) Block de-interleave** | Real, swept at depths 1/4/8/16 | `src/deinterleaver.py:33` |
| **(iii) Diagonal de-interleave** | Real, implemented inline, depths 4/8/16 | `src/blackboard.py:203` |
| **(iii) Convolutional de-interleave** | Function exists (Forney) but is **not wired** into the search — gap G3 | `src/deinterleaver.py:41` |
| **(iii) Pseudo-random de-interleave** | Only 3 hardcoded seeds — gap G4 | `src/blackboard.py:294` |
| **(iv) Convolutional + Viterbi** | **Works end-to-end**, K=7 rate-1/2, CRC-verified | `run_demo.py` Case 2 |
| **(iv) Reed-Solomon** | RS(255,223) decoder verified in unit test (corrects 8 byte errors) | `test_phase4_fec_decoders` |
| **(iv) Concatenated** | Decoder path exists; end-to-end test currently fails — gap G5 | `test_phase6_..._concat` |
| **(iv) LDPC** | 802.11n (648,324) decoder verified in unit test; end-to-end fails — gap G5 | `test_phase4_fec_decoders` |
| **(v) Bit-stream correlation** | Preamble (Barker/CCSDS) correlation only; header/payload segmentation **not implemented** — gap G6 | `src/synchronizer.py` |
| **GUI: spectrum** | Live PSD plot | `src/gui/app.py` |
| **GUI: constellation** | Live IQ scatter | `src/gui/app.py` |
| **GUI: waterfall** | **Time-frequency spectrogram, works for `.iq` and `.wav`** | `demo/waterfall_*.png` |

**Headline demo result:** `python demo/run_demo.py --runs 5` → **5/5 runs, 3/3 cases, 9/9
parameters each**, covering BPSK, QPSK+Viterbi, and a `.wav` input.

---

## 3. Honest note on each gap

- **G1 — Blind sampling rate.** Absolute Fs is *mathematically unrecoverable* from samples
  alone: a 200 kHz/50 kBaud capture and a 400 kHz/100 kBaud capture produce a bit-identical
  file. Only the **ratio** Rs/Fs is measurable, and we recover that exactly (SPS correct to
  0.01 in every test). Fs comes from the WAV header, a SigMF sidecar, or the operator.
- **G2 — QAM does not decode.** 16-QAM's *best* candidate BER is 0.354 even at 30 dB, so no
  bit-mapping hypothesis recovers it. Root cause is upstream: residual CFO error of ~3125 Hz
  on 8PSK/16-QAM spins the constellation. Diagnosed, not yet fixed.
- **G3 — Convolutional de-interleaver not wired.** The Forney implementation exists and is
  correct; the blackboard search simply never calls it.
- **G4 — Pseudo-random de-interleaving is not real.** It tries 3 fixed RNG seeds, which can
  only match a signal generated with those same seeds. A real implementation needs LFSR
  polynomial estimation.
- **G5 — LDPC and concatenated RS fail end-to-end.** Both decoders pass their unit tests on
  clean bits, so the defect is in the RF chain feeding them, not the decoders.
- **G6 — Header/payload correlation not implemented.** Only preamble correlation exists. The
  system currently identifies the payload via CRC-32 boundary search, not by correlating for a
  repeated header structure.
- **Test suite is flaky** (2-4 failures of 9 per run) because the end-to-end tests call the
  signal generator without a fixed seed, so every run uses different noise.

---

## 4. The three hardest questions a teacher can ask

**Q1. "Your accuracy is only ~69% for modulation classification. Isn't that too low?"**

Yes — against an industry bar of >95% it is low, and we measured it rather than claiming
otherwise. The cause is specific: classification uses a 6-dimensional higher-order-cumulant
feature vector, which separates *families* well (FSK vs PSK is near-perfect) but struggles
between *adjacent orders* of the same family — QPSK vs 8PSK vs 16-QAM sit close together in
cumulant space, especially at low SNR. The fix is not a better threshold, it is a richer
feature set: constellation-density images and cyclic-spectrum features fed to a small CNN,
which is what the published literature uses to exceed 95%. Our classifier is also now trained
on data pushed through the *real* DSP chain (69% held-out, 7 classes, 4-30 dB), rather than
hand-typed constants as before — that change alone is what makes the number trustworthy.

**Q2. "Why does a matched filter make your receiver worse? Every textbook says use one."**

This is our favourite result, because the textbook is right and so is our measurement. A
matched filter is only correct when the transmitter uses the *root* half of a Nyquist pair
(RRC at both ends). Our test signal generator shapes with `firwin`, a **full-Nyquist**
lowpass. Cascading an RRC receive filter onto an already-Nyquist pulse gives RRC × Nyquist,
which is no longer Nyquist — so it lowers noise (EVM improved 13.9% → 1.9%) while introducing
intersymbol interference (BER got *worse*: 0.0000 → 0.134, decode 100% → 50%). We reverted it.
The lesson: EVM and BER can disagree, and BER is the metric that decides. To get the
theoretical matched-filter gain we must first change the transmitter to true RRC.

**Q3. "How do you know it works, and not just on the one file you're demoing?"**

We built a ground-truth benchmark harness (`tools/benchmark_rx.py`) that generates captures
with known transmitted bits, runs the real pipeline, and measures true bit error rate with
alignment-invariant correlation — across a matrix of modulation × SNR × repeated trials, with
results saved to JSON for regression tracking. That harness is how we know BPSK/QPSK BER is
exactly 0.0000 at 30 dB, and it is also how we caught ourselves being wrong: an earlier
diagnosis blamed symbol timing recovery based on EVM, and the BER data disproved it. The demo
script runs 5 independent times with different noise on every run, so what you are seeing is
not one lucky seed.

---

## 5. How to run the demo

```bash
# 1. Environment (once)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. THE MAIN DEMO — 3 known-truth captures, TRUE vs ESTIMATED table, 5 repeats
python demo/run_demo.py --runs 5

# 3. The GUI (spectrum + constellation + waterfall)
python main.py --gui
#    then Browse to demo/demo_qpsk.cf32  (or any .iq/.wav) and press Run

# 4. Single file on the command line
python main.py --file demo/demo_qpsk.cf32 --fs 200000

# 5. The evidence behind the numbers in section 2
python -m tools.benchmark_rx --mods BPSK,QPSK --snrs 30,20 --trials 8

# 6. Unit / integration tests
python -m pytest tests/test_master.py -v
```

**Demo order that shows the best story:** run `demo/run_demo.py --runs 5` first (all green,
9/9 parameters), then open the GUI to show the waterfall and constellation visually, then show
`tools/benchmark_rx.py` as the evidence that the numbers are measured rather than asserted.

---

## 6. Future work (explicitly out of scope for this demo)

1. Fix residual CFO on short records — this is the blocker for 8PSK/16-QAM decoding (G2).
2. True RRC transmit shaping + matched receive filter (see Q2).
3. Real pseudo-random de-interleaving via LFSR polynomial estimation (G4).
4. Bit-stream correlation for header/payload segmentation (G6).
5. Wire the existing Forney convolutional de-interleaver into the blackboard search (G3).
6. CNN-based modulation classification on constellation-density images to push AMC past 95%.
7. Seed the end-to-end tests so the suite is deterministic.
