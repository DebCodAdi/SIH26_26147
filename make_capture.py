"""
make_capture.py - generate synthetic SDR capture files for testing the interceptor.

Put this in the SIH26_26147 folder (same level as main.py) and run:

    python make_capture.py

It writes several .cf32 capture files plus a truth_<name>.txt for each,
so you can compare what the pipeline reports against what was really sent.
"""
import numpy as np
from tests.testbench_gen import generate_test_vector


# name          modulation   fec     Fs Hz    baud Hz   SNR dB   CFO Hz
CAPTURES = [
    ("bpsk_easy",  "BPSK",  "NONE",  200000.0,  50000.0, 30.0, 350.0),
    ("qpsk_easy",  "QPSK",  "NONE",  200000.0,  50000.0, 30.0, 350.0),
    ("bpsk_sps8",  "BPSK",  "NONE",  400000.0,  50000.0, 30.0, 350.0),
    ("qpsk_noisy", "QPSK",  "NONE",  200000.0,  50000.0, 15.0, 350.0),
    ("bpsk_conv",  "BPSK",  "CONV",  200000.0,  50000.0, 30.0, 350.0),
]


def main():
    for name, mod, fec, fs, baud, snr, cfo in CAPTURES:
        truth = {}
        iq, payload = generate_test_vector(
            mod_type=mod,
            fec_type=fec,
            fs=fs,
            baud_rate=baud,
            snr_db=snr,
            cfo_hz=cfo,
            seed=1234,          # fixed seed = same file every run, so results are comparable
            truth_out=truth,
        )

        iq = iq.astype(np.complex64)
        iq.tofile(f"{name}.cf32")

        lines = [
            f"file        : {name}.cf32",
            f"modulation  : {mod}",
            f"fec         : {fec}",
            f"Fs (Hz)     : {fs:.1f}",
            f"baud (Hz)   : {baud:.1f}",
            f"SPS         : {truth.get('sps', fs / baud):.2f}",
            f"CFO (Hz)    : {cfo:.1f}",
            f"SNR (dB)    : {snr:.1f}",
            f"payload     : {payload.decode('ascii', 'replace')}",
            f"samples     : {len(iq)}",
            f"size (bytes): {iq.nbytes}",
        ]
        with open(f"truth_{name}.txt", "w") as fh:
            fh.write("\n".join(lines) + "\n")

        print(f"[+] {name}.cf32  ({len(iq)} samples, SPS={fs/baud:.1f}, {snr:.0f} dB)")

    print("\nRun one with:")
    print("    python main.py --file bpsk_easy.cf32 --fs 200000")


if __name__ == "__main__":
    main()
