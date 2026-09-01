"""PS 26147 Universal Blind SDR Interceptor - Unified CLI & Entry Point."""
import argparse
import sys
import os
import json
import numpy as np

from src.pipeline import run_full_pipeline

def main():
    parser = argparse.ArgumentParser(
        description="PS 26147: Universal Blind SDR Signal Interceptor & Multi-Hypothesis Blackboard Demodulator"
    )
    parser.add_argument("--file", "-f", type=str, help="Path to input raw/IQ SDR capture file.")
    parser.add_argument("--fs", type=float, default=None, help="Optional capture sample rate in Hz. If omitted, Fs is blindly determined automatically.")
    parser.add_argument("--meta", type=str, default=None, help="Path to capture metadata JSON.")
    parser.add_argument("--json", action="store_true", help="Print structured pipeline result as JSON.")
    parser.add_argument("--gui", action="store_true", help="Launch PyQt6 interactive graphical interceptor.")
    parser.add_argument("--testbench", action="store_true", help="Run automated master integration testbench.")

    args = parser.parse_args()

    if args.gui:
        from src.gui.app import launch_gui
        launch_gui()
        return

    if args.testbench:
        import pytest
        sys.exit(pytest.main(["tests/test_master.py", "-v"]))

    if not args.file:
        parser.print_help()
        print("\n[!] Please specify a capture file with --file <path> or launch the GUI with --gui.")
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"[!] File not found: {args.file}")
        sys.exit(1)

    # Execute core pipeline
    result = run_full_pipeline(args.file, user_fs=args.fs, meta_path=args.meta)

    # Format output
    payload_bytes = result.get('payload', b'')
    payload_ascii = ''.join([chr(b) if 32 <= b <= 126 or b in (10, 13) else '.' for b in payload_bytes]) if isinstance(payload_bytes, bytes) else str(payload_bytes)
    payload_hex = payload_bytes.hex() if isinstance(payload_bytes, bytes) else ""

    if args.json:
        # Exclude raw numpy arrays for JSON serialization
        clean_res = {
            k: v for k, v in result.items() 
            if not isinstance(v, (np.ndarray, bytes))
        }
        clean_res['payload_ascii'] = payload_ascii
        clean_res['payload_hex'] = payload_hex
        print(json.dumps(clean_res, indent=2))
    else:
        print("\n" + "=" * 60)
        print("  PS 26147 UNIVERSAL BLIND SDR INTERCEPTOR RESULTS")
        print("=" * 60)
        fs_mode = "Blindly Determined" if result.get('fs_estimated', True) else "User-Specified"
        print(f"Sample Rate (Fs):  {result.get('fs_hz', 0.0):.1f} Hz ({fs_mode})")
        print(f"Status:            {result['status']}")
        print(f"CRC Valid:         {result['crc_valid']}")
        print(f"Winning Branch:    {result.get('branch_name', 'NONE')}")
        print(f"Detected Mod:      {result.get('modulation', 'UNKNOWN')}")
        print(f"CFO Estimate:      {result.get('cfo_hz', 0.0):.3f} Hz")
        print(f"Baud Rate:         {result.get('baud_rate', 0.0):.2f} Baud (Est SPS: {result.get('sps', 0.0):.2f})")
        print(f"FEC Type:          {result.get('fec_type', 'NONE')}")
        print(f"Interleaver:       {result.get('interleaver', 'NONE')}")
        print(f"Bit Slip:          {result.get('bit_slip', 0)}")
        print(f"Entropy:           {result.get('entropy', 0.0):.4f} bits/byte ({result.get('payload_class', 'UNKNOWN')})")
        print("-" * 60)
        print("Decoded Payload (ASCII):")
        print(payload_ascii)
        print("-" * 60)
        print("Decoded Payload (HEX):")
        print(payload_hex)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
