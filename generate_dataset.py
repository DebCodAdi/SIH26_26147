"""Synthetic Dataset Generator for PS 26147 Blind SDR Interceptor.

Generates:
- 100 .iq files in dataset/iq/
- 100 .wav files in dataset/wav/
- dataset/ground_truth.json (complete metadata catalog)
- dataset/ground_truth.csv (spreadsheet catalog)
"""
import os
import csv
import json
import zlib
import numpy as np
import scipy.io.wavfile
from tests.testbench_gen import generate_test_vector

PLAINTEXT_TEMPLATES = [
    "TACTICAL_SITREP_UNIT_ALPHA_7: SECTOR_BRAVO_RECON_COMPLETED_STATUS_GREEN_ALL_SYSTEMS_GO",
    "ACARS_AERODROME_FLIGHT_IX204: POSITION_LAT_28.6139_LON_77.2090_ALT_34000FT_SPEED_480KTS",
    "COSPAS_SARSAT_EMERGENCY_LOCATOR_HEX_A3F982B001: LAT_12.9716_LON_77.5946_STATUS_ACTIVE",
    "AIS_MARINE_SURVEILLANCE_MMSI_419000888: COURSE_270_SPEED_14_2KTS_CARGO_CONTAINER_VESSEL",
    "WEATHER_OBSERVATION_METAR_VEGT: WIND_120_10KT_VIS_9999_SKC_TEMP_28_DP_22_QNH_1012_NOSIG",
    "TELEMETRY_SUBSYSTEM_HK_PACKET_99: BATTERY_V_28_4_SOLAR_PWR_145W_CPU_TEMP_42C_BUS_OK",
    "RADAR_TRACK_REPORT_AZ_045_EL_12_RNG_85KM: DOPPLER_FREQ_SHIFT_320HZ_CONFIRM_FRIENDLY",
    "GROUND_STATION_TELEMETRY_LINK_SYNC: FRAME_SEQ_10482_RSSI_NEG72DBM_SNR_24DB_LOCK_OK",
    "HF_MARITIME_VOICE_DIGITAL_SELECTIVE_CALL: MMSI_004192000_URGENCY_PRIORITY_CHANNEL_16",
    "UHF_SATCOM_DAMA_CHANNEL_BURST: TIME_SLOT_08_BURST_ID_7781_MODULATION_LOCKED_CRC32_OK"
]

def generate_full_dataset(base_dir: str = "dataset"):
    iq_dir = os.path.join(base_dir, "iq")
    wav_dir = os.path.join(base_dir, "wav")
    os.makedirs(iq_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    catalog = {}
    csv_rows = []

    # Modulation and FEC pools
    linear_mods = ["BPSK", "QPSK", "8PSK", "16-QAM"]
    fsk_mods = ["2-FSK", "4-FSK"]
    all_mods = linear_mods * 3 + fsk_mods * 2  # balanced distribution

    fec_types = ["NONE", "CONV", "RS(255,223)", "CONV+RS(255,223)", "LDPC"]
    interleaver_depths = [1, 4, 8, 16]
    sample_rates = [200000.0, 250000.0, 500000.0]

    np.random.seed(42)

    total_files = 100

    print("=" * 60)
    print(f"Generating PS 26147 Test Dataset (100 .iq + 100 .wav files)...")
    print("=" * 60)

    # 1. Generate 100 .IQ files
    for idx in range(1, total_files + 1):
        filename = f"capture_{idx:03d}.iq"
        filepath = os.path.join(iq_dir, filename)

        mod = all_mods[idx % len(all_mods)]
        fec = fec_types[idx % len(fec_types)]
        depth = interleaver_depths[idx % len(interleaver_depths)] if "RS" in fec else 1
        fs = sample_rates[idx % len(sample_rates)]
        baud = 50000.0
        cfo = float(np.random.uniform(-800.0, 800.0))
        snr = float(np.random.uniform(18.0, 32.0))

        # Payload choice
        is_encrypted = (idx % 4 == 0)
        if is_encrypted:
            raw_payload = np.random.bytes(64)
            payload_type = "ENCRYPTED"
        else:
            base_txt = PLAINTEXT_TEMPLATES[idx % len(PLAINTEXT_TEMPLATES)]
            if "CONV+RS" in fec:
                raw_payload = (base_txt.encode('ascii') * 3)[:245]
            elif "RS" in fec:
                raw_payload = (base_txt.encode('ascii') * 3)[:219]
            else:
                raw_payload = f"{base_txt}_PKT_{idx:03d}".encode('ascii')
            payload_type = "PLAINTEXT"

        rx_wave, gt_payload = generate_test_vector(
            mod_type=mod,
            fec_type=fec,
            interleaver_depth=depth,
            payload_text=raw_payload,
            fs=fs,
            baud_rate=baud,
            cfo_hz=cfo,
            snr_db=snr
        )

        # Save binary IQ complex64
        rx_wave.astype(np.complex64).tofile(filepath)

        record = {
            "id": f"IQ_{idx:03d}",
            "filename": filename,
            "filepath": filepath,
            "format": "iq",
            "sample_rate": fs,
            "baud_rate": baud,
            "sps": fs / baud,
            "modulation": mod,
            "fec_type": fec,
            "interleaver_depth": depth,
            "cfo_hz": round(cfo, 2),
            "snr_db": round(snr, 2),
            "payload_type": payload_type,
            "payload_len_bytes": len(gt_payload),
            "payload_ascii": gt_payload.decode('ascii', errors='replace') if payload_type == "PLAINTEXT" else "",
            "payload_hex": gt_payload.hex()
        }
        catalog[filename] = record
        csv_rows.append(record)

    print(f"[+] Successfully generated 100 .iq files in '{iq_dir}'")

    # 2. Generate 100 .WAV files
    for idx in range(1, total_files + 1):
        filename = f"capture_{idx:03d}.wav"
        filepath = os.path.join(wav_dir, filename)

        mod = all_mods[(idx + 2) % len(all_mods)]
        fec = fec_types[(idx + 1) % len(fec_types)]
        depth = interleaver_depths[(idx + 3) % len(interleaver_depths)] if "RS" in fec else 1
        fs = 200000.0
        baud = 50000.0
        cfo = float(np.random.uniform(-600.0, 600.0))
        snr = float(np.random.uniform(20.0, 34.0))

        # Payload choice
        is_encrypted = (idx % 5 == 0)
        if is_encrypted:
            raw_payload = np.random.bytes(64)
            payload_type = "ENCRYPTED"
        else:
            base_txt = PLAINTEXT_TEMPLATES[(idx + 3) % len(PLAINTEXT_TEMPLATES)]
            if "CONV+RS" in fec:
                raw_payload = (base_txt.encode('ascii') * 3)[:245]
            elif "RS" in fec:
                raw_payload = (base_txt.encode('ascii') * 3)[:219]
            else:
                raw_payload = f"{base_txt}_WAV_{idx:03d}".encode('ascii')
            payload_type = "PLAINTEXT"

        rx_wave, gt_payload = generate_test_vector(
            mod_type=mod,
            fec_type=fec,
            interleaver_depth=depth,
            payload_text=raw_payload,
            fs=fs,
            baud_rate=baud,
            cfo_hz=cfo,
            snr_db=snr
        )

        # Scale and write 16-bit stereo WAV (Left=I, Right=Q)
        peak = np.max(np.abs(rx_wave)) + 1e-12
        rx_norm = (rx_wave / peak) * 0.95
        i_int16 = (np.real(rx_norm) * 32767).astype(np.int16)
        q_int16 = (np.imag(rx_norm) * 32767).astype(np.int16)
        stereo_data = np.column_stack([i_int16, q_int16])
        scipy.io.wavfile.write(filepath, int(fs), stereo_data)

        record = {
            "id": f"WAV_{idx:03d}",
            "filename": filename,
            "filepath": filepath,
            "format": "wav",
            "sample_rate": fs,
            "baud_rate": baud,
            "sps": fs / baud,
            "modulation": mod,
            "fec_type": fec,
            "interleaver_depth": depth,
            "cfo_hz": round(cfo, 2),
            "snr_db": round(snr, 2),
            "payload_type": payload_type,
            "payload_len_bytes": len(gt_payload),
            "payload_ascii": gt_payload.decode('ascii', errors='replace') if payload_type == "PLAINTEXT" else "",
            "payload_hex": gt_payload.hex()
        }
        catalog[filename] = record
        csv_rows.append(record)

    print(f"[+] Successfully generated 100 .wav files in '{wav_dir}'")

    # 3. Write ground_truth.json
    gt_json_path = os.path.join(base_dir, "ground_truth.json")
    with open(gt_json_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"[+] Saved ground truth catalog to '{gt_json_path}'")

    # 4. Write ground_truth.csv
    gt_csv_path = os.path.join(base_dir, "ground_truth.csv")
    csv_fields = [
        "id", "filename", "format", "sample_rate", "baud_rate", "sps",
        "modulation", "fec_type", "interleaver_depth", "cfo_hz", "snr_db",
        "payload_type", "payload_len_bytes", "payload_ascii", "payload_hex"
    ]
    with open(gt_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[+] Saved CSV summary to '{gt_csv_path}'")
    print("=" * 60)
    print(f"Total files generated: {len(catalog)} (100 .iq + 100 .wav)")
    print("=" * 60)

if __name__ == "__main__":
    generate_full_dataset()
