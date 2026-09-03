"""Phase 3 Training & Calibration: Synthesizes labeled RF datasets and trains the Modulation Classifier."""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

from tests.testbench_gen import generate_test_vector
from src.spectral import estimate_cfo_and_baud, resample_to_2sps
from src.equalizers import apply_twopass_cma
from src.classifier import extract_features
from src.synchronizer import track_carrier_pll, resolve_sync_and_rotation

# NOTE: GMSK is intentionally excluded — there is no genuine Gaussian-filtered CPM/MSK waveform
# generator in tests/testbench_gen.py (GMSK signals are currently demodulated as a binary FSK
# discriminator elsewhere in the pipeline; see IMPROVEMENTS.md). Its centroid remains the
# original hand-typed placeholder until a real GMSK synthesizer exists to calibrate against.
MOD_CLASSES = ["BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "2-FSK", "4-FSK"]

def generate_training_dataset(samples_per_class: int = 200, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic RF signal captures across diverse SNRs and CFOs, pushed through the
    *actual production* spectral/equalization/sync chain (matching inference-time processing
    exactly, not idealized symbol arrays), and returns (X, y) feature/label pairs.
    """
    X_list = []
    y_list = []

    # Deliberately spans down into the regime where classification actually gets hard, since
    # a classifier calibrated only at 15-30 dB cannot be expected to generalize below that.
    snr_levels = [4.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0]
    cfo_levels = [0.0, 25.0, 75.0, 150.0, 300.0]
    rng = np.random.default_rng(seed)

    for label_idx, mod in enumerate(MOD_CLASSES):
        print(f"Generating training vectors for {mod}...")
        count = 0
        attempts = 0
        max_attempts = samples_per_class * 20
        while count < samples_per_class and attempts < max_attempts:
            snr = float(rng.choice(snr_levels))
            cfo = float(rng.choice(cfo_levels)) * (1 if rng.random() < 0.5 else -1)
            attempts += 1
            try:
                rx_wave, _ = generate_test_vector(
                    mod_type=mod,
                    cfo_hz=cfo,
                    snr_db=snr,
                    seed=int(rng.integers(0, 2**31 - 1))
                )
                cfo_est, rs, sps, x_bb = estimate_cfo_and_baud(rx_wave, 200000.0)
                if mod in ("2-FSK", "4-FSK"):
                    # Match pipeline.py: FSK features are extracted from PLL-tracked symbols
                    # derived from a QPSK-hypothesis lock, exactly as the live pipeline does.
                    payload_raw, _, _, _ = resolve_sync_and_rotation(x_bb, mod_type="QPSK")
                    y_syms = track_carrier_pll(payload_raw if len(payload_raw) > 0 else x_bb, mod_type="QPSK")
                elif sps <= 1.25 or len(rx_wave) < 512:
                    pwr = float(np.mean(np.abs(x_bb) ** 2))
                    y_syms = (x_bb / np.sqrt(pwr + 1e-12)).astype(np.complex64)
                else:
                    x_2sps = resample_to_2sps(x_bb, sps)
                    y_syms = apply_twopass_cma(x_2sps)
                    if len(y_syms) == 0:
                        y_syms = x_2sps[::2]
                feat = extract_features(y_syms)
                if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)) and np.any(feat != 0):
                    X_list.append(feat)
                    y_list.append(label_idx)
                    count += 1
            except Exception:
                continue

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

def train_and_save_classifier(output_dir: str = "models", samples_per_class: int = 200):
    """Trains both a full-covariance Mahalanobis model and a Random Forest Classifier."""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating synthetic RF training dataset (through the real production DSP chain)...")
    X, y = generate_training_dataset(samples_per_class=samples_per_class)
    print(f"Dataset generated: {X.shape[0]} samples with 6D features.")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 1. Compute empirical Mahalanobis Gaussian centroids and FULL 6x6 covariance matrices
    # per class (a diagonal-only covariance throws away the cross-cumulant correlations that
    # help separate PSK from QAM at the same C40 magnitude).
    centroids = {}
    covariances = {}
    for label_idx, mod in enumerate(MOD_CLASSES):
        mask = (y_train == label_idx)
        X_c = X_train[mask]
        mu = np.mean(X_c, axis=0)
        if X_c.shape[0] >= 8:
            cov = np.cov(X_c, rowvar=False) + np.eye(6) * 1e-3
        else:
            cov = np.diag(np.var(X_c, axis=0) + 1e-3)
        centroids[mod] = mu.astype(np.float64)
        covariances[mod] = cov.astype(np.float64)
        print(f"Class {mod:7s} (n={X_c.shape[0]:4d}) -> Centroid: {np.round(mu, 2)}")

    # 2. Held-out accuracy of the empirical Mahalanobis classifier itself (the number that
    # actually matters for classifier.py's runtime path), not just the auxiliary RF.
    precisions = {mod: np.linalg.inv(cov) for mod, cov in covariances.items()}
    correct = 0
    for xi, yi in zip(X_test, y_test):
        best_mod, best_d = None, float("inf")
        for mod, mu in centroids.items():
            diff = (xi - mu).astype(np.float64)
            d = float(diff @ precisions[mod] @ diff)
            if d < best_d:
                best_d, best_mod = d, mod
        if best_mod == MOD_CLASSES[yi]:
            correct += 1
    maha_acc = correct / max(1, len(y_test))
    print(f"\nHeld-out Mahalanobis classifier accuracy: {maha_acc * 100:.2f}% (n_test={len(y_test)})")

    # 3. Train Random Forest Classifier (auxiliary / cross-check model)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Held-out Random Forest accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=MOD_CLASSES))

    # 4. Save artifacts
    npz_path = os.path.join(output_dir, "classifier_params.npz")
    np.savez(
        npz_path,
        classes=np.array(MOD_CLASSES),
        held_out_accuracy=np.float64(maha_acc),
        **{f"centroid_{k}": v for k, v in centroids.items()},
        **{f"cov_{k}": v for k, v in covariances.items()}
    )

    rf_path = os.path.join(output_dir, "classifier_rf.joblib")
    joblib.dump(rf, rf_path)
    print(f"Model artifacts successfully saved to:\n - {npz_path}\n - {rf_path}")

if __name__ == "__main__":
    train_and_save_classifier()
