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

MOD_CLASSES = ["BPSK", "QPSK", "8PSK", "16-QAM", "2-FSK"]

def generate_training_dataset(samples_per_class: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Generates synthetic RF signal captures across diverse SNRs and CFOs, returning (X, y)."""
    X_list = []
    y_list = []

    snr_levels = [15.0, 20.0, 25.0, 30.0]
    cfo_levels = [0.0, 25.0, 50.0, 100.0]

    for label_idx, mod in enumerate(MOD_CLASSES):
        print(f"Generating training vectors for {mod}...")
        count = 0
        while count < samples_per_class:
            for snr in snr_levels:
                for cfo in cfo_levels:
                    if count >= samples_per_class:
                        break
                    try:
                        rx_wave, _ = generate_test_vector(
                            mod_type=mod,
                            cfo_hz=cfo,
                            snr_db=snr
                        )
                        cfo_est, rs, sps, x_bb = estimate_cfo_and_baud(rx_wave, 200000.0)
                        x_2sps = resample_to_2sps(x_bb, sps)
                        y_syms = apply_twopass_cma(x_2sps)
                        feat = extract_features(y_syms)
                        if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                            X_list.append(feat)
                            y_list.append(label_idx)
                            count += 1
                    except Exception:
                        continue

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)

def train_and_save_classifier(output_dir: str = "models"):
    """Trains both Mahalanobis Gaussian centroids and a Random Forest Classifier."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic RF training dataset...")
    X, y = generate_training_dataset(samples_per_class=30)
    print(f"Dataset generated: {X.shape[0]} samples with 6D features.")

    # 1. Compute empirical Mahalanobis Gaussian centroids and variances per class
    centroids = {}
    variances = {}
    for label_idx, mod in enumerate(MOD_CLASSES):
        mask = (y == label_idx)
        X_c = X[mask]
        mu = np.mean(X_c, axis=0)
        var = np.var(X_c, axis=0) + 1e-4  # regularized variance
        centroids[mod] = mu
        variances[mod] = var
        print(f"Class {mod:7s} -> Centroid: {np.round(mu, 2)} | Var: {np.round(var, 3)}")

    # 2. Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\nRandom Forest Training Accuracy: {acc * 100:.2f}%")
    print(classification_report(y, y_pred, target_names=MOD_CLASSES))

    # 3. Save artifacts
    npz_path = os.path.join(output_dir, "classifier_params.npz")
    np.savez(
        npz_path,
        classes=MOD_CLASSES,
        **{f"centroid_{k}": v for k, v in centroids.items()},
        **{f"variance_{k}": v for k, v in variances.items()}
    )
    
    rf_path = os.path.join(output_dir, "classifier_rf.joblib")
    joblib.dump(rf, rf_path)
    print(f"Model artifacts successfully saved to:\n - {npz_path}\n - {rf_path}")

if __name__ == "__main__":
    train_and_save_classifier()
