"""Phase 4: Byte-Level Shannon Entropy & Payload Encryption Classification."""
import numpy as np

def calculate_shannon_entropy(data: bytes) -> float:
    """Computes byte-level Shannon entropy in bits/byte."""
    if len(data) == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(data)
    return -float(np.sum(probs * np.log2(probs)))

def classify_payload_entropy(data: bytes, threshold: float = 7.50) -> str:
    """Classifies payload as plaintext or encrypted/compressed based on Shannon entropy threshold."""
    h = calculate_shannon_entropy(data)
    if h < threshold:
        return "VALID_PAYLOAD_PLAINTEXT"
    return "VALID_PAYLOAD_ENCRYPTED"
