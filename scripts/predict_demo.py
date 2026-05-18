"""Quick end-to-end inference check on record 100."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.data.load import load_record
from src.model.predict import load_model, predict_signal


def main() -> None:
    record = load_record("100", Path("data/raw/mitdb"))
    signal = record.signal[:, 0]  # channel 0 only
    model = load_model(Path("models/model.pt"))
    preds = predict_signal(signal, fs=record.fs, model=model)

    print(f"Detected {len(preds)} beats (annotation has {len(record.beat_samples)})")
    counts = Counter(p.predicted_class for p in preds)
    print(f"Predicted class distribution: {dict(counts)}")
    print(f"First 3 predictions: {preds[:3]}")


if __name__ == "__main__":
    main()