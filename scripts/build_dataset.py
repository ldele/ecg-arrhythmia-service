"""Build train (DS1) and test (DS2) datasets, save as .npy."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from src.data.build_dataset import (
    AAMI_CLASSES,
    DS1_RECORDS,
    DS2_RECORDS,
    build_dataset,
)


def _report(name: str, y: np.ndarray) -> None:
    counts = Counter(y.tolist())
    total = len(y)
    print(f"{name}: {total} beats")
    for idx, cls in enumerate(AAMI_CLASSES):
        n = counts.get(idx, 0)
        pct = 100 * n / total if total else 0
        print(f"  {cls} ({idx}): {n:>6} ({pct:5.2f}%)")


def main() -> None:
    data_dir = Path("data/raw/mitdb")
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building DS1 (train)...")
    train = build_dataset(DS1_RECORDS, data_dir)
    np.save(out_dir / "X_train.npy", train.X)
    np.save(out_dir / "y_train.npy", train.y)
    _report("DS1", train.y)

    print("\nBuilding DS2 (test)...")
    test = build_dataset(DS2_RECORDS, data_dir)
    np.save(out_dir / "X_test.npy", test.X)
    np.save(out_dir / "y_test.npy", test.y)
    _report("DS2", test.y)

    print(f"\nSaved to {out_dir.resolve()}")
    print(f"X_train: {train.X.shape}, X_test: {test.X.shape}")


if __name__ == "__main__":
    main()