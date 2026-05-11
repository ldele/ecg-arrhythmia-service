"""Train the 1D CNN on the preprocessed DS1, evaluate on DS2."""
from __future__ import annotations

from pathlib import Path

from src.model.dataset import EcgBeatDataset
from src.model.train import TrainConfig, train


def main() -> None:
    data_dir = Path("data/processed")
    out_dir = Path("models")

    train_ds = EcgBeatDataset(data_dir / "X_train.npy", data_dir / "y_train.npy")
    test_ds = EcgBeatDataset(data_dir / "X_test.npy", data_dir / "y_test.npy")
    print(f"Train: {len(train_ds)} beats | Test: {len(test_ds)} beats")

    train(train_ds, test_ds, TrainConfig(), out_dir)


if __name__ == "__main__":
    main()