"""PyTorch Dataset wrapping the preprocessed (X, y) arrays."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Class Q is excluded from training: too few examples (<20 total) to learn from.
N_CLASSES = 4
KEEP_LABELS = {0, 1, 2, 3}  # N, S, V, F


class EcgBeatDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads preprocessed beat windows from .npy files."""

    def __init__(self, x_path: Path, y_path: Path) -> None:
        X = np.load(x_path)
        y = np.load(y_path)
        mask = np.isin(y, list(KEEP_LABELS))
        self.X = X[mask].astype(np.float32)
        self.y = y[mask].astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Add channel dim: (window_length,) -> (1, window_length)
        x = torch.from_numpy(self.X[idx]).unsqueeze(0)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y