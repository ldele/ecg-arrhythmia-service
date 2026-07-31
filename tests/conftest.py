"""Shared test fixtures.

The API integration tests need a checkpoint on disk. If a trained artifact is
available (`MODEL_PATH`, default `models/model.pt`) the tests run against it;
otherwise they run against a randomly-initialised ECGNet saved to a temp file.
The API tests assert on response shape and value ranges, not on prediction
quality, so untrained weights exercise exactly the same code path. That is what
lets CI, which has no trained model, run them instead of skipping them.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from src.model.architecture import ECGNet
from src.model.dataset import N_CLASSES


@pytest.fixture(scope="session")
def model_path(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Path to a loadable checkpoint, exported as MODEL_PATH for the app."""
    trained = Path(os.environ.get("MODEL_PATH", "models/model.pt"))
    if trained.exists():
        yield trained
        return

    untrained = tmp_path_factory.mktemp("model") / "model.pt"
    torch.save(ECGNet(n_classes=N_CLASSES).state_dict(), untrained)

    previous = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = str(untrained)
    yield untrained
    if previous is None:
        del os.environ["MODEL_PATH"]
    else:
        os.environ["MODEL_PATH"] = previous
