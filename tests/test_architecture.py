"""Smoke tests for model architecture."""
from __future__ import annotations

import torch

from src.model.architecture import ECGNet


def test_forward_pass_shape() -> None:
    model = ECGNet(n_classes=4)
    batch = torch.randn(8, 1, 650)
    out = model(batch)
    assert out.shape == (8, 4)


def test_parameter_count_is_small() -> None:
    """Sanity check: model should be small (<100k params)."""
    model = ECGNet(n_classes=4)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, f"Model too large: {n_params} params"


def test_backward_pass_works() -> None:
    model = ECGNet(n_classes=4)
    x = torch.randn(4, 1, 650)
    y = torch.tensor([0, 1, 2, 3])
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()
    # At least one parameter should have a gradient.
    assert any(p.grad is not None for p in model.parameters())