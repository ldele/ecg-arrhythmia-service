"""Integration tests for the inference API.

Runs against the trained checkpoint when one exists and against untrained
weights otherwise; see `model_path` in conftest.py. Nothing here asserts on
prediction quality, only on the contract: status codes, response shape, and
value ranges.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(model_path: Path) -> Iterator[TestClient]:
    # Imported here so model_path has set MODEL_PATH before the app reads it.
    from src.api.main import app

    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_rejects_short_signal(client: TestClient) -> None:
    response = client.post("/predict", json={"signal": [0.0] * 10, "fs": 360})
    assert response.status_code == 422


def test_predict_rejects_bad_fs(client: TestClient) -> None:
    response = client.post(
        "/predict", json={"signal": [0.0] * 1000, "fs": 50}
    )
    assert response.status_code == 422


def test_predict_returns_valid_predictions(client: TestClient) -> None:
    """End-to-end: synthetic signal with periodic spikes -> non-empty predictions."""
    fs = 360
    n_samples = fs * 5  # 5 seconds
    rng = np.random.default_rng(0)
    signal = 0.05 * rng.standard_normal(n_samples)
    # Periodic spikes every ~0.8s to simulate beats and trigger peak detection.
    for peak in range(fs // 2, n_samples, int(0.8 * fs)):
        signal[peak] = 1.5

    response = client.post(
        "/predict", json={"signal": signal.tolist(), "fs": fs}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_beats"] >= 1
    for pred in body["predictions"]:
        assert pred["predicted_class"] in {"N", "S", "V", "F"}
        assert 0.0 <= pred["confidence"] <= 1.0
        assert pred["beat_index"] >= 0


def test_predict_response_schema(client: TestClient) -> None:
    """Response keys match the documented schema."""
    fs = 360
    signal = [0.0] * (fs * 3)
    response = client.post("/predict", json={"signal": signal, "fs": fs})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"n_beats", "predictions"}
