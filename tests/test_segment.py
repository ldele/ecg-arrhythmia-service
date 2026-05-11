"""Unit tests for beat segmentation."""
from __future__ import annotations

import numpy as np

from src.data.segment import (
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    WINDOW_LENGTH,
    segment_beats,
)


def _make_signal(n_samples: int = 10_000) -> np.ndarray:
    """Synthetic 2-channel signal with known values for assertion."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((n_samples, 2))


def test_segment_basic_shape() -> None:
    signal = _make_signal()
    beats = np.array([1000, 2000, 3000], dtype=np.int64)
    windows, valid = segment_beats(signal, beats)
    assert windows.shape == (3, WINDOW_LENGTH)
    assert valid.tolist() == [0, 1, 2]


def test_segment_drops_beats_near_start() -> None:
    signal = _make_signal()
    # First beat is too close to start (peak at 100, need 250 before).
    beats = np.array([100, 2000, 3000], dtype=np.int64)
    windows, valid = segment_beats(signal, beats)
    assert windows.shape == (2, WINDOW_LENGTH)
    assert valid.tolist() == [1, 2]


def test_segment_drops_beats_near_end() -> None:
    n = 10_000
    signal = _make_signal(n)
    # Last beat is too close to end (peak at n-100, need 400 after).
    beats = np.array([1000, 2000, n - 100], dtype=np.int64)
    windows, valid = segment_beats(signal, beats)
    assert windows.shape == (2, WINDOW_LENGTH)
    assert valid.tolist() == [0, 1]


def test_segment_window_alignment() -> None:
    """Window should contain channel-0 samples [peak-250, peak+400)."""
    signal = _make_signal()
    peak = 1000
    beats = np.array([peak], dtype=np.int64)
    windows, _ = segment_beats(signal, beats, channel=0)
    expected = signal[peak - SAMPLES_BEFORE : peak + SAMPLES_AFTER, 0].astype(
        np.float32
    )
    np.testing.assert_array_equal(windows[0], expected)


def test_segment_empty_input() -> None:
    signal = _make_signal()
    windows, valid = segment_beats(signal, np.array([], dtype=np.int64))
    assert windows.shape == (0, WINDOW_LENGTH)
    assert valid.shape == (0,)


def test_window_length_constant() -> None:
    assert WINDOW_LENGTH == SAMPLES_BEFORE + SAMPLES_AFTER