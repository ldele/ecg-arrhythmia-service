"""Unit tests for preprocessing."""
from __future__ import annotations

import numpy as np

from src.data.preprocess import bandpass_filter, normalize_window


def test_bandpass_removes_dc_offset() -> None:
    fs = 360
    t = np.arange(fs * 4) / fs
    # 5 Hz sinusoid + large DC offset
    signal = np.sin(2 * np.pi * 5 * t) + 10.0
    filtered = bandpass_filter(signal.reshape(-1, 1), fs)
    # DC should be removed
    assert abs(filtered.mean()) < 0.1


def test_bandpass_attenuates_high_freq() -> None:
    fs = 360
    t = np.arange(fs * 4) / fs
    # 5 Hz (passband) + 100 Hz (stopband)
    inband = np.sin(2 * np.pi * 5 * t)
    outband = np.sin(2 * np.pi * 100 * t)
    signal = (inband + outband).reshape(-1, 1)
    filtered = bandpass_filter(signal, fs).ravel()
    # Filtered signal should be close to the inband component
    rmse = np.sqrt(np.mean((filtered - inband) ** 2))
    assert rmse < 0.15


def test_bandpass_preserves_shape() -> None:
    fs = 360
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((2000, 2))
    filtered = bandpass_filter(signal, fs)
    assert filtered.shape == signal.shape


def test_normalize_window_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    window = rng.standard_normal(650) * 5 + 10
    normalized = normalize_window(window)
    assert abs(normalized.mean()) < 1e-5
    assert abs(normalized.std() - 1.0) < 1e-4


def test_normalize_batch() -> None:
    rng = np.random.default_rng(0)
    batch = rng.standard_normal((32, 650)) * 5 + 10
    normalized = normalize_window(batch)
    # Each window independently normalized
    assert np.allclose(normalized.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(normalized.std(axis=-1), 1.0, atol=1e-4)


def test_normalize_constant_signal_does_not_crash() -> None:
    """Edge case: zero-variance window should produce zeros, not NaN."""
    window = np.full(650, 5.0)
    normalized = normalize_window(window)
    assert not np.isnan(normalized).any()
    assert np.allclose(normalized, 0.0)


def test_normalize_dtype_is_float32() -> None:
    window = np.arange(650, dtype=np.float64)
    normalized = normalize_window(window)
    assert normalized.dtype == np.float32