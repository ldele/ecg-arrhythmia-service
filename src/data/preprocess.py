"""ECG signal preprocessing: filtering and normalization."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# Bandpass filter parameters (Hz)
LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 4


def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_samples,) or (n_samples, n_channels).
    fs : int
        Sampling frequency in Hz.
    lowcut, highcut : float
        Filter cutoff frequencies in Hz.
    order : int
        Filter order. Effective order is doubled due to filtfilt.

    Returns
    -------
    np.ndarray
        Filtered signal, same shape as input.
    """
    nyquist = fs / 2.0
    sos = butter(
        order,
        [lowcut / nyquist, highcut / nyquist],
        btype="band",
        output="sos",
    )
    # filtfilt along axis 0 (time)
    return sosfiltfilt(sos, signal, axis=0).astype(signal.dtype)


def normalize_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize a single window or batch of windows.

    Parameters
    ----------
    window : np.ndarray
        Shape (window_length,) or (n_windows, window_length).
    eps : float
        Numerical stability for division.

    Returns
    -------
    np.ndarray
        Normalized array, same shape as input, float32.
    """
    arr = np.asarray(window, dtype=np.float32)
    mean = arr.mean(axis=-1, keepdims=True)
    std = arr.std(axis=-1, keepdims=True)
    return (arr - mean) / (std + eps)