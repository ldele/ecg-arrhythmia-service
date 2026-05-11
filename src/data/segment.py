"""Beat segmentation: fixed-length windows around annotated R-peaks."""
from __future__ import annotations

import numpy as np

# Window definition (at 360 Hz sampling)
SAMPLES_BEFORE = 250
SAMPLES_AFTER = 400
WINDOW_LENGTH = SAMPLES_BEFORE + SAMPLES_AFTER  # 650


def segment_beats(
    signal: np.ndarray,
    beat_samples: np.ndarray,
    channel: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length windows centered on each annotated beat.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_samples, n_channels), as returned by load_record.
    beat_samples : np.ndarray
        Sample indices of annotated beats.
    channel : int
        Which channel to use. Default 0 (MLII for most records).

    Returns
    -------
    windows : np.ndarray
        Shape (n_valid_beats, window_length), float32.
    valid_indices : np.ndarray
        Indices into beat_samples that produced valid windows (beats
        too close to the start or end of the signal are dropped).
    """
    n_samples = signal.shape[0]
    channel_signal = signal[:, channel].astype(np.float32)

    windows: list[np.ndarray] = []
    valid_indices: list[int] = []

    for i, peak in enumerate(beat_samples):
        start = peak - SAMPLES_BEFORE
        end = peak + SAMPLES_AFTER
        if start < 0 or end > n_samples:
            continue
        windows.append(channel_signal[start:end])
        valid_indices.append(i)

    if not windows:
        return (
            np.empty((0, WINDOW_LENGTH), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    return (
        np.stack(windows, axis=0),
        np.asarray(valid_indices, dtype=np.int64),
    )