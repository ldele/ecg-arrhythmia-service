"""Inference: raw ECG signal in, per-beat classifications out.

This module is the single source of truth for the inference path. It MUST
produce the same preprocessing as training (bandpass -> segment -> normalize),
otherwise the model sees a different distribution than it was trained on.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

from src.data.preprocess import bandpass_filter, normalize_window
from src.data.segment import segment_beats
from src.model.architecture import ECGNet
from src.model.dataset import N_CLASSES

CLASS_NAMES: tuple[str, ...] = ("N", "S", "V", "F")

# R-peak detection parameters. Tuned for MIT-BIH 360 Hz signals after the
# 0.5-40 Hz bandpass. Not clinical-grade; see README limitations.
PEAK_MIN_DISTANCE_S = 0.25  # 240 bpm max
PEAK_MIN_HEIGHT_SD = 0.5  # relative to filtered-signal std


@dataclass
class BeatPrediction:
    beat_index: int  # sample index of the R-peak in the input signal
    predicted_class: str
    confidence: float


def detect_r_peaks(filtered_signal: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peaks on a bandpass-filtered single-channel ECG.

    Simple amplitude-and-distance peak finder. Adequate for the demo; a
    production system would use Pan-Tompkins or a learned detector.
    """
    min_distance = int(PEAK_MIN_DISTANCE_S * fs)
    threshold = PEAK_MIN_HEIGHT_SD * float(filtered_signal.std())
    peaks, _ = find_peaks(filtered_signal, distance=min_distance, height=threshold)
    return np.asarray(peaks, dtype=np.int64)


def load_model(model_path: Path, device: torch.device | None = None) -> ECGNet:
    """Load a trained ECGNet checkpoint."""
    device = device or torch.device("cpu")
    model = ECGNet(n_classes=N_CLASSES)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def predict_signal(
    signal: np.ndarray,
    fs: int,
    model: ECGNet,
    device: torch.device | None = None,
) -> list[BeatPrediction]:
    """Run the full inference pipeline on a raw single-channel ECG.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_samples,) or (n_samples, 1). Raw, un-preprocessed.
    fs : int
        Sampling frequency in Hz. Trained on 360 Hz; other rates will
        produce poor results (a real system would resample first).
    model : ECGNet
        A trained model in eval mode.
    """
    device = device or torch.device("cpu")
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.shape[1] != 1:
        raise ValueError(f"Expected single-channel signal, got shape {signal.shape}")

    # 1. Bandpass filter (same as training).
    filtered = bandpass_filter(signal, fs=fs)

    # 2. Detect R-peaks (training used ground-truth peaks; inference cannot).
    r_peaks = detect_r_peaks(filtered[:, 0], fs=fs)
    if len(r_peaks) == 0:
        return []

    # 3. Segment around peaks (same as training).
    windows, valid_idx = segment_beats(filtered, r_peaks)
    if len(windows) == 0:
        return []

    # 4. Per-window z-score (same as training).
    windows = normalize_window(windows)

    # 5. Batch inference.
    x = torch.from_numpy(windows).unsqueeze(1).to(device)  # (n, 1, 650)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)

    valid_peaks = r_peaks[valid_idx]
    return [
        BeatPrediction(
            beat_index=int(peak),
            predicted_class=CLASS_NAMES[int(pred)],
            confidence=float(conf),
        )
        for peak, pred, conf in zip(valid_peaks, preds.cpu(), confs.cpu())
    ]