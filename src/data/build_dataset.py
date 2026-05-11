"""Build the full (X, y) dataset from MIT-BIH records.

Uses the DS1/DS2 inter-patient split (de Chazal et al. 2004).
Paced records are excluded per standard practice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data.labels import AAMI_CLASSES, map_to_aami
from src.data.load import load_record
from src.data.preprocess import bandpass_filter, normalize_window
from src.data.segment import segment_beats

# Records excluded from analysis (paced beats dominate).
PACED_RECORDS = {"102", "104", "107", "217"}

# DS1: training records (de Chazal et al. 2004).
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]

# DS2: held-out test records (different patients from DS1).
DS2_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

# Class to integer index for model output.
CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(AAMI_CLASSES)}


@dataclass
class Dataset:
    X: np.ndarray  # (n_beats, window_length), float32
    y: np.ndarray  # (n_beats,), int64
    record_ids: np.ndarray  # (n_beats,), which record each beat came from


def build_dataset(record_ids: list[str], data_dir: Path) -> Dataset:
    """Build a Dataset from a list of MIT-BIH records.

    Pipeline per record: load -> bandpass filter -> segment around R-peaks
    -> per-window normalize -> map labels to AAMI integer classes.
    Beats with non-AAMI symbols (e.g. '+', '~') are dropped.
    """
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    rec_parts: list[np.ndarray] = []

    for record_id in record_ids:
        record = load_record(record_id, data_dir)

        # Filter raw 2-channel signal, then segment channel 0.
        filtered = bandpass_filter(record.signal, fs=record.fs)
        windows, valid_idx = segment_beats(filtered, record.beat_samples)
        windows = normalize_window(windows)

        # Map symbols of *valid* beats only.
        valid_symbols = [record.beat_symbols[i] for i in valid_idx]
        aami_labels = [map_to_aami(s) for s in valid_symbols]

        # Drop beats with no AAMI mapping.
        keep_mask = np.array([lbl is not None for lbl in aami_labels])
        if not keep_mask.any():
            continue

        kept_windows = windows[keep_mask]
        kept_labels = np.array(
            [CLASS_TO_IDX[lbl] for lbl, keep in zip(aami_labels, keep_mask) if keep],
            dtype=np.int64,
        )

        X_parts.append(kept_windows)
        y_parts.append(kept_labels)
        rec_parts.append(np.full(len(kept_labels), record_id, dtype=object))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    record_ids_arr = np.concatenate(rec_parts, axis=0)
    return Dataset(X=X, y=y, record_ids=record_ids_arr)