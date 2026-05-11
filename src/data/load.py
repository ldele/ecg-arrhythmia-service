"""MIT-BIH Arrhythmia Database loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


@dataclass
class EcgRecord:
    """A single MIT-BIH record: signal + beat annotations."""

    record_id: str
    signal: np.ndarray  # shape (n_samples, n_channels), float64
    fs: int  # sampling frequency, Hz
    beat_samples: np.ndarray  # sample indices of annotated beats, shape (n_beats,)
    beat_symbols: list[str]  # AAMI-mappable symbols, length n_beats


def load_record(record_id: str, data_dir: Path) -> EcgRecord:
    """Load one MIT-BIH record and its beat annotations.

    Parameters
    ----------
    record_id : str
        MIT-BIH record name, e.g. "100".
    data_dir : Path
        Directory containing the .dat / .hea / .atr files.

    Returns
    -------
    EcgRecord
        Signal array, sampling frequency, and beat annotations.
    """
    record_path = str(data_dir / record_id)

    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, extension="atr")

    return EcgRecord(
        record_id=record_id,
        signal=np.asarray(record.p_signal, dtype=np.float64),
        fs=int(record.fs),
        beat_samples=np.asarray(annotation.sample, dtype=np.int64),
        beat_symbols=list(annotation.symbol),
    )


def download_mitbih(data_dir: Path) -> None:
    """Download the MIT-BIH Arrhythmia Database to data_dir."""
    data_dir.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(data_dir))