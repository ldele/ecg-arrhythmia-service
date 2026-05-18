"""Build small pre-rendered signal slices for the Streamlit demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.load import load_record

# Records chosen for variety:
# 100 - mostly normal
# 208 - lots of ventricular ectopics (V class)
# 232 - supraventricular activity (S class)
# 233 - mixed arrhythmias
DEMO_RECORDS = ["100", "208", "232", "233"]
SECONDS_PER_SAMPLE = 10  # 10 seconds = ~10 beats at 60 bpm
CHANNEL = 0


def main() -> None:
    src_dir = Path("data/raw/mitdb")
    out_dir = Path("app/demo_samples")
    out_dir.mkdir(parents=True, exist_ok=True)

    for record_id in DEMO_RECORDS:
        record = load_record(record_id, src_dir)
        n = SECONDS_PER_SAMPLE * record.fs
        slice_ = record.signal[:n, CHANNEL].astype(np.float32)
        out = out_dir / f"record_{record_id}.npy"
        np.save(out, slice_)
        print(f"{record_id}: {slice_.shape} samples @ {record.fs} Hz -> {out}")


if __name__ == "__main__":
    main()