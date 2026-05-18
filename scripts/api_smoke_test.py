"""Hit the running local API with a real MIT-BIH signal."""
from __future__ import annotations

from pathlib import Path

import requests

from src.data.load import load_record


def main() -> None:
    record = load_record("100", Path("data/raw/mitdb"))
    # First 10 seconds of channel 0 = 3600 samples at 360 Hz, ~10 beats.
    signal_slice = record.signal[: 10 * record.fs, 0].tolist()

    response = requests.post(
        "http://localhost:8000/predict",
        json={"signal": signal_slice, "fs": record.fs},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    print(f"n_beats: {data['n_beats']}")
    print(f"first 3: {data['predictions'][:3]}")


if __name__ == "__main__":
    main()