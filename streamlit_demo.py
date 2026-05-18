"""Streamlit demo client for the ECG Arrhythmia inference API."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st

DEFAULT_API_URL = "https://ecg-api-lvij6dnkaa-ew.a.run.app"
SAMPLES_DIR = Path(__file__).parent / "demo_samples"
FS = 360  # all MIT-BIH samples were recorded at 360 Hz

CLASS_COLORS = {"N": "#2ca02c", "S": "#ff7f0e", "V": "#d62728", "F": "#9467bd"}
CLASS_LABELS = {
    "N": "Normal",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion",
}


def list_samples() -> dict[str, Path]:
    if not SAMPLES_DIR.exists():
        return {}
    return {p.stem.replace("record_", "Record "): p for p in sorted(SAMPLES_DIR.glob("*.npy"))}


def call_predict(api_url: str, signal: np.ndarray, fs: int) -> dict:
    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        json={"signal": signal.tolist(), "fs": fs},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def plot_signal_with_beats(signal: np.ndarray, fs: int, predictions: list[dict]) -> plt.Figure:
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, signal, color="#333", linewidth=0.8)
    for pred in predictions:
        idx = pred["beat_index"]
        if idx >= len(signal):
            continue
        color = CLASS_COLORS.get(pred["predicted_class"], "#888")
        ax.axvline(idx / fs, color=color, alpha=0.4, linewidth=1)
        ax.annotate(
            pred["predicted_class"],
            xy=(idx / fs, signal[idx]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=color,
            weight="bold",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(alpha=0.2)
    return fig


def main() -> None:
    st.set_page_config(page_title="ECG Arrhythmia Demo", layout="wide")
    st.title("ECG Arrhythmia Classification — Live Demo")
    st.caption(
        "Demo client for the inference API. Pick a MIT-BIH sample, send it to the "
        "live Cloud Run service, and see per-beat classifications overlaid on the signal."
    )

    with st.expander("📖 About ECG signals and MIT-BIH"):
        background_path = Path(__file__).parent / "ECG_BACKGROUND.md"
        if background_path.exists():
            st.markdown(background_path.read_text(encoding="utf-8"))
        else:
            st.info("Background documentation not available in this deployment.")
    
    api_url = os.environ.get("API_URL", DEFAULT_API_URL)

    with st.sidebar:
        st.header("Configuration")
        st.text_input("API URL", value=api_url, disabled=True, help="Set via API_URL env var.")
        st.markdown("---")
        st.markdown("**Class legend**")
        for cls, label in CLASS_LABELS.items():
            st.markdown(
                f"<span style='color:{CLASS_COLORS[cls]};font-weight:bold'>{cls}</span> "
                f"— {label}",
                unsafe_allow_html=True,
            )

    samples = list_samples()
    if not samples:
        st.error("No demo samples found. Run `python -m scripts.build_demo_samples` first.")
        return

    sample_name = st.selectbox("Choose a sample", list(samples.keys()))
    signal = np.load(samples[sample_name])
    st.caption(f"Signal: {len(signal)} samples · {len(signal) / FS:.1f} seconds · {FS} Hz")

    if st.button("Classify beats", type="primary"):
        with st.spinner("Calling inference API (first request may take 10–20 s if cold)..."):
            try:
                result = call_predict(api_url, signal, FS)
            except requests.RequestException as exc:
                st.error(f"API call failed: {exc}")
                return

        st.success(f"Classified {result['n_beats']} beats")
        fig = plot_signal_with_beats(signal, FS, result["predictions"])
        st.pyplot(fig)

        # Distribution
        counts: dict[str, int] = {}
        for p in result["predictions"]:
            counts[p["predicted_class"]] = counts.get(p["predicted_class"], 0) + 1
        cols = st.columns(len(CLASS_LABELS))
        for col, cls in zip(cols, CLASS_LABELS):
            col.metric(f"{cls} ({CLASS_LABELS[cls]})", counts.get(cls, 0))

        with st.expander("Raw predictions"):
            st.dataframe(result["predictions"])


if __name__ == "__main__":
    main()