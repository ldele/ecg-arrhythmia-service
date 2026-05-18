"""Request/response schemas for the inference API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One single-channel ECG signal to classify."""

    signal: list[float] = Field(
        ...,
        description="Raw single-channel ECG samples.",
        min_length=650,  # at least one window
    )
    fs: int = Field(
        default=360,
        description="Sampling frequency in Hz. Model was trained on 360 Hz.",
        ge=100,
        le=2000,
    )


class BeatPredictionOut(BaseModel):
    beat_index: int = Field(..., description="Sample index of the R-peak.")
    predicted_class: str = Field(..., description="One of N, S, V, F.")
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    n_beats: int
    predictions: list[BeatPredictionOut]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool