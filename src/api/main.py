"""FastAPI inference service for ECG arrhythmia classification."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from src.utils.gcs import download_gcs_blob, is_gcs_uri
from src.api.logging_config import configure_logging
from src.api.schemas import (
    BeatPredictionOut,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)
from src.model.predict import load_model, predict_signal

logger = logging.getLogger("ecg.api")

# Holds the model after startup; loaded once, reused per request.
_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup. Supports gs:// URIs and local paths."""
    configure_logging()
    model_uri = os.environ.get("MODEL_PATH", "models/model.pt")
    logger.info("Loading model", extra={"model_uri": model_uri})

    if is_gcs_uri(model_uri):
        local_path = download_gcs_blob(model_uri)
    else:
        local_path = Path(model_uri)

    device = torch.device("cpu")
    model = load_model(local_path, device=device)
    _state["model"] = model
    _state["device"] = device
    logger.info("Model loaded")
    yield
    _state.clear()


app = FastAPI(
    title="ECG Arrhythmia Inference",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="model" in _state)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    signal = np.asarray(request.signal, dtype=np.float64)
    try:
        preds = predict_signal(
            signal,
            fs=request.fs,
            model=_state["model"],  # type: ignore[arg-type]
            device=_state["device"],  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Prediction completed",
        extra={
            "n_samples": int(signal.shape[0]),
            "fs": request.fs,
            "n_beats": len(preds),
            "elapsed_ms": round(elapsed_ms, 2),
        },
    )

    return PredictResponse(
        n_beats=len(preds),
        predictions=[
            BeatPredictionOut(
                beat_index=p.beat_index,
                predicted_class=p.predicted_class,
                confidence=p.confidence,
            )
            for p in preds
        ],
    )