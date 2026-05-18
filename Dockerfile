# syntax=docker/dockerfile:1.6

# ---------- Build stage ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Install CPU-only PyTorch + other deps into a dedicated prefix.
COPY requirements.txt .
RUN pip install --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt


# ---------- Runtime stage ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    MODEL_PATH=/app/models/model.pt

# Non-root user for security.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copy installed packages from the builder.
COPY --from=builder /install /usr/local

# Copy only what the runtime needs.
COPY src/ ./src/
COPY models/model.pt ./models/model.pt

USER appuser
EXPOSE 8000

# Use sh -c so $PORT expands (Cloud Run injects $PORT at runtime).
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]