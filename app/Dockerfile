# syntax=docker/dockerfile:1.6

# ---- builder: install deps, then drop what a runtime never needs ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends binutils \
 && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .

# Bytecode is deliberately kept (pip compiles by default). The runtime sets
# PYTHONDONTWRITEBYTECODE, so without .pyc every cold start would recompile,
# and Cloud Run cold-starts from zero on every idle period.
RUN pip install --prefix=/install -r requirements.txt

# pyarrow stays: streamlit imports it lazily inside st.dataframe, which the
# demo uses for the predictions table. Removing it breaks that path at runtime.
RUN set -eux; \
    SP=/install/lib/python3.11/site-packages; \
    find "$SP" -type d -regex '.*/tests?$' -prune -exec rm -rf {} +; \
    rm -rf "$SP/pyarrow/include"; \
    find "$SP" -name '*.a' -delete; \
    find "$SP" -name '*.pyi' -delete; \
    find "$SP" -name '*.so' -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find "$SP" -name '*.so.*' -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    rm -rf "$SP/pip" "$SP/setuptools" "$SP/wheel" "$SP/pkg_resources"

# ---- runtime ----
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    API_URL=https://ecg-api-lvij6dnkaa-ew.a.run.app

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /install /usr/local

WORKDIR /app

COPY app/streamlit_demo.py .
COPY app/ECG_BACKGROUND.md .
COPY app/demo_samples/ ./demo_samples/

USER appuser
EXPOSE 8080

CMD ["sh", "-c", "streamlit run streamlit_demo.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
