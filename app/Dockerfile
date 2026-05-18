# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080 \
    API_URL=https://ecg-api-lvij6dnkaa-ew.a.run.app

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY app/requirements.txt .
RUN pip install -r requirements.txt

COPY app/streamlit_demo.py .
COPY app/demo_samples/ ./demo_samples/

USER appuser
EXPOSE 8080

CMD ["sh", "-c", "streamlit run streamlit_demo.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]