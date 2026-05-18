"""GCS helpers for loading model artifacts."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

from google.cloud import storage

logger = logging.getLogger(__name__)

_GS_URI_RE = re.compile(r"^gs://([^/]+)/(.+)$")


def is_gcs_uri(path: str) -> bool:
    return path.startswith("gs://")


def download_gcs_blob(uri: str) -> Path:
    """Download a GCS object to a local temp file and return the path."""
    match = _GS_URI_RE.match(uri)
    if not match:
        raise ValueError(f"Invalid GCS URI: {uri!r}")
    bucket_name, blob_name = match.group(1), match.group(2)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    tmp = NamedTemporaryFile(delete=False, suffix=Path(blob_name).suffix)
    tmp.close()
    logger.info(
        "Downloading model from GCS",
        extra={"bucket": bucket_name, "blob": blob_name, "local_path": tmp.name},
    )
    blob.download_to_filename(tmp.name)
    return Path(tmp.name)