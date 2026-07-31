"""GCS helpers for loading model artifacts."""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from tempfile import mkstemp

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

    # Named file the GCS client can write to; the caller owns it afterwards.
    fd, local_path = mkstemp(suffix=Path(blob_name).suffix)
    os.close(fd)
    logger.info(
        "Downloading model from GCS",
        extra={"bucket": bucket_name, "blob": blob_name, "local_path": local_path},
    )
    blob.download_to_filename(local_path)
    return Path(local_path)
