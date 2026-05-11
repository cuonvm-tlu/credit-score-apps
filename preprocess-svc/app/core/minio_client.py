import os
import time
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_INIT_RETRIES = int(os.getenv("MINIO_INIT_RETRIES", "20"))
MINIO_INIT_RETRY_SECONDS = float(os.getenv("MINIO_INIT_RETRY_SECONDS", "1"))
BUCKET_NAMES = ["landing-zone", "clean-zone", "anonymize-zone"]


def get_minio_client() -> Any:
    """Return a configured boto3 S3 client for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name=MINIO_REGION,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket(client: Any, bucket_name: str) -> None:
    """Ensure the specified bucket exists in MinIO, creating it if needed."""
    try:
        client.head_bucket(Bucket=bucket_name)
    except ClientError:
        client.create_bucket(Bucket=bucket_name)


def init_minio() -> None:
    """Create required buckets on application startup."""
    last_error: Exception | None = None
    for _ in range(MINIO_INIT_RETRIES):
        try:
            client = get_minio_client()
            for bucket_name in BUCKET_NAMES:
                ensure_bucket(client, bucket_name)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(MINIO_INIT_RETRY_SECONDS)
    if last_error is not None:
        raise last_error
