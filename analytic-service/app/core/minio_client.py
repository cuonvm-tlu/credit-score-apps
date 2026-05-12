from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

MINIO_ENDPOINT = "http://127.0.0.1:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "password"
MINIO_REGION = "us-east-1"


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


def download_parquet_from_minio(client: Any, bucket: str, key: str, local_path: str) -> None:
    """Download a Parquet file from MinIO to a local path."""
    client.download_file(Bucket=bucket, Key=key, Filename=local_path)


def upload_model_to_minio(client: Any, local_path: str, bucket: str, key: str) -> None:
    """Upload a model file to MinIO."""
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=f,
            ContentType="application/octet-stream",
        )