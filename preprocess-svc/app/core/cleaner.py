import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def clean_and_upload(
    client: Any,
    source_bucket: str,
    source_key: str,
    clean_bucket: str,
    version_folder: str,
    original_filename: str,
) -> str:
    """Download a raw file from MinIO, clean it, save Parquet locally, and upload back to MinIO."""
    raw_file_path = _download_object_to_temp(client, source_bucket, source_key, original_filename)
    df = _clean_dataframe(raw_file_path)

    clean_filename = _build_clean_filename(original_filename)
    local_parquet_path = Path(tempfile.gettempdir()) / clean_filename
    df.to_parquet(local_parquet_path, index=False)

    with local_parquet_path.open("rb") as parquet_file:
        client.put_object(
            Bucket=clean_bucket,
            Key=f"{version_folder}/{clean_filename}",
            Body=parquet_file,
            ContentLength=os.path.getsize(local_parquet_path),
            ContentType="application/octet-stream",
        )

    return f"{clean_bucket}/{version_folder}/{clean_filename}"


def _download_object_to_temp(client: Any, bucket_name: str, object_key: str, original_filename: str) -> str:
    suffix = Path(original_filename).suffix or ".data"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        client.download_fileobj(Bucket=bucket_name, Key=object_key, Fileobj=temp_file)
    finally:
        temp_file.close()
    return temp_file.name


def _clean_dataframe(raw_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        raw_file_path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True,
    )

    df.dropna(inplace=True)
    df.drop(columns=["fnlwgt"], inplace=True)

    string_columns = df.select_dtypes(include="object").columns
    for column in string_columns:
        df[column] = df[column].astype(str).str.strip()

    df["income"] = df["income"].str.rstrip(".").str.strip()
    df["income"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df = df[df["income"].notna()]

    return df


def _build_clean_filename(original_filename: str) -> str:
    base_name = Path(original_filename).stem.replace(".", "_")
    return f"{base_name}_clean.parquet"
