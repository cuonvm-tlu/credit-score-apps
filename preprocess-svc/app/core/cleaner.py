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


def _detect_has_header(raw_file_path: str) -> bool:
    """Check if the CSV file starts with a text header row (not a data row)."""
    try:
        with open(raw_file_path, "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline().strip()
        # If the first field can be parsed as an integer, it's likely a data row (age)
        first_field = first_line.split(",")[0].strip()
        int(first_field)
        return False
    except ValueError:
        return True


def _clean_dataframe(raw_file_path: str) -> pd.DataFrame:
    has_header = _detect_has_header(raw_file_path)
    df = pd.read_csv(
        raw_file_path,
        header=0 if has_header else None,
        names=None if has_header else COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True,
    )
    # Standardise column names when header is present (strip spaces, lower-case)
    if has_header:
        df.columns = [c.strip().lower() for c in df.columns]
    # Re-order / select only the expected columns (drop extras like fnlwgt if present)
    available = [c for c in COLUMN_NAMES if c in df.columns]
    df = df[available]

    df.dropna(inplace=True)
    if "fnlwgt" in df.columns:
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
