import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.basic_mondrian_adapter import anonymize_adult_dataframe
from app.core.spark_session import get_spark_session


def anonymize_cleaned_adult_k_anonymity_and_upload(
    client: Any,
    clean_bucket: str,
    clean_object_key: str,
    k: int = 10,
) -> str:
    """
    Download one cleaned parquet from MinIO, apply Basic_Mondrian, then upload anonymized parquet.
    Returns MinIO path in format: "<bucket>/<key>".
    """
    local_clean_path = _download_object_to_temp(client, clean_bucket, clean_object_key, suffix=".parquet")
    df = _read_parquet_with_pyspark(local_clean_path)

    if not _is_adult_dataframe(df):
        raise ValueError("Input parquet is not Adult cleaned schema.")

    anon_df, _, _ = anonymize_adult_dataframe(df, k=k)
    anon_object_key = _build_anonymized_key(clean_object_key, k=k)
    local_anon_path = Path(tempfile.gettempdir()) / Path(anon_object_key).name
    anon_df.to_parquet(local_anon_path, index=False)

    with local_anon_path.open("rb") as parquet_file:
        client.put_object(
            Bucket=clean_bucket,
            Key=anon_object_key,
            Body=parquet_file,
            ContentLength=os.path.getsize(local_anon_path),
            ContentType="application/octet-stream",
        )

    return f"{clean_bucket}/{anon_object_key}"


def _download_object_to_temp(client: Any, bucket_name: str, object_key: str, suffix: str) -> str:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        client.download_fileobj(Bucket=bucket_name, Key=object_key, Fileobj=temp_file)
    finally:
        temp_file.close()
    return temp_file.name


def _build_anonymized_key(clean_object_key: str, k: int) -> str:
    path = Path(clean_object_key)
    base = path.stem
    if base.endswith("_clean"):
        base = base[: -len("_clean")]
    anon_name = f"{base}_anon_k{k}.parquet"
    return str(path.with_name(anon_name)).replace("\\", "/")


def _read_parquet_with_pyspark(local_clean_path: str) -> pd.DataFrame:
    parquet_path = str(Path(local_clean_path).resolve()).replace("\\", "/")
    spark = get_spark_session("preprocess-svc-anonymize", "spark-warehouse-preprocess-svc")
    return spark.read.parquet(parquet_path).toPandas()


def _is_adult_dataframe(df: pd.DataFrame) -> bool:
    expected = {
        "age",
        "workclass",
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
    }
    return expected.issubset(set(df.columns))

