import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame as SparkDataFrame

from app.core.basic_mondrian_adapter import OUTPUT_COLUMNS, anonymize_adult_spark_dataframe
from app.core.spark_session import get_spark_session

_REQUIRED_ADULT_COLUMNS = {
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


def anonymize_cleaned_adult_k_anonymity_and_upload(
    client: Any,
    clean_bucket: str,
    clean_object_key: str,
    anonymize_bucket: str = "anonymize-zone",
    k: int = 10,
) -> str:
    """
    Download one cleaned parquet from MinIO, apply Basic_Mondrian using Spark for I/O and
    preprocessing, then upload anonymized parquet. Returns MinIO path "<bucket>/<key>".
    """
    local_clean_path = _download_object_to_temp(client, clean_bucket, clean_object_key, suffix=".parquet")

    spark = get_spark_session("preprocess-svc-anonymize", "spark-warehouse-preprocess-svc")
    parquet_uri = str(Path(local_clean_path).resolve()).replace("\\", "/")
    clean_df = spark.read.parquet(parquet_uri)

    if not _is_adult_dataframe(clean_df):
        raise ValueError("Input parquet is not Adult cleaned schema.")

    records, _, _ = anonymize_adult_spark_dataframe(clean_df, k=k)

    anon_object_key = _build_anonymized_key(clean_object_key, k=k)
    local_anon_path = Path(tempfile.gettempdir()) / Path(anon_object_key).name
    _write_records_as_parquet(records, OUTPUT_COLUMNS, local_anon_path)

    with local_anon_path.open("rb") as parquet_file:
        client.put_object(
            Bucket=anonymize_bucket,
            Key=anon_object_key,
            Body=parquet_file,
            ContentLength=os.path.getsize(local_anon_path),
            ContentType="application/octet-stream",
        )

    return f"{anonymize_bucket}/{anon_object_key}"


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


def _is_adult_dataframe(df: SparkDataFrame) -> bool:
    return _REQUIRED_ADULT_COLUMNS.issubset(set(df.columns))


def _write_records_as_parquet(
    records: list[list[str]],
    columns: list[str],
    local_parquet_path: Path,
) -> None:
    """
    Write Mondrian result records (list of string rows) to a local parquet file via pyarrow.

    We deliberately do not route this through Spark: the records already live on the driver
    after Mondrian, and ``spark.createDataFrame(python_list).write.parquet`` is unreliable on
    Windows-local Spark (HADOOP_HOME/winutils + Python worker socket issues).
    """
    columns_data: dict[str, list[str]] = {col: [] for col in columns}
    for record in records:
        for i, col in enumerate(columns):
            columns_data[col].append(record[i])
    table = pa.table(columns_data)
    pq.write_table(table, str(local_parquet_path))
