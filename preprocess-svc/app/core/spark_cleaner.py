"""
spark_cleaner.py — Thay thế _clean_dataframe() trong cleaner.py
Dùng Spark thay vì Pandas, thêm generalization (age_group, continent_code).
"""
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pycountry
import pycountry_convert as pc
from rapidfuzz import process as fuzz_process

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from app.core.spark_session import get_spark_session

# ── Country → continent mapping (build once on import) ─────────────────────

_SPECIAL_CASES = {
    "England": "United Kingdom", "Scotland": "United Kingdom",
    "Wales": "United Kingdom", "Yugoslavia": "Serbia",
    "Outlying US": "United States", "South": "Other",
}
_COUNTRY_NAMES = [c.name for c in pycountry.countries]


def _normalize(name: str) -> str:
    name = name.strip().replace("-", " ").replace("&", "and")
    name = re.sub(r"\(.*\)", "", name)
    return re.sub(r"\s+", " ", name).strip()


def _to_continent(name: str | None) -> str:
    if not name:
        return "Other"
    name = _normalize(name)
    name = _SPECIAL_CASES.get(name, name)
    if name == "Other":
        return "Other"
    try:
        alpha2 = pycountry.countries.lookup(name).alpha_2
        return pc.country_alpha2_to_continent_code(alpha2)
    except Exception:
        pass
    match, score, _ = fuzz_process.extractOne(name, _COUNTRY_NAMES)
    if score >= 80:
        try:
            alpha2 = pycountry.countries.lookup(match).alpha_2
            return pc.country_alpha2_to_continent_code(alpha2)
        except Exception:
            return "Other"
    return "Other"


# ── Main function (drop-in cho clean_and_upload) ────────────────────────────

def spark_clean_and_upload(
    client: Any,
    source_bucket: str,
    source_key: str,
    clean_bucket: str,
    version_folder: str,
    original_filename: str,
) -> str:
    """
    Giống clean_and_upload() trong cleaner.py nhưng dùng Spark.
    Thêm 2 cột: age_group, continent_code (generalization cho K-anonymity).
    """
    # Download raw từ MinIO về temp
    suffix = Path(original_filename).suffix or ".data"
    temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        client.download_fileobj(
            Bucket=source_bucket, Key=source_key, Fileobj=temp_raw
        )
    finally:
        temp_raw.close()

    # Spark clean + generalize
    df_spark = _spark_clean(temp_raw.name)

    # Ghi parquet tạm
    clean_filename = Path(original_filename).stem.replace(".", "_") + "_clean.parquet"
    local_parquet = Path(tempfile.gettempdir()) / clean_filename

    # Spark write → folder, lấy file duy nhất ra
    tmp_spark_out = str(local_parquet) + "_spark_out"
    df_spark.coalesce(1).write.mode("overwrite").parquet(tmp_spark_out)
    part_file = next(
        Path(tmp_spark_out).glob("part-*.parquet"), None
    )
    if part_file is None:
        raise RuntimeError("Spark write produced no parquet file")
    part_file.rename(local_parquet)

    # Upload lên MinIO clean-zone
    with local_parquet.open("rb") as f:
        client.put_object(
            Bucket=clean_bucket,
            Key=f"{version_folder}/{clean_filename}",
            Body=f,
            ContentLength=os.path.getsize(local_parquet),
            ContentType="application/octet-stream",
        )

    return f"{clean_bucket}/{version_folder}/{clean_filename}"


def _spark_clean(raw_file_path: str):
    """
    Core cleaning logic bằng Spark — tương đương _clean_dataframe() Pandas
    nhưng thêm age_group và continent_code.
    """
    spark = get_spark_session(
        "preprocess-svc-spark-cleaner",
        "spark-warehouse-cleaner"
    )

    COLUMN_NAMES = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
    ]

    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
    schema = StructType([
        StructField("age",            IntegerType(), True),
        StructField("workclass",      StringType(),  True),
        StructField("fnlwgt",         IntegerType(), True),
        StructField("education",      StringType(),  True),
        StructField("education-num",  IntegerType(), True),
        StructField("marital-status", StringType(),  True),
        StructField("occupation",     StringType(),  True),
        StructField("relationship",   StringType(),  True),
        StructField("race",           StringType(),  True),
        StructField("sex",            StringType(),  True),
        StructField("capital-gain",   IntegerType(), True),
        StructField("capital-loss",   IntegerType(), True),
        StructField("hours-per-week", IntegerType(), True),
        StructField("native-country", StringType(),  True),
        StructField("income",         StringType(),  True),
    ])

    df = (
        spark.read
        .schema(schema)
        .option("header", "false")
        .option("mode", "PERMISSIVE")
        .option("nanValue", "?")
        .csv(raw_file_path)
    )

    # S1: trim whitespace tất cả string cols
    str_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, StringType)]
    for col in str_cols:
        df = df.withColumn(col, F.trim(F.col(f"`{col}`")))

    # S2: "?" → null
    for col in ["workclass", "occupation", "native-country"]:
        df = df.withColumn(col, F.when(F.col(f"`{col}`") == "?", None)
                                 .otherwise(F.col(f"`{col}`")))

    # S3: drop fnlwgt (như team đã làm)
    df = df.drop("fnlwgt")

    # S4: normalize income (strip trailing dot, map 0/1)
    df = df.withColumn("income",
        F.regexp_replace(F.trim(F.col("income")), r"\.$", ""))
    df = df.withColumn("income",
        F.when(F.col("income") == "<=50K", F.lit(0))
         .when(F.col("income") == ">50K",  F.lit(1))
         .otherwise(None).cast(IntegerType()))

    # S5: drop null ở QI + SA
    critical = ["age", "sex", "marital-status", "occupation",
                "native-country", "income"]
    df = df.dropna(subset=critical)

    # S6: age_group generalization (native Spark, không cần UDF)
    df = df.withColumn("age_group",
        F.concat(
            (F.floor(F.col("age") / 10) * 10).cast("int").cast("string"),
            F.lit("-"),
            ((F.floor(F.col("age") / 10) * 10) + 10).cast("int").cast("string")
        )
    )

    # S7: continent_code generalization (build map trên driver, apply bằng create_map)
    countries = [
        r["native-country"]
        for r in df.select("`native-country`").distinct().collect()
        if r["native-country"] is not None
    ]
    mapping = {c: _to_continent(c) for c in countries}

    map_pairs = []
    for k, v in mapping.items():
        map_pairs.extend([F.lit(k), F.lit(v)])
    spark_map = F.create_map(*map_pairs)

    df = df.withColumn("continent_code",
        F.coalesce(spark_map[F.col("`native-country`")], F.lit("Other")))

    return df