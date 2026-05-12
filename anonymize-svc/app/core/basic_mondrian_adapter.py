from typing import Any

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F

from app.core.anonymization_shared.gentree import GenTree
from app.core.anonymization_shared.numrange import NumRange
from app.core.basic_mondrian.mondrian import mondrian

QI_COLUMNS = [
    "age",
    "workclass",
    "education_num",
    "marital_status",
    "occupation",
    "race",
    "sex",
    "native_country",
]
SA_COLUMN = "income"
OUTPUT_COLUMNS = QI_COLUMNS + [SA_COLUMN]

# QI indices treated as categorical in the original implementation.
# (age, education_num are numeric)
IS_CAT = [False, True, False, True, True, True, True, True]
NUMERIC_QI_COLUMNS = [col for col, is_cat in zip(QI_COLUMNS, IS_CAT) if not is_cat]
CATEGORICAL_QI_COLUMNS = [col for col, is_cat in zip(QI_COLUMNS, IS_CAT) if is_cat]

_RENAME_MAP = {
    "education-num": "education_num",
    "marital-status": "marital_status",
    "native-country": "native_country",
}


def _normalize_spark_dataframe(df: SparkDataFrame) -> SparkDataFrame:
    """Rename cleaned Adult columns to Mondrian names, cast numerics, drop nulls on QI+SA."""
    normalized = df
    for src, dst in _RENAME_MAP.items():
        if src in normalized.columns and dst not in normalized.columns:
            normalized = normalized.withColumnRenamed(src, dst)

    missing = [c for c in (QI_COLUMNS + [SA_COLUMN]) if c not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns for Basic_Mondrian Adult flow: {missing}")

    # Mondrian expects integer-compatible string values; Spark/Parquet may carry them as long/double.
    for col in NUMERIC_QI_COLUMNS:
        normalized = normalized.withColumn(col, F.col(col).cast("long").cast("string"))

    for col in CATEGORICAL_QI_COLUMNS + [SA_COLUMN]:
        normalized = normalized.withColumn(col, F.col(col).cast("string"))

    normalized = normalized.select(*QI_COLUMNS, SA_COLUMN)
    normalized = normalized.na.drop(subset=QI_COLUMNS + [SA_COLUMN])
    return normalized


def _build_flat_categorical_tree(values: list[str]) -> dict[str, GenTree]:
    """
    Two-level categorical generalization hierarchy built from data:
    '*' -> each unique value. Removes dependency on static tree files.
    """
    att_tree: dict[str, GenTree] = {"*": GenTree("*")}
    root = att_tree["*"]
    for value in sorted(set(values)):
        if value not in att_tree:
            att_tree[value] = GenTree(value, root, isleaf=True)
    return att_tree


def _build_numeric_numrange_from_counts(value_counts: list[tuple[str, int]]) -> NumRange:
    support: dict[str, int] = {value: count for value, count in value_counts}
    sorted_values = sorted(support.keys(), key=lambda x: int(x))
    return NumRange(sorted_values, support)


def _build_adult_att_trees(df: SparkDataFrame) -> list[Any]:
    """Build att_trees by aggregating distinct/counts via Spark, not by collecting full rows."""
    trees: list[Any] = []
    for i, col in enumerate(QI_COLUMNS):
        if IS_CAT[i]:
            rows = df.select(col).distinct().collect()
            values = [str(r[col]) for r in rows if r[col] is not None]
            trees.append(_build_flat_categorical_tree(values))
        else:
            rows = df.groupBy(col).count().collect()
            value_counts = [(str(r[col]), int(r["count"])) for r in rows if r[col] is not None]
            trees.append(_build_numeric_numrange_from_counts(value_counts))
    return trees


def _collect_records(df: SparkDataFrame) -> list[list[str]]:
    """Materialize the normalized Spark DataFrame as QI+SA string records on the driver."""
    rows = df.select(*QI_COLUMNS, SA_COLUMN).collect()
    records: list[list[str]] = []
    for row in rows:
        record = [str(row[c]) for c in QI_COLUMNS]
        record.append(str(row[SA_COLUMN]))
        records.append(record)
    return records


def anonymize_adult_spark_dataframe(
    df: SparkDataFrame,
    k: int = 10,
) -> tuple[list[list[str]], float, float]:
    """
    Apply Basic_Mondrian to an Adult cleaned Spark DataFrame.
    Returns (records, ncp, runtime_seconds) where each record is a list of strings
    aligned with OUTPUT_COLUMNS (QI columns followed by SA_COLUMN).

    Spark is used for I/O and preprocessing (rename/cast/dropna), and aggregation when
    building the generalization hierarchies. The Mondrian algorithm itself is a sequential
    recursive partitioning over collected records, so we return the raw list and let the
    caller persist it without round-tripping through ``spark.createDataFrame`` (which
    requires Python workers and is unreliable on Windows-local Spark).
    """
    normalized = _normalize_spark_dataframe(df).cache()
    try:
        if normalized.limit(1).count() == 0:
            raise ValueError("Input dataframe has no valid rows after Adult schema normalization.")

        att_trees = _build_adult_att_trees(normalized)
        records = _collect_records(normalized)
    finally:
        normalized.unpersist()

    result, (ncp, rtime) = mondrian(att_trees, records, k)
    return result, float(ncp), float(rtime)
