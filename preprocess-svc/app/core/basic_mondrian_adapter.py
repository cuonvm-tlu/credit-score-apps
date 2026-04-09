from typing import Any

import pandas as pd

from app.core.basic_mondrian.mondrian import mondrian
from app.core.anonymization_shared.gentree import GenTree
from app.core.anonymization_shared.numrange import NumRange

# Keep exactly the Adult QI order used by Basic_Mondrian/utils/read_adult_data.py
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

# QI indices that are treated as categorical in the original implementation.
# (age, education_num are numeric)
IS_CAT = [False, True, False, True, True, True, True, True]


def _normalize_columns_for_mondrian(df: pd.DataFrame) -> pd.DataFrame:
    """Map cleaned Adult columns to names expected by Basic_Mondrian code."""
    rename_map = {
        "education-num": "education_num",
        "marital-status": "marital_status",
        "native-country": "native_country",
    }
    normalized = df.rename(columns=rename_map).copy()
    missing = [c for c in (QI_COLUMNS + [SA_COLUMN]) if c not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns for Basic_Mondrian Adult flow: {missing}")
    return normalized


def _build_flat_categorical_tree(values: list[str]) -> dict[str, GenTree]:
    """
    Build a two-level categorical GH from data itself:
    '*' -> each unique value.
    This removes dependency on static tree files.
    """
    att_tree: dict[str, GenTree] = {"*": GenTree("*")}
    root = att_tree["*"]
    for value in sorted(set(values)):
        if value not in att_tree:
            att_tree[value] = GenTree(value, root, isleaf=True)
    return att_tree


def _build_numeric_numrange(values: list[str]) -> NumRange:
    support: dict[str, int] = {}
    for value in values:
        support[value] = support.get(value, 0) + 1
    sorted_values = sorted(support.keys(), key=lambda x: int(x))
    return NumRange(sorted_values, support)


def _build_adult_att_trees(df: pd.DataFrame) -> list[Any]:
    """Build att_trees from cleaned dataframe (no external data folder)."""
    trees: list[Any] = []
    for i, col in enumerate(QI_COLUMNS):
        col_values = df[col].astype(str).tolist()
        if IS_CAT[i]:
            trees.append(_build_flat_categorical_tree(col_values))
        else:
            trees.append(_build_numeric_numrange(col_values))
    return trees


def anonymize_adult_dataframe(df: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, float, float]:
    """
    Apply Basic_Mondrian to Adult cleaned dataframe and return:
    (anonymized_df, ncp, runtime_seconds).
    """
    normalized = _normalize_columns_for_mondrian(df)
    normalized = normalized.dropna(subset=QI_COLUMNS + [SA_COLUMN]).copy()

    # Build input records in the exact expected shape: QIs + SA as last column.
    records: list[list[str]] = []
    for _, row in normalized.iterrows():
        qis = [str(row[c]) for c in QI_COLUMNS]
        sa = str(row[SA_COLUMN])
        records.append(qis + [sa])

    att_trees = _build_adult_att_trees(normalized)
    result, (ncp, rtime) = mondrian(att_trees, records, k)

    out_cols = QI_COLUMNS + [SA_COLUMN]
    anonymized_df = pd.DataFrame(result, columns=out_cols)
    return anonymized_df, float(ncp), float(rtime)

