from typing import Any

import pandas as pd

from app.core.anonymization_shared.gentree import GenTree
from app.core.anonymization_shared.numrange import NumRange
from app.core.mondrian_l_diversity.mondrian_l_diversity import mondrian_l_diversity

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
IS_CAT = [False, True, False, True, True, True, True, True]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "education-num": "education_num",
        "marital-status": "marital_status",
        "native-country": "native_country",
    }
    normalized = df.rename(columns=rename_map).copy()
    missing = [c for c in (QI_COLUMNS + [SA_COLUMN]) if c not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns for L-diversity Adult flow: {missing}")
    return normalized


def _build_flat_categorical_tree(values: list[str]) -> dict[str, GenTree]:
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


def _build_att_trees(df: pd.DataFrame) -> list[Any]:
    trees: list[Any] = []
    for i, col in enumerate(QI_COLUMNS):
        vals = df[col].astype(str).tolist()
        if IS_CAT[i]:
            trees.append(_build_flat_categorical_tree(vals))
        else:
            trees.append(_build_numeric_numrange(vals))
    return trees


def anonymize_adult_dataframe_l_diversity(
    df: pd.DataFrame,
    l_value: int = 2,
) -> tuple[pd.DataFrame, float, float]:
    normalized = _normalize_columns(df)
    normalized = normalized.dropna(subset=QI_COLUMNS + [SA_COLUMN]).copy()
    records = []
    for _, row in normalized.iterrows():
        qis = [str(row[c]) for c in QI_COLUMNS]
        sa = str(row[SA_COLUMN])
        records.append(qis + [sa])

    att_trees = _build_att_trees(normalized)
    result, (ncp, rtime) = mondrian_l_diversity(att_trees, records, l_value)
    out_cols = QI_COLUMNS + [SA_COLUMN]
    out_df = pd.DataFrame(result, columns=out_cols)
    return out_df, float(ncp), float(rtime)

