"""
Adapter layer for Differential Privacy integration with API routes.

Provides functions to apply DP post-processing to anonymized data
and upload results to MinIO, similar to k-anonymity and l-diversity adapters.
"""

import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.core.dp_mechanisms import DPAnonymizationIntegration, load_dp_execution_plan
from app.core.dp_mechanisms.above_threshold import AboveThresholdMechanism
from app.core.dp_mechanisms.exponential_mechanism import ExponentialMechanism
from app.core.dp_metrics import record_dp_run_metric


def apply_dp_protection_and_upload(
    client: Any,
    clean_bucket: str,
    clean_object_key: str,
    anonymize_bucket: str = "anonymize-zone",
    epsilon: float = 0.3,
    mechanism: str = "laplace",
    delta: float = 0.0,
    profile_name: str = "default",
    numerical_sensitivity: Optional[Dict[str, float]] = None,
    epsilon_allocation: Optional[Dict[str, float]] = None,
    numeric_mechanism: Optional[str] = None,
    categorical_mechanism: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
    categorical_epsilon_allocation: Optional[Dict[str, float]] = None,
    threshold_rules: Optional[List[Dict[str, Any]]] = None,
    above_threshold_values: Optional[Dict[str, float]] = None,
    above_threshold_return_none: bool = False,
    max_epsilon_per_attribute: float = 0.1,
    output_name_template: Optional[str] = None,
) -> str:
    """
    Download cleaned parquet from MinIO, apply Differential Privacy noise,
    then upload DP-protected parquet.

    Args:
        client: MinIO client object
        clean_bucket: Bucket name (e.g., "clean-zone")
        clean_object_key: Object key of cleaned parquet
        epsilon: Privacy budget for DP noise (default 0.3)
        mechanism: DP mechanism to apply (laplace or exponential)
        delta: Delta for approximate-DP mechanisms

    Returns:
        MinIO path in format: "<bucket>/<key>" pointing to DP-protected file

    Raises:
        ValueError: If input data schema is invalid
    """
    trace_id = uuid.uuid4().hex
    run_started_at = datetime.utcnow().isoformat() + "Z"
    run_start = time.perf_counter()

    latency_download_ms: Optional[float] = None
    latency_read_parquet_ms: Optional[float] = None
    latency_dp_apply_ms: Optional[float] = None
    latency_write_parquet_ms: Optional[float] = None
    latency_upload_ms: Optional[float] = None

    row_count: Optional[int] = None
    column_count: Optional[int] = None
    dp_object_key: Optional[str] = None
    status = "failed"
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    try:
        # Step 1: Download cleaned parquet from MinIO
        download_start = time.perf_counter()
        local_clean_path = _download_object_to_temp(
            client, clean_bucket, clean_object_key, suffix=".parquet"
        )
        latency_download_ms = round((time.perf_counter() - download_start) * 1000, 3)

        # Step 2: Read with Pandas
        read_start = time.perf_counter()
        df = _read_parquet_with_pandas(local_clean_path)
        latency_read_parquet_ms = round((time.perf_counter() - read_start) * 1000, 3)
        row_count = int(len(df))
        column_count = int(len(df.columns))

        # Step 3: Validate Adult dataframe schema
        if not _is_adult_dataframe(df):
            raise ValueError("Input parquet is not Adult cleaned schema.")

        # Step 4: Apply DP protection
        dp_apply_start = time.perf_counter()
        dp_protected_df = _apply_dp_to_dataframe(
            df,
            epsilon,
            mechanism=mechanism,
            delta=delta,
            numerical_sensitivity=numerical_sensitivity,
            epsilon_allocation=epsilon_allocation,
            numeric_mechanism=numeric_mechanism,
            categorical_mechanism=categorical_mechanism,
            categorical_columns=categorical_columns,
            categorical_epsilon_allocation=categorical_epsilon_allocation,
            threshold_rules=threshold_rules,
            above_threshold_values=above_threshold_values,
            above_threshold_return_none=above_threshold_return_none,
            max_epsilon_per_attribute=max_epsilon_per_attribute,
        )
        latency_dp_apply_ms = round((time.perf_counter() - dp_apply_start) * 1000, 3)
        row_count = int(len(dp_protected_df))
        column_count = int(len(dp_protected_df.columns))

        # Step 5: Upload DP-protected parquet to MinIO
        dp_object_key = _build_dp_protected_key(
            clean_object_key,
            epsilon,
            profile_name=profile_name,
            output_name_template=output_name_template,
        )
        local_dp_path = Path(tempfile.gettempdir()) / Path(dp_object_key).name

        write_start = time.perf_counter()
        dp_protected_df.to_parquet(local_dp_path, index=False)
        latency_write_parquet_ms = round((time.perf_counter() - write_start) * 1000, 3)

        upload_start = time.perf_counter()
        with local_dp_path.open("rb") as parquet_file:
            client.put_object(
                Bucket=anonymize_bucket,
                Key=dp_object_key,
                Body=parquet_file,
                ContentLength=os.path.getsize(local_dp_path),
                ContentType="application/octet-stream",
            )
        latency_upload_ms = round((time.perf_counter() - upload_start) * 1000, 3)

        status = "success"
        return f"{anonymize_bucket}/{dp_object_key}"
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        raise
    finally:
        latency_total_ms = round((time.perf_counter() - run_start) * 1000, 3)
        record_dp_run_metric(
            {
                "event_time_utc": run_started_at,
                "trace_id": trace_id,
                "service": "anonymize-svc",
                "component": "dp_mechanism",
                "profile_name": profile_name,
                "epsilon": epsilon,
                "mechanism": mechanism,
                "delta": delta,
                "source_bucket": clean_bucket,
                "source_key": clean_object_key,
                "target_bucket": anonymize_bucket,
                "target_key": dp_object_key,
                "row_count": row_count,
                "column_count": column_count,
                "status": status,
                "error_type": error_type,
                "error_message": error_message,
                "latency_total_ms": latency_total_ms,
                "latency_download_ms": latency_download_ms,
                "latency_read_parquet_ms": latency_read_parquet_ms,
                "latency_dp_apply_ms": latency_dp_apply_ms,
                "latency_write_parquet_ms": latency_write_parquet_ms,
                "latency_upload_ms": latency_upload_ms,
            }
        )


def apply_configured_dp_protection_and_upload(
    client: Any,
    clean_bucket: str,
    clean_object_key: str,
    anonymize_bucket: str = "anonymize-zone",
    config_path: Optional[str] = None,
) -> List[str]:
    """Run DP for one profile or many profiles based on the execution config."""
    plan = load_dp_execution_plan(config_path)
    outputs: List[str] = []

    for profile in plan.get_profiles_to_run():
        try:
            output_path = apply_dp_protection_and_upload(
                client=client,
                clean_bucket=clean_bucket,
                clean_object_key=clean_object_key,
                anonymize_bucket=anonymize_bucket,
                epsilon=profile.epsilon,
                mechanism=profile.mechanism,
                delta=profile.delta,
                profile_name=profile.name,
                numerical_sensitivity=profile.numerical_sensitivity,
                epsilon_allocation=profile.epsilon_allocation,
                numeric_mechanism=profile.numeric_mechanism,
                categorical_mechanism=profile.categorical_mechanism,
                categorical_columns=profile.categorical_columns,
                categorical_epsilon_allocation=profile.categorical_epsilon_allocation,
                threshold_rules=profile.threshold_rules,
                above_threshold_values=profile.above_threshold_values,
                above_threshold_return_none=profile.above_threshold_return_none,
                max_epsilon_per_attribute=profile.max_epsilon_per_attribute,
                output_name_template=plan.output_name_template,
            )
            outputs.append(output_path)
        except Exception:
            if plan.stop_on_error:
                raise

    return outputs


def _apply_dp_to_dataframe(
    df: pd.DataFrame,
    epsilon: float,
    mechanism: str = "laplace",
    delta: float = 0.0,
    numerical_sensitivity: Optional[Dict[str, float]] = None,
    epsilon_allocation: Optional[Dict[str, float]] = None,
    numeric_mechanism: Optional[str] = None,
    categorical_mechanism: Optional[str] = None,
    categorical_columns: Optional[List[str]] = None,
    categorical_epsilon_allocation: Optional[Dict[str, float]] = None,
    threshold_rules: Optional[List[Dict[str, Any]]] = None,
    above_threshold_values: Optional[Dict[str, float]] = None,
    above_threshold_return_none: bool = False,
    max_epsilon_per_attribute: float = 0.1,
) -> pd.DataFrame:
    """
    Apply DP mechanism to numerical columns in Adult dataframe.

    Args:
        df: Pandas DataFrame with Adult schema
        epsilon: Privacy budget
        mechanism: DP mechanism name

    Returns:
        DataFrame with DP-protected numerical columns
    """
    integration = DPAnonymizationIntegration(
        epsilon=epsilon,
        max_epsilon_per_attribute=max_epsilon_per_attribute,
    )

    # Known numerical columns in Adult dataset
    adult_numerical = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

    # Coerce known numeric columns to float (handles object/string dtypes from parquet)
    df = df.copy()
    for col in adult_numerical:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=[col for col in adult_numerical if col in df.columns], inplace=True)

    # Identify numerical columns (excluding sensitive attribute income)
    numerical_cols = [
        col
        for col in df.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns
        if col not in ["income"]
    ]

    # Define sensitivities for Adult dataset
    sensitivities = numerical_sensitivity or {
        "age": 100.0,
        "education-num": 16.0,
        "capital-gain": 100000.0,
        "capital-loss": 5000.0,
        "hours-per-week": 168.0,
    }

    categorical_cols = _resolve_categorical_columns(df, categorical_columns)

    # Filter to only existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    if not numerical_cols and not categorical_cols and not threshold_rules:
        return df.copy()
    sensitivities = {col: sensitivities.get(col, 1.0) for col in numerical_cols}

    mechanism_normalized = (mechanism or "laplace").strip().lower()
    allocation = epsilon_allocation or {
        col: epsilon / len(numerical_cols) for col in numerical_cols
    }

    if mechanism_normalized == "laplace":
        return integration.apply_dp_to_numerical_columns(
            df=df,
            numerical_columns=numerical_cols,
            sensitivities=sensitivities,
            epsilon_allocation=allocation,
        )

    if mechanism_normalized == "exponential":
        return _apply_exponential_to_numerical_columns(
            df=df,
            numerical_columns=numerical_cols,
            sensitivities=sensitivities,
            epsilon_allocation=allocation,
        )

    if mechanism_normalized == "above_threshold":
        return _apply_above_threshold_to_numerical_columns(
            df=df,
            numerical_columns=numerical_cols,
            epsilon_allocation=allocation,
            delta=delta,
            threshold_values=above_threshold_values,
            return_none_for_below=above_threshold_return_none,
        )

    if mechanism_normalized == "mixed":
        df_mixed = df.copy()

        if numerical_cols:
            selected_numeric_mechanism = (numeric_mechanism or "laplace").strip().lower()
            if selected_numeric_mechanism != "laplace":
                raise ValueError(f"Unsupported numeric mechanism: {selected_numeric_mechanism}")
            df_mixed = integration.apply_dp_to_numerical_columns(
                df=df_mixed,
                numerical_columns=numerical_cols,
                sensitivities=sensitivities,
                epsilon_allocation=allocation,
            )

        if categorical_cols:
            selected_categorical_mechanism = (categorical_mechanism or "exponential").strip().lower()
            if selected_categorical_mechanism != "exponential":
                raise ValueError(
                    f"Unsupported categorical mechanism: {selected_categorical_mechanism}"
                )
            df_mixed = _apply_exponential_to_categorical_columns(
                df=df_mixed,
                categorical_columns=categorical_cols,
                epsilon_allocation=_build_categorical_allocation(
                    categorical_cols,
                    epsilon,
                    categorical_epsilon_allocation,
                ),
            )

        if threshold_rules:
            df_mixed = _apply_threshold_rules(
                df=df_mixed,
                threshold_rules=threshold_rules,
                default_epsilon=epsilon,
                default_delta=delta,
                default_return_none=above_threshold_return_none,
            )

        return df_mixed

    raise ValueError(f"Unsupported DP mechanism: {mechanism}")


def _apply_exponential_to_numerical_columns(
    df: pd.DataFrame,
    numerical_columns: List[str],
    sensitivities: Dict[str, float],
    epsilon_allocation: Dict[str, float],
) -> pd.DataFrame:
    """Apply Exponential mechanism by sampling per-row from value candidates."""
    df_noisy = df.copy()

    for col in numerical_columns:
        if col not in df_noisy.columns:
            continue

        eps = float(epsilon_allocation.get(col, 1e-6))
        sens = max(float(sensitivities.get(col, 1.0)), 1e-9)
        mechanism = ExponentialMechanism(
            epsilon=max(eps, 1e-6),
            sensitivity=sens,
            name=f"EXP_{col}",
        )

        series = pd.to_numeric(df_noisy[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        # Candidate outputs are taken from empirical quantiles to bound runtime.
        quantiles = np.linspace(0.0, 1.0, num=min(15, max(3, len(valid.unique()))))
        candidates = np.unique(np.quantile(valid.values, quantiles)).tolist()
        if not candidates:
            continue

        updated_values: List[float] = []
        for value in series.tolist():
            if pd.isna(value):
                updated_values.append(np.nan)
                continue

            utilities = [-(abs(float(value) - float(c)) / sens) for c in candidates]
            selected = mechanism.select(candidates, utilities)
            updated_values.append(float(selected))

        df_noisy[col] = updated_values

    return df_noisy


def _apply_above_threshold_to_numerical_columns(
    df: pd.DataFrame,
    numerical_columns: List[str],
    epsilon_allocation: Dict[str, float],
    delta: float,
    threshold_values: Optional[Dict[str, float]],
    return_none_for_below: bool,
) -> pd.DataFrame:
    """Apply Above-Threshold mechanism to each numerical column independently."""
    df_noisy = df.copy()
    thresholds = threshold_values or {}

    for col in numerical_columns:
        if col not in df_noisy.columns:
            continue

        series = pd.to_numeric(df_noisy[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            continue

        eps = max(float(epsilon_allocation.get(col, 1e-6)), 1e-6)
        threshold = float(thresholds.get(col, float(valid.median())))
        mechanism = AboveThresholdMechanism(
            epsilon=eps,
            threshold=threshold,
            delta=max(delta, 0.0),
            name=f"AT_{col}",
        )

        updated_values: List[float] = []
        for value in series.tolist():
            if pd.isna(value):
                updated_values.append(np.nan)
                continue

            answer = mechanism.query(float(value), description=f"Column {col}")
            if answer is None:
                if return_none_for_below:
                    updated_values.append(np.nan)
                else:
                    updated_values.append(float(value))
            else:
                updated_values.append(float(answer))

        df_noisy[col] = updated_values

    return df_noisy


def _apply_exponential_to_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: List[str],
    epsilon_allocation: Dict[str, float],
) -> pd.DataFrame:
    """Apply Exponential mechanism to categorical columns."""
    df_noisy = df.copy()

    for col in categorical_columns:
        if col not in df_noisy.columns:
            continue

        series = df_noisy[col].astype("string")
        valid = series.dropna()
        if valid.empty:
            continue

        options = valid.value_counts(dropna=True).index.tolist()
        if len(options) <= 1:
            continue

        frequencies = valid.value_counts(normalize=True).to_dict()
        mechanism = ExponentialMechanism(
            epsilon=max(float(epsilon_allocation.get(col, 1e-6)), 1e-6),
            sensitivity=1.0,
            name=f"EXP_CAT_{col}",
        )

        updated_values: List[Optional[str]] = []
        for value in series.tolist():
            if pd.isna(value):
                updated_values.append(pd.NA)
                continue

            utilities = [
                (1.0 if option == value else 0.0) + float(frequencies.get(option, 0.0))
                for option in options
            ]
            selected = mechanism.select(options, utilities)
            updated_values.append(str(selected))

        df_noisy[col] = pd.Series(updated_values, index=df_noisy.index, dtype="string")

    return df_noisy


def _apply_threshold_rules(
    df: pd.DataFrame,
    threshold_rules: List[Dict[str, Any]],
    default_epsilon: float,
    default_delta: float,
    default_return_none: bool,
) -> pd.DataFrame:
    """Apply configurable Above-Threshold rules and emit result columns."""
    df_rules = df.copy()

    for index, rule in enumerate(threshold_rules, start=1):
        column = str(rule.get("column", "")).strip()
        if not column or column not in df_rules.columns or "threshold" not in rule:
            continue

        threshold = float(rule["threshold"])
        epsilon = max(float(rule.get("epsilon", default_epsilon)), 1e-6)
        delta = max(float(rule.get("delta", default_delta)), 0.0)
        mode = str(rule.get("mode", "flag")).strip().lower()
        result_column = str(
            rule.get("result_column") or f"{column}_threshold_rule_{index}"
        )
        return_none_for_below = bool(
            rule.get("return_none_for_below", default_return_none)
        )

        series = pd.to_numeric(df_rules[column], errors="coerce")
        mechanism = AboveThresholdMechanism(
            epsilon=epsilon,
            threshold=threshold,
            delta=delta,
            name=f"AT_RULE_{column}_{index}",
        )

        outputs: List[Any] = []
        for value in series.tolist():
            if pd.isna(value):
                outputs.append(np.nan)
                continue

            answer = mechanism.query(float(value), description=result_column)
            if mode == "flag":
                outputs.append(1 if answer is not None else 0)
            elif answer is None:
                outputs.append(np.nan if return_none_for_below else float(value))
            else:
                outputs.append(float(answer))

        df_rules[result_column] = outputs

    return df_rules


def _build_categorical_allocation(
    categorical_columns: List[str],
    epsilon: float,
    categorical_epsilon_allocation: Optional[Dict[str, float]],
) -> Dict[str, float]:
    if categorical_epsilon_allocation:
        return {
            col: float(categorical_epsilon_allocation.get(col, 1e-6))
            for col in categorical_columns
        }

    if not categorical_columns:
        return {}

    epsilon_per_column = epsilon / len(categorical_columns)
    return {col: epsilon_per_column for col in categorical_columns}


def _resolve_categorical_columns(
    df: pd.DataFrame,
    configured_columns: Optional[List[str]],
) -> List[str]:
    if configured_columns:
        return [col for col in configured_columns if col in df.columns]

    return [
        col for col in df.columns
        if col != "income" and not pd.api.types.is_numeric_dtype(df[col])
    ]


def _download_object_to_temp(
    client: Any, bucket_name: str, object_key: str, suffix: str
) -> str:
    """Download object from MinIO to temporary file."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        client.download_fileobj(Bucket=bucket_name, Key=object_key, Fileobj=temp_file)
    finally:
        temp_file.close()
    return temp_file.name


def _read_parquet_with_pandas(local_path: str) -> pd.DataFrame:
    """Read parquet file using pandas (no PySpark needed)."""
    return pd.read_parquet(local_path)


def _build_dp_protected_key(
    clean_object_key: str,
    epsilon: float,
    profile_name: str = "default",
    output_name_template: Optional[str] = None,
) -> str:
    """Build MinIO key for DP-protected file."""
    path = Path(clean_object_key)
    base = path.stem

    # Remove existing suffixes
    for suffix in ["_clean", "_anon_k", "_ldiv"]:
        if base.endswith(suffix):
            # Handle _anon_k10 case
            if suffix == "_anon_k" and base[-3:-1].isdigit():
                base = base[: base.rfind("_anon_k")]
            else:
                base = base[: -len(suffix)]

    # Format epsilon for filename (e.g., 0.3 -> dp_e0_3)
    epsilon_str = f"{epsilon:.2f}".replace(".", "_")
    source_version = _extract_source_version(path)
    run_version = datetime.now().strftime("%Y%m%d%H%M%S")
    template = (
        output_name_template
        or "{base}_dp_e{epsilon}_src{source_version}_run{run_version}.parquet"
    )
    dp_name = template.format(
        base=base,
        profile=profile_name,
        epsilon=epsilon_str,
        source_version=source_version,
        run_version=run_version,
    )

    return str(path.with_name(dp_name)).replace("\\", "/")


def _extract_source_version(path: Path) -> str:
    parent = path.parent.name
    if parent and parent != ".":
        return parent
    return "unknown"


def _is_adult_dataframe(df: pd.DataFrame) -> bool:
    """Validate that dataframe has Adult dataset schema."""
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
