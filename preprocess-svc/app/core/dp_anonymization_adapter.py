"""
Adapter layer for Differential Privacy integration with API routes.

Provides functions to apply DP post-processing to anonymized data
and upload results to MinIO, similar to k-anonymity and l-diversity adapters.
"""

import os
import tempfile
from pathlib import Path
from typing import Any
import logging

import pandas as pd

from app.core.dp_mechanisms import DPAnonymizationIntegration

logger = logging.getLogger(__name__)


def apply_dp_protection_and_upload(
    client: Any,
    clean_bucket: str,
    clean_object_key: str,
    epsilon: float = 0.3,
) -> str:
    """
    Download anonymized parquet from MinIO, apply Differential Privacy noise,
    then upload DP-protected parquet.
    
    Args:
        client: MinIO client object
        clean_bucket: Bucket name (e.g., "clean-zone")
        clean_object_key: Object key of anonymized parquet
        epsilon: Privacy budget for DP noise (default 0.3)
    
    Returns:
        MinIO path in format: "<bucket>/<key>" pointing to DP-protected file
    
    Raises:
        ValueError: If input data schema is invalid
    """
    logger.info(f"Applying DP protection with epsilon={epsilon:.4f}")
    
    # Step 1: Download anonymized parquet from MinIO
    local_clean_path = _download_object_to_temp(
        client, clean_bucket, clean_object_key, suffix=".parquet"
    )
    # Step 2: Read with Pandas
    df = _read_parquet_with_pandas(local_clean_path)
    
    # Step 3: Validate Adult dataframe schema
    if not _is_adult_dataframe(df):
        raise ValueError("Input parquet is not Adult cleaned schema.")
    
    # Step 4: Apply DP protection
    logger.info("Applying Differential Privacy mechanism (Laplace noise)")
    dp_protected_df = _apply_dp_to_dataframe(df, epsilon)
    
    # Step 5: Upload DP-protected parquet to MinIO
    dp_object_key = _build_dp_protected_key(clean_object_key, epsilon)
    local_dp_path = Path(tempfile.gettempdir()) / Path(dp_object_key).name
    
    dp_protected_df.to_parquet(local_dp_path, index=False)
    
    with local_dp_path.open("rb") as parquet_file:
        client.put_object(
            Bucket=clean_bucket,
            Key=dp_object_key,
            Body=parquet_file,
            ContentLength=os.path.getsize(local_dp_path),
            ContentType="application/octet-stream",
        )
    
    logger.info(f"DP-protected file uploaded: {clean_bucket}/{dp_object_key}")
    
    return f"{clean_bucket}/{dp_object_key}"


def _apply_dp_to_dataframe(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    """
    Apply DP noise to numerical columns in Adult dataframe.
    
    Args:
        df: Pandas DataFrame with Adult schema
        epsilon: Privacy budget
    
    Returns:
        DataFrame with DP-protected numerical columns
    """
    integration = DPAnonymizationIntegration(epsilon=epsilon)
    
    # Identify numerical columns (excluding sensitive attribute income)
    numerical_cols = [
        col for col in df.select_dtypes(include=['float64', 'int64']).columns
        if col not in ['income']
    ]
    
    # Define sensitivities for Adult dataset
    sensitivities = {
        'age': 100.0,  # Age range: 0-100
        'education-num': 16.0,  # Education levels: 0-16
        'capital-gain': 100000.0,  # Up to 100k
        'capital-loss': 5000.0,  # Up to 5k
        'hours-per-week': 168.0,  # Up to 168 hours
    }
    
    # Filter to only existing columns
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    sensitivities = {col: sensitivities.get(col, 1.0) for col in numerical_cols}
    
    # Apply DP noise
    try:
        df_protected = integration.apply_dp_to_numerical_columns(
            df=df,
            numerical_columns=numerical_cols,
            sensitivities=sensitivities,
            epsilon_allocation={col: epsilon / len(numerical_cols) for col in numerical_cols}
        )
        logger.info(f"DP applied to {len(numerical_cols)} numerical columns")
        return df_protected
    except Exception as e:
        logger.error(f"Error applying DP: {e}")
        raise


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


def _build_dp_protected_key(clean_object_key: str, epsilon: float) -> str:
    """Build MinIO key for DP-protected file."""
    path = Path(clean_object_key)
    base = path.stem
    
    # Remove existing suffixes
    for suffix in ["_clean", "_anon_k", "_ldiv"]:
        if base.endswith(suffix):
            # Handle _anon_k10 case
            if suffix == "_anon_k" and base[-3:-1].isdigit():
                base = base[:base.rfind("_anon_k")]
            else:
                base = base[:-len(suffix)]
    
    # Format epsilon for filename (e.g., 0.3 -> dp_e0_3)
    epsilon_str = f"{epsilon:.2f}".replace(".", "_")
    dp_name = f"{base}_dp_e{epsilon_str}.parquet"
    
    return str(path.with_name(dp_name)).replace("\\", "/")


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
