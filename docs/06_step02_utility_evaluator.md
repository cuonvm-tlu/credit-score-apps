# BƯỚC 2 — Implement `utility_evaluator.py`

> **File tạo mới:** `preprocess-svc/app/core/utility_evaluator.py`
> **Mục tiêu:** Đo lường Privacy-Utility Tradeoff — yêu cầu quan trọng nhất còn thiếu

---

## 2.1 Tạo File `utility_evaluator.py`

```python
"""
utility_evaluator.py
Đo lường privacy-utility tradeoff cho Adult Census Income dataset.

Metrics:
  1. NCP  — Information loss từ generalization (Mondrian output)
  2. DP Query Error — Sai số trung bình Laplace noise theo cột
  3. Classification Accuracy — Logistic Regression trước/sau anonymization
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_range_or_value(val) -> float:
    """Chuyển range string \'20-30\' → 25.0 (midpoint) hoặc parse float."""
    try:
        return float(val)
    except (ValueError, TypeError):
        s = str(val)
        if "-" in s:
            parts = s.split("-", 1)
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except ValueError:
                pass
    return 0.0


def _interpret_ncp(ncp: float) -> str:
    if ncp < 0.10:
        return "Excellent — very little information loss"
    elif ncp < 0.30:
        return "Good — moderate information loss"
    elif ncp < 0.50:
        return "Acceptable — significant information loss"
    return "Poor — high information loss, reduce k or l"


def _epsilon_to_level(eps: float) -> str:
    if eps <= 0.1:
        return "VERY_HIGH"
    elif eps <= 0.5:
        return "HIGH"
    elif eps <= 1.0:
        return "MEDIUM"
    elif eps <= 5.0:
        return "LOW"
    return "VERY_LOW"


# ─── NCP ─────────────────────────────────────────────────────────────────────

def build_ncp_report(ncp_raw: float, num_records: int, method: str) -> dict:
    """
    Wrap NCP đã trả về từ Mondrian vào dict chuẩn.

    Args:
        ncp_raw: Giá trị NCP từ mondrian() / mondrian_l_diversity()
        num_records: Số bản ghi sau anonymization
        method: "k_anonymity" hoặc "l_diversity"
    """
    return {
        "method": method,
        "ncp": round(float(ncp_raw), 6),
        "interpretation": _interpret_ncp(ncp_raw),
        "num_records": num_records,
    }


# ─── DP Query Error ──────────────────────────────────────────────────────────

NUMERICAL_COLS_ADULT = [
    "age", "education-num", "capital-gain", "capital-loss", "hours-per-week"
]


def measure_dp_query_error(
    original_df: pd.DataFrame,
    dp_df: pd.DataFrame,
    epsilon: float,
    numerical_columns: Optional[list] = None,
) -> dict:
    """
    Đo sai số DP noise trên từng cột numerical.

    Returns dict chứa:
      - column_metrics: {col: {original_mean, dp_mean, mae, relative_error_pct}}
      - avg_relative_error_pct: trung bình relative error
      - epsilon: giá trị epsilon đã dùng
      - privacy_level: mức độ privacy
    """
    if numerical_columns is None:
        numerical_columns = NUMERICAL_COLS_ADULT

    col_metrics = {}
    for col in numerical_columns:
        if col not in original_df.columns or col not in dp_df.columns:
            logger.warning(f"Column {col} not found, skipping")
            continue

        orig_mean = float(original_df[col].mean())
        dp_mean   = float(dp_df[col].mean())
        orig_std  = float(original_df[col].std())
        dp_std    = float(dp_df[col].std())

        mae = abs(orig_mean - dp_mean)
        rel_err = (mae / abs(orig_mean) * 100) if orig_mean != 0 else float("inf")

        col_metrics[col] = {
            "original_mean":      round(orig_mean, 4),
            "dp_mean":            round(dp_mean,   4),
            "mean_absolute_error": round(mae,       4),
            "relative_error_pct": round(rel_err,    2),
            "original_std":       round(orig_std,   4),
            "dp_std":             round(dp_std,     4),
        }

    avg_rel = float(np.mean([v["relative_error_pct"]
                              for v in col_metrics.values()])) if col_metrics else 0.0

    return {
        "epsilon": epsilon,
        "privacy_level": _epsilon_to_level(epsilon),
        "avg_relative_error_pct": round(avg_rel, 2),
        "column_metrics": col_metrics,
    }


# ─── Classification Accuracy ─────────────────────────────────────────────────

def measure_classification_accuracy(
    df: pd.DataFrame,
    target_col: str = "income",
    feature_cols: Optional[list] = None,
    cv_folds: int = 5,
) -> dict:
    """
    Đo accuracy Logistic Regression trên dataframe bằng cross-validation.

    Args:
        df: DataFrame (original, k-anon, l-div, hoặc dp-protected)
        target_col: Cột nhãn
        feature_cols: Cột feature (None = tất cả trừ target)
        cv_folds: Số fold cross-validation

    Returns dict:
        accuracy, std, num_records, num_features, cv_folds
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    df_clean = df[feature_cols + [target_col]].dropna().copy()

    if len(df_clean) < 50:
        logger.warning("Too few records for accuracy measurement")
        return {"accuracy": None, "error": "Insufficient data (<50 records)"}

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    # Encode target
    if y.dtype == object or str(y.dtype) == "string":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))

    # Encode categorical features
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Handle range strings từ Mondrian generalization ("20-30" → 25)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(_parse_range_or_value)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")

    return {
        "accuracy":     round(float(scores.mean()), 4),
        "std":          round(float(scores.std()),  4),
        "num_records":  len(df_clean),
        "num_features": len(feature_cols),
        "cv_folds":     cv_folds,
    }


# ─── Full Privacy-Utility Report ─────────────────────────────────────────────

def build_privacy_utility_report(
    original_df: pd.DataFrame,
    k_anon_df:   Optional[pd.DataFrame] = None,
    l_div_df:    Optional[pd.DataFrame] = None,
    dp_df:       Optional[pd.DataFrame] = None,
    k_ncp:       float = 0.0,
    l_ncp:       float = 0.0,
    epsilon:     float = 0.3,
    numerical_cols: Optional[list] = None,
) -> dict:
    """
    Tổng hợp toàn bộ privacy-utility report.

    Workflow:
      1. Đo accuracy trên original data (baseline)
      2. Đo NCP + accuracy trên k-anonymized data
      3. Đo NCP + accuracy trên l-diversified data
      4. Đo DP query error + accuracy trên dp-protected data
      5. Tính accuracy_loss_pct cho từng bước so với baseline

    Args:
        original_df: DataFrame sau Spark cleaning (chưa anonymize)
        k_anon_df:   DataFrame sau k-anonymity
        l_div_df:    DataFrame sau l-diversity
        dp_df:       DataFrame sau DP
        k_ncp:       NCP trả về từ mondrian()
        l_ncp:       NCP trả về từ mondrian_l_diversity()
        epsilon:     Epsilon đã dùng cho DP
        numerical_cols: List cột số để đo DP error

    Returns:
        dict với keys: original, k_anonymity, l_diversity, differential_privacy
    """
    report: dict = {}

    # --- Baseline ---
    orig_acc_result = measure_classification_accuracy(original_df)
    orig_acc = orig_acc_result.get("accuracy")
    report["original"] = {
        "num_records": len(original_df),
        **orig_acc_result,
    }

    # --- K-Anonymity ---
    if k_anon_df is not None:
        k_acc_result = measure_classification_accuracy(k_anon_df)
        k_acc = k_acc_result.get("accuracy")
        acc_loss = None
        if orig_acc is not None and k_acc is not None:
            acc_loss = round((orig_acc - k_acc) * 100, 2)
        report["k_anonymity"] = {
            **build_ncp_report(k_ncp, len(k_anon_df), "k_anonymity"),
            "records_dropped": len(original_df) - len(k_anon_df),
            **k_acc_result,
            "accuracy_loss_pct": acc_loss,
        }

    # --- L-Diversity ---
    if l_div_df is not None:
        l_acc_result = measure_classification_accuracy(l_div_df)
        l_acc = l_acc_result.get("accuracy")
        acc_loss = None
        if orig_acc is not None and l_acc is not None:
            acc_loss = round((orig_acc - l_acc) * 100, 2)
        report["l_diversity"] = {
            **build_ncp_report(l_ncp, len(l_div_df), "l_diversity"),
            "records_dropped": len(original_df) - len(l_div_df),
            **l_acc_result,
            "accuracy_loss_pct": acc_loss,
        }

    # --- Differential Privacy ---
    if dp_df is not None:
        dp_err   = measure_dp_query_error(original_df, dp_df, epsilon, numerical_cols)
        dp_acc_result = measure_classification_accuracy(dp_df)
        dp_acc   = dp_acc_result.get("accuracy")
        acc_loss = None
        if orig_acc is not None and dp_acc is not None:
            acc_loss = round((orig_acc - dp_acc) * 100, 2)
        report["differential_privacy"] = {
            "epsilon":       epsilon,
            "privacy_level": _epsilon_to_level(epsilon),
            "query_error":   dp_err,
            **dp_acc_result,
            "accuracy_loss_pct": acc_loss,
        }

    return report
```

---

## 2.2 Thêm `scikit-learn` vào `requirements.txt`

```
# preprocess-svc/requirements.txt — THÊM:
scikit-learn>=1.4.0
```

---

## 2.3 Update `anonymize_k_anonymity.py` — Trả Về NCP

File hiện tại bỏ qua `ncp` và `rtime` từ Mondrian. Sửa để trả về:

```python
# Trong anonymize_cleaned_adult_k_anonymity_and_upload() — thêm return ncp:
def anonymize_cleaned_adult_k_anonymity_and_upload(
    client, clean_bucket, clean_object_key, k=10
) -> tuple[str, float]:          # ← đổi return type
    ...
    anon_df, ncp, rtime = anonymize_adult_dataframe(df, k=k)
    ...
    return f"{clean_bucket}/{anon_object_key}", ncp   # ← trả về cả ncp
```

Tương tự cho `anonymize_l_diversity.py`:

```python
def anonymize_cleaned_adult_l_diversity_and_upload(
    client, clean_bucket, clean_object_key, l_value=2
) -> tuple[str, float]:
    ...
    anon_df, ncp, rtime = anonymize_adult_dataframe_l_diversity(df, l_value=l_value)
    ...
    return f"{clean_bucket}/{anon_object_key}", ncp
```

---

## 2.4 Update `routes.py` — Dùng `utility_evaluator`

```python
# Thêm imports vào routes.py:
import io
import os
import tempfile
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.anonymize_k_anonymity import anonymize_cleaned_adult_k_anonymity_and_upload
from app.core.anonymize_l_diversity import anonymize_cleaned_adult_l_diversity_and_upload
from app.core.spark_cleaner import spark_clean_and_upload
from app.core.kafka_producer import send_cleaning_success_event
from app.core.minio_client import ensure_bucket, get_minio_client
from app.core.dp_anonymization_adapter import apply_dp_protection_and_upload
from app.core.utility_evaluator import build_privacy_utility_report   # ← MỚI
from app.core.spark_session import get_spark_session                   # ← MỚI

router = APIRouter()

EPSILON = 0.3
K_VALUE = 10
L_VALUE = 2


def _load_parquet_from_minio(client, path: str) -> "pd.DataFrame":
    """Download parquet từ MinIO và load bằng pandas."""
    import pandas as pd
    bucket, key = path.split("/", 1)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
    try:
        client.download_fileobj(Bucket=bucket, Key=key, Fileobj=tmp)
    finally:
        tmp.close()
    return pd.read_parquet(tmp.name)


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)) -> dict:
    version_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    client = get_minio_client()

    ensure_bucket(client, "landing-zone")
    ensure_bucket(client, "clean-zone")

    landing_zone_paths: List[str] = []
    clean_zone_paths: List[str]   = []
    privacy_report: dict          = {}

    for upload_file in files:
        filename = os.path.basename(upload_file.filename or "")
        if not filename:
            raise HTTPException(status_code=400, detail="File must have a filename.")

        object_key = f"{version_folder}/{filename}"
        file_bytes = await upload_file.read()

        # Upload raw → landing-zone
        client.put_object(
            Bucket="landing-zone", Key=object_key,
            Body=io.BytesIO(file_bytes), ContentLength=len(file_bytes),
            ContentType=upload_file.content_type or "application/octet-stream",
        )
        landing_zone_paths.append(f"landing-zone/{object_key}")

        if filename in {"adult.data", "adult.test"} or filename.lower().endswith(".csv"):
            # ── Step 1: Spark clean ──
            cleaned_path = spark_clean_and_upload(
                client=client, source_bucket="landing-zone",
                source_key=object_key, clean_bucket="clean-zone",
                version_folder=version_folder, original_filename=filename,
            )
            clean_zone_paths.append(cleaned_path)
            clean_bucket, clean_key = cleaned_path.split("/", 1)

            # ── Step 2: K-Anonymity ──
            k_path, k_ncp = anonymize_cleaned_adult_k_anonymity_and_upload(
                client=client, clean_bucket=clean_bucket,
                clean_object_key=clean_key, k=K_VALUE,
            )
            clean_zone_paths.append(k_path)

            # ── Step 3: L-Diversity ──
            l_path, l_ncp = anonymize_cleaned_adult_l_diversity_and_upload(
                client=client, clean_bucket=clean_bucket,
                clean_object_key=clean_key, l_value=L_VALUE,
            )
            clean_zone_paths.append(l_path)

            # ── Step 4: Differential Privacy ──
            dp_path = apply_dp_protection_and_upload(
                client=client, clean_bucket=clean_bucket,
                clean_object_key=clean_key, epsilon=EPSILON,
            )
            clean_zone_paths.append(dp_path)

            # ── Step 5: Evaluate Privacy-Utility Tradeoff ──
            try:
                original_df = _load_parquet_from_minio(client, cleaned_path)
                k_anon_df   = _load_parquet_from_minio(client, k_path)
                l_div_df    = _load_parquet_from_minio(client, l_path)
                dp_df       = _load_parquet_from_minio(client, dp_path)

                privacy_report = build_privacy_utility_report(
                    original_df=original_df,
                    k_anon_df=k_anon_df, l_div_df=l_div_df, dp_df=dp_df,
                    k_ncp=k_ncp, l_ncp=l_ncp, epsilon=EPSILON,
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Privacy report failed: {e}")
                privacy_report = {"error": str(e)}

    if clean_zone_paths:
        send_cleaning_success_event(version_folder, clean_zone_paths)

    return {
        "message":            "Upload completed",
        "landing_zone_paths": landing_zone_paths,
        "clean_zone_paths":   clean_zone_paths,
        "privacy_report":     privacy_report,
    }
```

---

## 2.5 Commit Bước 2

```bash
git add preprocess-svc/app/core/utility_evaluator.py
git add preprocess-svc/app/core/anonymize_k_anonymity.py
git add preprocess-svc/app/core/anonymize_l_diversity.py
git add preprocess-svc/app/api/routes.py
git add preprocess-svc/requirements.txt
git commit -m "feat: add utility_evaluator, return NCP from Mondrian, add privacy_report to API"
```

---

## 2.6 Test Nhanh `utility_evaluator`

```python
# preprocess-svc/test_utility_evaluator.py
import pandas as pd
import numpy as np
from app.core.utility_evaluator import (
    measure_dp_query_error,
    measure_classification_accuracy,
    build_privacy_utility_report,
)

# Tạo data giả để test
np.random.seed(42)
n = 1000
df_orig = pd.DataFrame({
    "age":          np.random.randint(18, 90, n),
    "education-num": np.random.randint(1, 16, n),
    "capital-gain": np.random.randint(0, 50000, n),
    "capital-loss": np.random.randint(0, 5000,  n),
    "hours-per-week": np.random.randint(10, 80, n),
    "income":       np.random.choice([0, 1], n),
})

# DP dataframe (thêm noise)
df_dp = df_orig.copy()
df_dp["age"] = df_dp["age"] + np.random.laplace(0, 100/0.3, n)

# Test DP error
err = measure_dp_query_error(df_orig, df_dp, epsilon=0.3)
print("DP Error:", err["avg_relative_error_pct"], "%")

# Test accuracy
acc = measure_classification_accuracy(df_orig)
print("Accuracy:", acc["accuracy"])

# Test full report
report = build_privacy_utility_report(
    original_df=df_orig, dp_df=df_dp, epsilon=0.3
)
print("Report keys:", list(report.keys()))
```

Chạy:
```bash
cd preprocess-svc
python test_utility_evaluator.py
```

---

## 2.7 Checklist Bước 2

- [ ] Tạo `preprocess-svc/app/core/utility_evaluator.py`
- [ ] Thêm `scikit-learn` vào `requirements.txt`
- [ ] Sửa `anonymize_k_anonymity.py` trả về `(path, ncp)`
- [ ] Sửa `anonymize_l_diversity.py` trả về `(path, ncp)`
- [ ] Cập nhật `routes.py` tích hợp `build_privacy_utility_report`
- [ ] Test với data giả
- [ ] Commit

**Tiếp theo:** [Bước 3 — Privacy Budget Accountant](./07_step03_budget_accountant.md)
