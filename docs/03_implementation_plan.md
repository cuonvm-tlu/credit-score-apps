# Kế Hoạch Implement: Privacy-Utility Tradeoff Evaluation

> **Đây là phần quan trọng nhất còn thiếu trong project**  
> Yêu cầu: *"Evaluate the trade-off between privacy levels and analytical accuracy"*

---

## Bước 1: Fix Bug Ngay Lập Tức

### 1.1 Xóa `breakpoint()` trong DP integration

**File:** `preprocess-svc/app/core/dp_mechanisms/dp_anonymization_integration.py` — dòng 71

```python
# XÓA dòng này:
breakpoint()
```

### 1.2 Thêm missing import vào routes.py

**File:** `preprocess-svc/app/api/routes.py` — thêm vào đầu file:

```python
from app.core.dp_anonymization_adapter import apply_dp_protection_and_upload
```

### 1.3 Bỏ comment income mapping trong spark_cleaner.py

**File:** `preprocess-svc/app/core/spark_cleaner.py` — bỏ comment S4:

```python
# Bỏ comment khối này:
df = df.withColumn("income",
    F.when(F.col("income") == "<=50K", F.lit(0))
     .when(F.col("income") == ">50K",  F.lit(1))
     .otherwise(None).cast(IntegerType()))
```

---

## Bước 2: Tạo `utility_evaluator.py`

**File mới:** `preprocess-svc/app/core/utility_evaluator.py`

```python
"""
utility_evaluator.py — Đo lường Privacy-Utility Tradeoff

Metrics:
1. NCP (Normalized Certainty Penalty) — information loss từ generalization
2. DP Query Error — sai số trung bình của Laplace noise
3. Classification Accuracy — so sánh accuracy trước/sau anonymization
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


# ─── NCP Calculation ──────────────────────────────────────────────────────────

def compute_ncp_from_mondrian(ncp_raw: float, num_records: int) -> dict:
    """
    Wrapper xung quanh NCP đã có từ Mondrian.
    Mondrian trả về NCP trực tiếp — chỉ cần ghi lại.
    """
    return {
        "ncp": round(ncp_raw, 4),
        "interpretation": _interpret_ncp(ncp_raw),
        "num_records": num_records,
    }


def _interpret_ncp(ncp: float) -> str:
    if ncp < 0.1:
        return "Excellent utility — very little information loss"
    elif ncp < 0.3:
        return "Good utility — moderate information loss"
    elif ncp < 0.5:
        return "Acceptable — significant information loss"
    else:
        return "Poor utility — high information loss, consider reducing k/l"


# ─── DP Query Error ───────────────────────────────────────────────────────────

def measure_dp_query_error(
    original_df: pd.DataFrame,
    dp_df: pd.DataFrame,
    numerical_columns: list[str],
    epsilon: float,
) -> dict:
    """
    Đo sai số của DP noise trên từng cột numerical.
    
    Metrics:
    - mean_absolute_error: |E[original] - E[dp]|
    - relative_error: MAE / |E[original]|
    - theoretical_bound: sensitivity / epsilon (Laplace noise scale)
    """
    results = {}
    
    for col in numerical_columns:
        if col not in original_df.columns or col not in dp_df.columns:
            continue
        
        orig_mean = original_df[col].mean()
        dp_mean = dp_df[col].mean()
        orig_std = original_df[col].std()
        dp_std = dp_df[col].std()
        
        mae = abs(orig_mean - dp_mean)
        rel_error = mae / abs(orig_mean) if orig_mean != 0 else float('inf')
        
        results[col] = {
            "original_mean": round(float(orig_mean), 4),
            "dp_mean": round(float(dp_mean), 4),
            "mean_absolute_error": round(float(mae), 4),
            "relative_error_pct": round(float(rel_error * 100), 2),
            "original_std": round(float(orig_std), 4),
            "dp_std": round(float(dp_std), 4),
        }
    
    avg_rel_error = np.mean([v["relative_error_pct"] for v in results.values()])
    
    return {
        "epsilon": epsilon,
        "column_metrics": results,
        "avg_relative_error_pct": round(float(avg_rel_error), 2),
        "privacy_level": _epsilon_to_level(epsilon),
    }


def _epsilon_to_level(epsilon: float) -> str:
    if epsilon <= 0.1:
        return "VERY_HIGH"
    elif epsilon <= 0.5:
        return "HIGH"
    elif epsilon <= 1.0:
        return "MEDIUM"
    elif epsilon <= 5.0:
        return "LOW"
    else:
        return "VERY_LOW"


# ─── Classification Accuracy ──────────────────────────────────────────────────

def measure_classification_accuracy(
    df: pd.DataFrame,
    target_col: str = "income",
    feature_cols: Optional[list] = None,
    cv_folds: int = 5,
) -> dict:
    """
    Đo classification accuracy của Logistic Regression trên dataframe.
    Dùng cross-validation để đánh giá chính xác.
    
    Returns accuracy score (0-1).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    
    df_clean = df[feature_cols + [target_col]].dropna().copy()
    
    if len(df_clean) < 100:
        logger.warning("Too few records for reliable accuracy measurement")
        return {"accuracy": None, "error": "Insufficient data"}
    
    # Encode categorical columns
    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()
    
    # Label encode target if needed
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    # Encode categorical features
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Handle range strings like "20-30" from Mondrian generalization
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].apply(_parse_range_or_value)
    
    X = X.fillna(X.median(numeric_only=True))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring="accuracy")
    
    return {
        "accuracy": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "num_records": len(df_clean),
        "num_features": len(feature_cols),
        "cv_folds": cv_folds,
    }


def _parse_range_or_value(val):
    """Parse Mondrian range string '20-30' → 25 (midpoint)"""
    try:
        return float(val)
    except (ValueError, TypeError):
        val_str = str(val)
        if "-" in val_str:
            parts = val_str.split("-")
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                pass
        return 0.0


# ─── Full Privacy Report ──────────────────────────────────────────────────────

def build_privacy_utility_report(
    original_df: pd.DataFrame,
    k_anon_df: Optional[pd.DataFrame] = None,
    l_div_df: Optional[pd.DataFrame] = None,
    dp_df: Optional[pd.DataFrame] = None,
    k_ncp: float = 0.0,
    l_ncp: float = 0.0,
    epsilon: float = 0.3,
    numerical_cols: Optional[list] = None,
) -> dict:
    """
    Tổng hợp toàn bộ privacy-utility report.
    """
    if numerical_cols is None:
        numerical_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    
    report = {
        "original": {
            "num_records": len(original_df),
            "accuracy": measure_classification_accuracy(original_df)["accuracy"],
        }
    }
    
    if k_anon_df is not None:
        k_acc = measure_classification_accuracy(k_anon_df)
        report["k_anonymity"] = {
            "ncp": round(k_ncp, 4),
            "ncp_interpretation": _interpret_ncp(k_ncp),
            "num_records": len(k_anon_df),
            "records_dropped": len(original_df) - len(k_anon_df),
            "accuracy": k_acc["accuracy"],
            "accuracy_loss_pct": round(
                (report["original"]["accuracy"] - k_acc["accuracy"]) * 100, 2
            ) if report["original"]["accuracy"] and k_acc["accuracy"] else None,
        }
    
    if l_div_df is not None:
        l_acc = measure_classification_accuracy(l_div_df)
        report["l_diversity"] = {
            "ncp": round(l_ncp, 4),
            "ncp_interpretation": _interpret_ncp(l_ncp),
            "num_records": len(l_div_df),
            "records_dropped": len(original_df) - len(l_div_df),
            "accuracy": l_acc["accuracy"],
            "accuracy_loss_pct": round(
                (report["original"]["accuracy"] - l_acc["accuracy"]) * 100, 2
            ) if report["original"]["accuracy"] and l_acc["accuracy"] else None,
        }
    
    if dp_df is not None:
        dp_error = measure_dp_query_error(original_df, dp_df, numerical_cols, epsilon)
        dp_acc = measure_classification_accuracy(dp_df)
        report["differential_privacy"] = {
            "epsilon": epsilon,
            "privacy_level": _epsilon_to_level(epsilon),
            "query_error": dp_error,
            "accuracy": dp_acc["accuracy"],
            "accuracy_loss_pct": round(
                (report["original"]["accuracy"] - dp_acc["accuracy"]) * 100, 2
            ) if report["original"]["accuracy"] and dp_acc["accuracy"] else None,
        }
    
    return report
```

---

## Bước 3: Cập Nhật `routes.py` — Trả Về Privacy Report

```python
# Trong routes.py, sau khi có tất cả anonymized data:

from app.core.utility_evaluator import build_privacy_utility_report

# Sau khi đã upload tất cả files, load lại để evaluate:
# ... (download và load các DataFrame)
privacy_report = build_privacy_utility_report(
    original_df=clean_df,        # DataFrame sau cleaning
    k_anon_df=k_anon_df,        # DataFrame sau k-anonymity
    l_div_df=l_div_df,          # DataFrame sau l-diversity
    dp_df=dp_df,                 # DataFrame sau DP
    k_ncp=k_ncp,                # NCP từ Mondrian
    l_ncp=l_ncp,
    epsilon=0.3,
)

return {
    "message": "Upload completed",
    "landing_zone_paths": landing_zone_paths,
    "clean_zone_paths": clean_zone_paths,
    "privacy_report": privacy_report,   # ← THÊM VÀO ĐÂY
}
```

---

## Bước 4: Tạo Jupyter Notebook Visualization

**File:** `preprocess-svc/notebooks/privacy_utility_analysis.ipynb`

Nội dung notebook cần có:

```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Privacy-Utility Curve cho K-Anonymity
k_values = [5, 10, 25, 50, 100]
k_ncp_values = [...]   # Chạy Mondrian với từng k
k_accuracy_values = [...] # Accuracy với từng k

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(k_values, k_ncp_values, 'b-o', label='NCP (Information Loss)')
plt.plot(k_values, [1 - a for a in k_accuracy_values], 'r-o', label='Accuracy Loss')
plt.xlabel('k (K-Anonymity parameter)')
plt.ylabel('Loss')
plt.title('K-Anonymity: Privacy-Utility Tradeoff')
plt.legend()

# 2. Privacy-Utility Curve cho DP
epsilon_values = [0.01, 0.1, 0.3, 0.5, 1.0, 5.0]
dp_errors = [...]      # Query error với từng epsilon
dp_accuracies = [...]  # Accuracy với từng epsilon

plt.subplot(1, 2, 2)
plt.plot(epsilon_values, dp_errors, 'b-o', label='Query Error (%)')
plt.plot(epsilon_values, [1 - a for a in dp_accuracies], 'r-o', label='Accuracy Loss')
plt.xscale('log')
plt.xlabel('ε (Privacy Budget)')
plt.ylabel('Error/Loss')
plt.title('Differential Privacy: Privacy-Utility Tradeoff')
plt.legend()

plt.tight_layout()
plt.savefig('privacy_utility_tradeoff.png', dpi=150)
plt.show()
```

---

## Bước 5: Thêm `privacy_budget_accountant.py`

**File:** `preprocess-svc/app/core/privacy_budget_accountant.py`

```python
"""
Privacy Budget Accountant — Theo dõi tổng epsilon đã dùng.
Quan trọng để đảm bảo không vượt quá budget khi compose nhiều queries.
"""
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class QueryRecord:
    mechanism: str
    epsilon: float
    columns: List[str]
    description: str = ""


class PrivacyBudgetAccountant:
    """Sequential composition accountant for (ε,0)-DP."""
    
    def __init__(self, total_epsilon: float):
        if total_epsilon <= 0:
            raise ValueError("Total epsilon must be positive")
        self.total_epsilon = total_epsilon
        self.records: List[QueryRecord] = []
    
    @property
    def spent(self) -> float:
        return sum(r.epsilon for r in self.records)
    
    @property
    def remaining(self) -> float:
        return max(0.0, self.total_epsilon - self.spent)
    
    def consume(self, epsilon: float, mechanism: str, columns: list, description: str = ""):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.spent + epsilon > self.total_epsilon:
            raise RuntimeError(
                f"Privacy budget exceeded! "
                f"Trying to consume ε={epsilon}, "
                f"but only ε={self.remaining:.4f} remaining "
                f"(total={self.total_epsilon})"
            )
        self.records.append(QueryRecord(mechanism, epsilon, columns, description))
        logger.info(
            f"[PrivacyAccountant] {mechanism}: consumed ε={epsilon:.4f}, "
            f"total spent={self.spent:.4f}/{self.total_epsilon}"
        )
    
    def report(self) -> dict:
        return {
            "total_epsilon": self.total_epsilon,
            "spent_epsilon": round(self.spent, 6),
            "remaining_epsilon": round(self.remaining, 6),
            "usage_pct": round(self.spent / self.total_epsilon * 100, 1),
            "num_queries": len(self.records),
            "query_details": [
                {
                    "mechanism": r.mechanism,
                    "epsilon": r.epsilon,
                    "columns": r.columns,
                    "description": r.description,
                }
                for r in self.records
            ],
        }
```

---

## Kết Quả Mong Đợi Sau Khi Implement

Response từ `POST /upload` sẽ có dạng:

```json
{
  "message": "Upload completed",
  "privacy_report": {
    "original": {
      "num_records": 32561,
      "accuracy": 0.847
    },
    "k_anonymity": {
      "ncp": 0.231,
      "ncp_interpretation": "Good utility — moderate information loss",
      "num_records": 32411,
      "records_dropped": 150,
      "accuracy": 0.821,
      "accuracy_loss_pct": 3.07
    },
    "l_diversity": {
      "ncp": 0.318,
      "ncp_interpretation": "Good utility — moderate information loss",
      "num_records": 32201,
      "records_dropped": 360,
      "accuracy": 0.808,
      "accuracy_loss_pct": 4.60
    },
    "differential_privacy": {
      "epsilon": 0.3,
      "privacy_level": "HIGH",
      "query_error": {
        "avg_relative_error_pct": 4.23,
        "column_metrics": {
          "age": {"relative_error_pct": 3.1, ...},
          "education-num": {"relative_error_pct": 5.2, ...}
        }
      },
      "accuracy": 0.803,
      "accuracy_loss_pct": 5.19
    }
  }
}
```
