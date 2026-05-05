# BƯỚC 3 — Privacy Budget Accountant

> **File tạo mới:** `preprocess-svc/app/core/privacy_budget_accountant.py`
> **Mục tiêu:** Tracking tổng epsilon đã dùng theo Sequential Composition theorem

---

## 3.1 Tại Sao Cần Budget Accountant?

Hiện tại trong `dp_anonymization_adapter.py`, epsilon được chia đều cho các cột:

```python
epsilon_allocation={col: epsilon / len(numerical_cols) for col in numerical_cols}
```

Nếu có 5 cột và epsilon=0.3 thì mỗi cột dùng 0.06. Nhưng theo **Sequential Composition**:
- Tổng epsilon thực sự đã dùng = 0.06 × 5 = **0.3** ✓
- Nếu sau đó dùng thêm 1 query nữa với epsilon=0.1 → tổng = **0.4** (vượt budget!)

Budget Accountant ngăn điều này xảy ra.

---

## 3.2 Code `privacy_budget_accountant.py`

```python
"""
privacy_budget_accountant.py
Track epsilon usage theo Sequential Composition theorem.

Sequential Composition: Nếu dùng k mechanisms với epsilon_1..k,
tổng epsilon = sum(epsilon_1..k).

Dùng trong preprocess-svc để đảm bảo tổng epsilon không vượt budget.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


class PrivacyBudgetExhaustedError(Exception):
    """Raise khi epsilon spent vượt quá total budget."""
    pass


@dataclass
class QueryRecord:
    """Ghi lại một lần dùng epsilon."""
    mechanism:   str
    epsilon:     float
    delta:       float
    columns:     List[str]
    description: str = ""


class PrivacyBudgetAccountant:
    """
    Sequential Composition Privacy Budget Accountant.

    Theo dõi tổng epsilon đã dùng, raise lỗi nếu vượt budget.

    Attributes:
        total_epsilon: Tổng budget (float > 0)
        total_delta:   Tổng delta budget cho (eps, delta)-DP (default 0)

    Usage:
        accountant = PrivacyBudgetAccountant(total_epsilon=1.0)
        accountant.consume(epsilon=0.3, mechanism="Laplace", columns=["age"])
        accountant.consume(epsilon=0.3, mechanism="Laplace", columns=["education-num"])
        print(accountant.report())
    """

    def __init__(self, total_epsilon: float, total_delta: float = 0.0):
        if total_epsilon <= 0:
            raise ValueError(f"total_epsilon must be > 0, got {total_epsilon}")
        self.total_epsilon = total_epsilon
        self.total_delta   = total_delta
        self._records: List[QueryRecord] = []
        logger.info(
            f"[BudgetAccountant] Initialized: total_epsilon={total_epsilon}, "
            f"total_delta={total_delta}"
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def spent_epsilon(self) -> float:
        return sum(r.epsilon for r in self._records)

    @property
    def spent_delta(self) -> float:
        return sum(r.delta for r in self._records)

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0.0, self.total_delta - self.spent_delta)

    @property
    def usage_pct(self) -> float:
        return round(self.spent_epsilon / self.total_epsilon * 100, 1)

    # ── Core Method ──────────────────────────────────────────────────────────

    def consume(
        self,
        epsilon: float,
        mechanism: str,
        columns: Optional[List[str]] = None,
        delta: float = 0.0,
        description: str = "",
    ) -> None:
        """
        Đăng ký một lần dùng epsilon.

        Args:
            epsilon:     Epsilon của lần này
            mechanism:   Tên mechanism (vd: "Laplace", "Exponential")
            columns:     Cột bị ảnh hưởng
            delta:       Delta (nếu dùng Gaussian mechanism)
            description: Mô tả thêm

        Raises:
            ValueError:                 Nếu epsilon <= 0
            PrivacyBudgetExhaustedError: Nếu spent + epsilon > total
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")

        new_spent = self.spent_epsilon + epsilon
        if new_spent > self.total_epsilon + 1e-9:
            raise PrivacyBudgetExhaustedError(
                f"Budget exceeded! Trying to consume eps={epsilon:.4f}, "
                f"but remaining={self.remaining_epsilon:.4f} "
                f"(spent={self.spent_epsilon:.4f}, total={self.total_epsilon})"
            )

        new_spent_delta = self.spent_delta + delta
        if new_spent_delta > self.total_delta + 1e-12 and self.total_delta > 0:
            raise PrivacyBudgetExhaustedError(
                f"Delta budget exceeded! delta={delta}, remaining={self.remaining_delta}"
            )

        record = QueryRecord(
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            columns=columns or [],
            description=description,
        )
        self._records.append(record)

        logger.info(
            f"[BudgetAccountant] {mechanism} consumed eps={epsilon:.4f} | "
            f"total_spent={self.spent_epsilon:.4f}/{self.total_epsilon} "
            f"({self.usage_pct}%)"
        )

    # ── Reporting ────────────────────────────────────────────────────────────

    def report(self) -> dict:
        """Trả về full report về budget usage."""
        return {
            "total_epsilon":     self.total_epsilon,
            "total_delta":       self.total_delta,
            "spent_epsilon":     round(self.spent_epsilon, 6),
            "spent_delta":       round(self.spent_delta,   9),
            "remaining_epsilon": round(self.remaining_epsilon, 6),
            "remaining_delta":   round(self.remaining_delta,   9),
            "usage_pct":         self.usage_pct,
            "num_queries":       len(self._records),
            "queries": [
                {
                    "mechanism":   r.mechanism,
                    "epsilon":     r.epsilon,
                    "delta":       r.delta,
                    "columns":     r.columns,
                    "description": r.description,
                }
                for r in self._records
            ],
        }

    def reset(self) -> None:
        """Reset budget counter (dùng khi bắt đầu request mới)."""
        self._records.clear()
        logger.info("[BudgetAccountant] Reset")

    def __repr__(self) -> str:
        return (
            f"PrivacyBudgetAccountant("
            f"total_eps={self.total_epsilon}, "
            f"spent={self.spent_epsilon:.4f}, "
            f"remaining={self.remaining_epsilon:.4f})"
        )
```

---

## 3.3 Tích Hợp Vào `dp_anonymization_adapter.py`

Sửa `apply_dp_protection_and_upload` để dùng accountant:

```python
# dp_anonymization_adapter.py — sửa hàm _apply_dp_to_dataframe:
from app.core.privacy_budget_accountant import PrivacyBudgetAccountant

def _apply_dp_to_dataframe(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    integration = DPAnonymizationIntegration(epsilon=epsilon)

    numerical_cols = [
        col for col in df.select_dtypes(include=["float64", "int64"]).columns
        if col not in ["income"]
    ]

    sensitivities = {
        "age":           100.0,
        "education-num":  16.0,
        "capital-gain":   100_000.0,
        "capital-loss":   5_000.0,
        "hours-per-week": 168.0,
    }

    numerical_cols = [col for col in numerical_cols if col in df.columns]
    sensitivities  = {col: sensitivities.get(col, 1.0) for col in numerical_cols}

    # ── Budget Accountant ──────────────────────────────────────────────────
    accountant = PrivacyBudgetAccountant(total_epsilon=epsilon)
    eps_per_col = epsilon / len(numerical_cols) if numerical_cols else epsilon

    for col in numerical_cols:
        accountant.consume(
            epsilon=eps_per_col,
            mechanism="Laplace",
            columns=[col],
            description=f"Protect column {col}",
        )

    logger.info(f"Budget report: {accountant.report()}")
    # ──────────────────────────────────────────────────────────────────────

    try:
        df_protected = integration.apply_dp_to_numerical_columns(
            df=df,
            numerical_columns=numerical_cols,
            sensitivities=sensitivities,
            epsilon_allocation={col: eps_per_col for col in numerical_cols},
        )
        return df_protected
    except Exception as e:
        logger.error(f"Error applying DP: {e}")
        raise
```

---

## 3.4 Thêm Budget Report Vào API Response

Trong `routes.py`, truyền budget report vào response:

```python
# routes.py — trong block xử lý file:
from app.core.privacy_budget_accountant import PrivacyBudgetAccountant

TOTAL_EPSILON = 1.0   # Tổng budget cho toàn bộ pipeline

# Khởi tạo accountant cho mỗi request
accountant = PrivacyBudgetAccountant(total_epsilon=TOTAL_EPSILON)

# Mỗi bước dùng epsilon đăng ký vào accountant:
# K-anonymity không dùng epsilon (structural privacy)
# L-diversity không dùng epsilon

# DP dùng EPSILON = 0.3
accountant.consume(
    epsilon=EPSILON,
    mechanism="Laplace",
    columns=["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
    description="DP post-processing after anonymization",
)

budget_report = accountant.report()

# Đưa vào response:
return {
    "message":            "Upload completed",
    "landing_zone_paths": landing_zone_paths,
    "clean_zone_paths":   clean_zone_paths,
    "privacy_report":     privacy_report,
    "budget_report":      budget_report,   # ← THÊM
}
```

---

## 3.5 Test Budget Accountant

```python
# test_budget.py
from app.core.privacy_budget_accountant import (
    PrivacyBudgetAccountant,
    PrivacyBudgetExhaustedError,
)

# Test normal flow
acc = PrivacyBudgetAccountant(total_epsilon=1.0)
acc.consume(0.3, "Laplace", ["age"], "Age protection")
acc.consume(0.3, "Laplace", ["education-num"], "Edu protection")
acc.consume(0.3, "Laplace", ["capital-gain"], "Capital protection")
print(acc.report())
# spent_epsilon = 0.9, remaining = 0.1

# Test vượt budget
try:
    acc.consume(0.5, "Laplace", ["hours-per-week"])   # Sẽ raise!
except PrivacyBudgetExhaustedError as e:
    print(f"Caught: {e}")   # Budget exceeded!
```

---

## 3.6 Commit Bước 3

```bash
git add preprocess-svc/app/core/privacy_budget_accountant.py
git add preprocess-svc/app/core/dp_anonymization_adapter.py
git add preprocess-svc/app/api/routes.py
git commit -m "feat: add PrivacyBudgetAccountant, integrate into DP adapter and routes"
```

---

## 3.7 Checklist Bước 3

- [ ] Tạo `privacy_budget_accountant.py`
- [ ] Tích hợp accountant vào `dp_anonymization_adapter.py`
- [ ] Thêm `budget_report` vào API response
- [ ] Test: normal flow + vượt budget
- [ ] Commit

**Tiếp theo:** [Bước 4 — Encryption Service](./08_step04_encryption_svc.md)
