# BƯỚC 1 — Tạo Nhánh & Sửa Bug Ngay

> **Nhánh:** `feature/full` (checkout từ `main`)  
> **Mục tiêu:** Chuẩn bị môi trường và sửa 3 bug nghiêm trọng trước khi implement feature mới

---

## 1.1 Tạo Nhánh

```bash
cd /home/tienpv16/Desktop/Workspace/credit-score-apps
git checkout main
git pull origin main
git checkout -b feature/full
git push -u origin feature/full
```

Xác nhận:
```bash
git branch       # Phải thấy * feature/full
git log --oneline -3
```

---

## 1.2 Cấu Trúc Thư Mục Hiện Tại

```
credit-score-apps/
├── docs/                          ← Tài liệu (vừa tạo)
└── preprocess-svc/
    ├── README.md
    ├── docker-compose.yml         ← Chỉ có Kafka
    ├── requirements.txt
    └── app/
        ├── __init__.py
        ├── main.py
        ├── api/
        │   ├── __init__.py
        │   └── routes.py          ← BUG: missing import
        └── core/
            ├── __init__.py
            ├── cleaner.py
            ├── spark_cleaner.py   ← BUG: income mapping bị comment
            ├── kafka_producer.py
            ├── minio_client.py
            ├── spark_session.py
            ├── anonymize_k_anonymity.py
            ├── anonymize_l_diversity.py
            ├── basic_mondrian_adapter.py
            ├── l_diversity_adapter.py
            ├── dp_anonymization_adapter.py
            ├── basic_mondrian/
            ├── mondrian_l_diversity/
            ├── anonymization_shared/
            └── dp_mechanisms/
                ├── __init__.py
                ├── dp_anonymization_integration.py  ← BUG: breakpoint()
                ├── laplace_mechanism.py
                ├── exponential_mechanism.py
                ├── above_threshold.py
                ├── dp_utils.py
                └── config.py
```

---

## 1.3 Bug #1 — Xóa `breakpoint()` trong DP Integration

**File:** `preprocess-svc/app/core/dp_mechanisms/dp_anonymization_integration.py`

**Tìm dòng 71 — xóa `breakpoint()`:**

```python
# TRƯỚC (dòng 70-74):
    def apply_dp_to_numerical_columns(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        sensitivities: Optional[dict] = None,
        epsilon_allocation: Optional[dict] = None,
    ) -> pd.DataFrame:
        df_noisy = df.copy()
        breakpoint()          # ← XÓA DÒNG NÀY
        if sensitivities is None:
            sensitivities = {col: 1.0 for col in numerical_columns}

# SAU:
    def apply_dp_to_numerical_columns(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        sensitivities: Optional[dict] = None,
        epsilon_allocation: Optional[dict] = None,
    ) -> pd.DataFrame:
        df_noisy = df.copy()
        if sensitivities is None:
            sensitivities = {col: 1.0 for col in numerical_columns}
```

Kiểm tra:
```bash
grep -n "breakpoint" preprocess-svc/app/core/dp_mechanisms/dp_anonymization_integration.py
# Không được in ra dòng nào
```

---

## 1.4 Bug #2 — Thêm Missing Import trong `routes.py`

**File:** `preprocess-svc/app/api/routes.py`

Thêm import ở dòng 13 (sau các import hiện tại):

```python
# Thêm dòng này vào routes.py (sau dòng import kafka_producer):
from app.core.dp_anonymization_adapter import apply_dp_protection_and_upload
```

File sau khi sửa (phần đầu):

```python
import io
import os
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.anonymize_k_anonymity import anonymize_cleaned_adult_k_anonymity_and_upload
from app.core.anonymize_l_diversity import anonymize_cleaned_adult_l_diversity_and_upload
from app.core.cleaner import clean_and_upload
from app.core.spark_cleaner import spark_clean_and_upload
from app.core.kafka_producer import send_cleaning_success_event
from app.core.minio_client import ensure_bucket, get_minio_client
from app.core.dp_anonymization_adapter import apply_dp_protection_and_upload  # ← THÊM

router = APIRouter()
```

---

## 1.5 Bug #3 — Bỏ Comment Income Mapping trong `spark_cleaner.py`

**File:** `preprocess-svc/app/core/spark_cleaner.py`

Tìm block S4 (khoảng dòng 170-175) và bỏ comment:

```python
# TRƯỚC:
    # S4: normalize income (strip trailing dot, map 0/1)
    # df = df.withColumn("income",
    #     F.when(F.col("income") == "<=50K", F.lit(0))
    #      .when(F.col("income") == ">50K",  F.lit(1))
    #      .otherwise(None).cast(IntegerType()))

# SAU — thêm import IntegerType và bỏ comment:
    from pyspark.sql.types import IntegerType
    # S4: normalize income → binary 0/1
    df = df.withColumn("income",
        F.when(F.col("income") == "<=50K", F.lit(0))
         .when(F.col("income") == ">50K",  F.lit(1))
         .otherwise(None).cast(IntegerType()))
```

> **Lưu ý:** Import `IntegerType` đã có trong block `_spark_clean()` ở dòng 128, chỉ cần bỏ comment phần S4.

---

## 1.6 Bonus — Bỏ Comment `age_group` Generalization

Tìm block S6 và bỏ comment:

```python
# TRƯỚC:
    # S6: age_group generalization (native Spark, không cần UDF)
    # df = df.withColumn("age_group",
    #     F.concat(
    #         (F.floor(F.col("age") / 10) * 10).cast("int").cast("string"),
    #         F.lit("-"),
    #         ((F.floor(F.col("age") / 10) * 10) + 10).cast("int").cast("string")
    #     )
    # )

# SAU:
    # S6: age_group generalization
    df = df.withColumn("age_group",
        F.concat(
            (F.floor(F.col("age") / 10) * 10).cast("int").cast("string"),
            F.lit("-"),
            ((F.floor(F.col("age") / 10) * 10) + 10).cast("int").cast("string")
        )
    )
```

---

## 1.7 Commit Bug Fixes

```bash
cd /home/tienpv16/Desktop/Workspace/credit-score-apps
git add preprocess-svc/app/core/dp_mechanisms/dp_anonymization_integration.py
git add preprocess-svc/app/api/routes.py
git add preprocess-svc/app/core/spark_cleaner.py
git commit -m "fix: remove breakpoint, add missing import, enable income mapping and age_group"
```

---

## 1.8 Kiểm Tra Sau Khi Fix

```bash
cd preprocess-svc
pip install -r requirements.txt   # Nếu chưa install

# Chạy thử API
uvicorn app.main:app --reload --port 8000

# Test upload (cần MinIO + Kafka đang chạy)
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@/path/to/adult.data"

# Expected response (không còn bị block bởi breakpoint):
# {"message": "Upload completed", "landing_zone_paths": [...], "clean_zone_paths": [...]}
```

---

## 1.9 Checklist Bước 1

- [ ] Tạo nhánh `feature/full` từ `main`
- [ ] Xóa `breakpoint()` trong `dp_anonymization_integration.py`
- [ ] Thêm import `apply_dp_protection_and_upload` vào `routes.py`
- [ ] Bỏ comment income mapping trong `spark_cleaner.py`
- [ ] Bỏ comment age_group generalization
- [ ] Commit với message rõ ràng
- [ ] Test API chạy không lỗi

**Tiếp theo:** [Bước 2 — Implement Privacy-Utility Evaluator](./06_step02_utility_evaluator.md)
