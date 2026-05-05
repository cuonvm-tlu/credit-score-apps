# Những Gì Còn Thiếu & Kế Hoạch Triển Khai

> **Dự án:** `credit-score-apps / preprocess-svc`  
> **Chuẩn so sánh:** Privacy-Preserving Big Data Processing Requirements

---

## 🔴 BUG NGHIÊM TRỌNG CẦN SỬA NGAY

### Bug #1: `breakpoint()` trong production code
**File:** `app/core/dp_mechanisms/dp_anonymization_integration.py` — dòng 71

```python
# Dòng hiện tại - SẼ BLOCK TOÀN BỘ SERVER!
def apply_dp_to_numerical_columns(self, df, ...):
    df_noisy = df.copy()
    breakpoint()   ← XÓA NGAY!
    ...
```

**Fix:**
```python
def apply_dp_to_numerical_columns(self, df, ...):
    df_noisy = df.copy()
    # (xóa dòng breakpoint())
    if sensitivities is None:
        ...
```

### Bug #2: Missing import trong `routes.py`
**File:** `app/api/routes.py` — dòng 100 gọi `apply_dp_protection_and_upload` nhưng **không có import**!

```python
# Cần thêm vào đầu file routes.py:
from app.core.dp_anonymization_adapter import apply_dp_protection_and_upload
```

### Bug #3: Income column bị comment out mapping
**File:** `app/core/spark_cleaner.py` — S4 bị comment out, income vẫn là string

```python
# Bỏ comment dòng này trong spark_cleaner.py (S4):
df = df.withColumn("income",
    F.when(F.col("income") == "<=50K", F.lit(0))
     .when(F.col("income") == ">50K",  F.lit(1))
     .otherwise(None).cast(IntegerType()))
```

---

## 🟡 THIẾU TÍNH NĂNG THEO YÊU CẦU

### Thiếu #1: Privacy-Utility Tradeoff Evaluation (yêu cầu bắt buộc!)

**Đây là yêu cầu quan trọng nhất bị thiếu hoàn toàn:**

> "Evaluate the trade-off between privacy levels and analytical accuracy"

Hiện tại **không có bất kỳ metric nào** đo lường:
- Information Loss (NCP — Normalized Certainty Penalty)
- Classification accuracy trước/sau anonymization
- Query error từ DP noise

**Cần implement:**

#### a) NCP (Normalized Certainty Penalty) cho K-Anonymity & L-Diversity
```python
# Mondrian đã trả về NCP nhưng không được lưu/trả về API!
anon_df, ncp, rtime = anonymize_adult_dataframe(df, k=k)
# ncp và rtime bị bỏ qua!
```

#### b) DP Query Error Measurement
```python
# Đo lường sai số trung bình của DP noise
def measure_dp_utility(original_df, dp_df, epsilon):
    metrics = {}
    for col in numerical_cols:
        true_mean = original_df[col].mean()
        noisy_mean = dp_df[col].mean()
        metrics[col] = {
            "relative_error": abs(true_mean - noisy_mean) / true_mean,
            "epsilon": epsilon
        }
    return metrics
```

#### c) Classification Accuracy Comparison
```python
# Chạy Logistic Regression/Decision Tree trên:
# - Original data
# - K-anonymized data (k=5, 10, 25, 50)
# - L-diverse data (l=2, 3, 5)
# - DP data (ε=0.1, 0.3, 0.5, 1.0, 5.0)
# So sánh accuracy → privacy-utility curve
```

---

### Thiếu #2: Gaussian Mechanism (ε,δ)-DP

Chỉ có Laplace (pure DP). Gaussian mechanism thường dùng cho ML applications:

```python
# Cần thêm: app/core/dp_mechanisms/gaussian_mechanism.py
class GaussianNoiseMechanism:
    """(ε, δ)-Differential Privacy via Gaussian noise"""
    def __init__(self, epsilon: float, delta: float):
        # Noise scale: σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε
        self.sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    
    def apply(self, query_result, sensitivity):
        noise = np.random.normal(0, self.sigma * sensitivity, size=query_result.shape)
        return query_result + noise
```

---

### Thiếu #3: Spark-based Anonymization (Big Data requirement)

K-anonymity và L-diversity hiện dùng **pandas** — không scale được với dataset lớn!

```python
# Hiện tại: Spark → Pandas → Mondrian (single-node)
spark.read.parquet(path).toPandas()  # ← không scale!

# Cần: Distributed Mondrian on Spark
# Gợi ý: implement partition-based k-anonymity bằng Spark SQL
```

**Giải pháp đơn giản hơn:** Dùng **ARX Framework** hoặc implement Mondrian partition trên Spark DataFrame.

---

### Thiếu #4: T-Closeness

L-diversity có thể bị tấn công skewness/similarity. T-Closeness yêu cầu phân phối SA trong mỗi equivalence class phải gần với phân phối toàn cục:

```python
# Cần thêm: app/core/t_closeness_adapter.py
def check_t_closeness(df, qi_columns, sa_column, t_threshold=0.2):
    """Verify t-closeness: EMD between group SA dist and overall SA dist <= t"""
    overall_dist = df[sa_column].value_counts(normalize=True)
    for group_key, group_df in df.groupby(qi_columns):
        group_dist = group_df[sa_column].value_counts(normalize=True)
        emd = compute_emd(overall_dist, group_dist)
        if emd > t_threshold:
            return False, group_key
    return True, None
```

---

### Thiếu #5: DP Composition Tracking (Privacy Budget Accounting)

Hiện tại epsilon được dùng nhiều lần mà không tracking tổng budget:

```python
# Trong routes.py, chạy:
# - k-anonymity (không dùng epsilon)
# - l-diversity (không dùng epsilon)  
# - DP với epsilon=0.3 (apply to numerical cols riêng lẻ)
# Nhưng không có global budget tracking!

# Cần implement Privacy Budget Accountant:
class PrivacyBudgetAccountant:
    def __init__(self, total_epsilon: float, total_delta: float = 0.0):
        self.total_epsilon = total_epsilon
        self.spent_epsilon = 0.0
    
    def consume(self, epsilon: float, mechanism: str):
        self.spent_epsilon += epsilon
        if self.spent_epsilon > self.total_epsilon:
            raise PrivacyBudgetExhausted(f"Budget exceeded: {self.spent_epsilon} > {self.total_epsilon}")
        logger.info(f"[{mechanism}] Consumed ε={epsilon}, Total: {self.spent_epsilon}/{self.total_epsilon}")
```

---

### Thiếu #6: API Endpoint Trả Về Privacy Metrics

API hiện tại chỉ trả về file paths, không trả về privacy metrics:

```json
// Hiện tại:
{
  "message": "Upload completed",
  "landing_zone_paths": [...],
  "clean_zone_paths": [...]
}

// Cần trả về:
{
  "message": "Upload completed",
  "privacy_report": {
    "k_anonymity": {"k": 10, "ncp": 0.23, "records_dropped": 150},
    "l_diversity": {"l": 2, "ncp": 0.31, "records_dropped": 200},
    "differential_privacy": {
      "epsilon": 0.3,
      "mechanisms": ["laplace"],
      "columns_protected": ["age", "education-num", "capital-gain"],
      "avg_relative_error": 0.045
    },
    "utility_metrics": {
      "original_records": 32561,
      "anonymized_records": 32411,
      "data_loss_pct": 0.46,
      "income_accuracy_before": 0.847,
      "income_accuracy_after_kanon": 0.821,
      "income_accuracy_after_dp": 0.803
    }
  }
}
```

---

### Thiếu #7: Monitoring & Observability

Không có:
- Prometheus metrics
- Privacy budget dashboard
- Audit log (ai truy cập, epsilon đã dùng bao nhiêu)

---

### Thiếu #8: End-to-End Testing

File `tests.py` được đề cập trong IMPLEMENTATION_SUMMARY nhưng **không tồn tại** trong `dp_mechanisms/` folder!

---

## 🟢 KẾ HOẠCH TRIỂN KHAI (Ưu tiên)

### Phase 1: Sửa Bug Ngay (1-2 ngày)

```
[P0] Xóa breakpoint() trong dp_anonymization_integration.py
[P0] Thêm import apply_dp_protection_and_upload vào routes.py  
[P0] Bỏ comment income mapping trong spark_cleaner.py
[P1] Bỏ comment age_group generalization trong spark_cleaner.py
```

### Phase 2: Privacy-Utility Evaluation (3-5 ngày)

```
[1] Lưu NCP từ Mondrian vào response và MinIO metadata
[2] Implement DP query error measurement (relative error per column)
[3] Implement classification accuracy benchmark:
    - Train baseline model trên clean data
    - Evaluate trên k-anonymized (k=5,10,25,50)
    - Evaluate trên DP data (ε=0.1,0.3,0.5,1.0,5.0)
[4] Tạo API endpoint GET /privacy-report trả về metrics
[5] Thêm privacy_report vào response của POST /upload
```

**File mới cần tạo:**
- `app/core/utility_evaluator.py` — tính NCP, query error, classification accuracy
- `app/core/privacy_budget_accountant.py` — tracking epsilon
- `notebooks/privacy_utility_analysis.ipynb` — visualization tradeoff curves

### Phase 3: Gaussian Mechanism & T-Closeness (2-3 ngày)

```
[1] app/core/dp_mechanisms/gaussian_mechanism.py
[2] app/core/t_closeness_adapter.py
[3] Tích hợp vào routes.py
[4] Thêm vào privacy_report
```

### Phase 4: Spark-scale Anonymization (5-7 ngày)

```
[1] Nghiên cứu distributed Mondrian hoặc ARX trên Spark
[2] Implement partition-based anonymization với Spark SQL
[3] Benchmark performance: Pandas vs Spark trên dataset lớn
```

### Phase 5: Testing & Documentation (2-3 ngày)

```
[1] Viết unit tests cho tất cả mechanisms
[2] Viết integration test cho full pipeline
[3] Cập nhật README với privacy-utility results
[4] Thêm Jupyter notebook demo
```

---

## 📊 Checklist Yêu Cầu Đề Bài

| Yêu cầu | Hiện trạng | Cần làm |
|---|---|---|
| DP (ε-privacy) statistical queries | ✅ Có code (có bug) | Fix bug, thêm tracking |
| K-Anonymity | ✅ Working | Thêm metrics output |
| L-Diversity | ✅ Working | Thêm metrics output |
| Privacy-Utility Tradeoff Evaluation | ❌ Chưa có | **Implement từ đầu** |
| Big Data (Spark) processing | ⚠️ Chỉ bước cleaning | Scale anonymization |
| Credit Scoring application theme | ✅ Phù hợp | — |
| Adult Census dataset | ✅ Đúng | — |
| Gaussian mechanism | ❌ Chưa có | Thêm vào |
| T-Closeness | ❌ Chưa có | Thêm vào (optional) |
| Tests | ❌ Missing | Viết tests |
| Monitoring | ❌ Chưa có | Thêm vào (optional) |

---

## 🔗 Tài Liệu Tham Khảo

1. Dwork & Roth (2014) — *The Algorithmic Foundations of Differential Privacy*
2. [Google Differential Privacy Library](https://github.com/google/differential-privacy)
3. [OpenDP — Differential Privacy Library](https://opendp.org/)
4. [ARX Data Anonymization Tool](https://arx.deidentifier.org/)
5. [Basic_Mondrian Reference](https://github.com/QiyuanZhao/k-anonymity)
