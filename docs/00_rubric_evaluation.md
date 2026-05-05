# Đánh Giá Barem Điểm — Project `credit-score-apps`

> **Thời điểm đánh giá:** 2026-05-05  
> **Nhánh:** `main` (hiện tại) + chiến lược `feature/full` (7 bước)  
> **Rubric:** General Grading Rubric Guidelines (5 tiêu chí)

---

## TÓM TẮT NHANH

| # | Tiêu chí | Trọng số | Hiện tại (main) | Sau feature/full | Rủi ro |
|---|---|---|---|---|---|
| 01 | System Analysis & Design | 20% | ⚠️ 60-65% | ✅ 80-85% | Trung bình |
| 02 | Technical Implementation | 30% | ⚠️ 55-60% | ✅ 80-85% | Cao |
| 03 | Optimization & Research Depth | 20% | ❌ 30-35% | ⚠️ 60-65% | Cao |
| 04 | Evaluation & Testing | 15% | ❌ 20-25% | ⚠️ 65-70% | Cao |
| 05 | Report & Presentation | 15% | ⚠️ 50% | ✅ 70-75% | Trung bình |

**Điểm ước tính hiện tại (main):** ~47–52/100  
**Điểm ước tính sau feature/full:** ~73–78/100

---

## 01. System Analysis & Design — 20%

### Yêu cầu Excellence:
- Clear problem statement
- Robust, scalable architecture
- Proper selection of Big Data ecosystem components

---

### ✅ ĐÃ ĐI ĐÚNG HƯỚNG

**Dataset chọn đúng:**
- Adult Census Income phù hợp với chủ đề Inter-bank Credit Scoring
- Có đủ QI columns (age, sex, race, occupation) và Sensitive Attribute (income)
- Đây là benchmark dataset được dùng trong nhiều nghiên cứu privacy

**Big Data stack đúng:**
- **Spark** cho data processing (đúng — scalable hơn Pandas)
- **Kafka** cho event-driven messaging (đúng — async pipeline)
- **MinIO** (S3-compatible object storage) — đúng hướng NoSQL/object store
- **Parquet** format — đúng cho big data analytics

**Problem statement rõ:**
- Anonymize sensitive demographic data trước khi dùng cho credit scoring
- Pipeline: Raw → Clean → K-Anonymity → L-Diversity → DP → Encrypt

---

### ⚠️ CÒN THIẾU / YẾU

**1. Không có sơ đồ kiến trúc chính thức**
- Rubric yêu cầu "Robust, scalable architecture" — cần có diagram hình ảnh
- Hiện chỉ có ASCII art trong file .md, không phải architecture diagram chuẩn

**2. MinIO không phải "NoSQL" thực sự**
- Rubric đề cập "Kafka, NoSQL" — MinIO là object storage, không phải NoSQL DB
- Nên thêm 1 NoSQL component thực sự:
  - **MongoDB/Cassandra** để lưu privacy audit logs
  - Hoặc **Elasticsearch** để query anonymized data
  - Hoặc **Redis** để cache epsilon budget state

**3. Scalability chưa được chứng minh**
- Spark chỉ dùng local mode (không có cluster config)
- Mondrian k-anonymity vẫn chạy trên Pandas (single node), không scale
- Chưa có partition strategy cho large dataset

**4. Problem statement chưa được viết dạng academic**
- Chưa có formal problem definition: P(re-identification) < 1/k
- Chưa có threat model: ai là adversary, attack nào đang defend

### 🔧 ĐỀ XUẤT BỔ SUNG (cho feature/full)

```
- Vẽ architecture diagram chính thức bằng draw.io hoặc Mermaid
- Thêm MongoDB để lưu audit logs (= NoSQL component rõ ràng)
- Viết formal problem statement trong README: "Given dataset D with QI 
  attributes Q and sensitive attribute S, find transformation T(D) such that 
  T(D) satisfies k-anonymity with minimum information loss"
- Thêm config Spark cluster mode (ít nhất local[*] thay vì local)
```

---

## 02. Technical Implementation — 30% ← QUAN TRỌNG NHẤT

### Yêu cầu Excellence:
- Stable pipeline/application
- Proficiency in tools (Spark, Kafka, NoSQL)
- Clean, well-documented, version-controlled code

---

### ✅ ĐÃ ĐI ĐÚNG HƯỚNG

**Spark:**
- `spark_cleaner.py` dùng Spark DataFrame API đúng cách
- Schema định nghĩa rõ ràng (StructType)
- Xử lý continent_code generalization bằng native Spark (không UDF)
- `coalesce(1)` để ghi 1 file output

**Kafka:**
- Producer gửi event `DATA_CLEANING_COMPLETED` sau preprocessing
- Event schema có `event_type`, `status`, `version_id`, `clean_file_paths`
- Screenshot cho thấy đã consume được từ terminal

**Code structure:**
- Chia rõ `api/` và `core/` (separation of concerns)
- Dùng adapters pattern (basic_mondrian_adapter, l_diversity_adapter)
- FastAPI với proper routing

**Version control:**
- Git history rõ ràng với commit messages có ý nghĩa
- Nhánh ltlong → merge PR vào main đúng workflow

---

### ⚠️ VẤN ĐỀ NGHIÊM TRỌNG (cần sửa trong feature/full)

**Bug #1: `breakpoint()` trong production code**
- `dp_anonymization_integration.py` dòng 71
- **Impact:** Server sẽ hang khi gọi DP endpoint — KHÔNG thể demo!
- **Fix:** Bước 1 trong 7 steps

**Bug #2: Missing import trong routes.py**
- `apply_dp_protection_and_upload` được gọi nhưng không được import
- **Impact:** API sẽ crash với NameError khi upload file
- **Fix:** Bước 1 trong 7 steps

**Bug #3: Income mapping bị comment out**
- Spark cleaner không map income → 0/1
- **Impact:** income vẫn là string "<=50K", accuracy evaluation sai
- **Fix:** Bước 1 trong 7 steps

**Thiếu: Kafka Consumer**
- Event bắn ra nhưng không ai consume
- **Impact:** Pipeline dừng ở preprocess, không có encryption
- **Fix:** Bước 4 (encryption-svc)

**Thiếu: Spark Proficiency trong Anonymization**
- K-Anonymity và L-Diversity convert về Pandas để chạy Mondrian
- Đây là điểm yếu lớn: nói dùng Spark nhưng anonymization không scale
- **Partial fix:** Ghi rõ limitation trong documentation

**Code documentation:**
- Nhiều hàm thiếu docstring hoàn chỉnh
- Không có type hints nhất quán

---

### 🔧 ĐỀ XUẤT BỔ SUNG (cho feature/full)

```
- Sửa 3 bugs ngay (Bước 1)
- Tạo encryption-svc consumer (Bước 4)
- Thêm Dockerfile cho preprocess-svc
- Thêm NoSQL: MongoDB để lưu privacy audit logs
  db.privacy_logs.insert_one({
      "version_id": version_folder,
      "k": 10, "l": 2, "epsilon": 0.3,
      "ncp": 0.18, "accuracy_loss": 1.97,
      "timestamp": datetime.now()
  })
- Viết docstrings chuẩn Google style cho tất cả public functions
```

---

## 03. Optimization & Research Depth — 20%

### Yêu cầu Excellence:
- Evidence of optimization (skew handling, latency reduction)
- Comparative analysis of different configurations/methods

---

### ❌ YẾU NHẤT — CẦN CẢI THIỆN NHIỀU NHẤT

**Hiện tại gần như KHÔNG CÓ:**
- Không có bất kỳ benchmark nào
- Không so sánh k=5 vs k=10 vs k=25
- Không so sánh epsilon=0.1 vs 0.3 vs 1.0
- Không so sánh Laplace vs Gaussian
- Không có skew handling (data skew là vấn đề lớn trong Mondrian)

---

### ⚠️ CHIẾN LƯỢC 7 BƯỚC CÓ ĐI ĐÚNG HƯỚNG KHÔNG?

**Bước 2 (utility_evaluator):** ✅ Đúng hướng nhưng chưa đủ
- Đo được accuracy loss và DP error
- Nhưng chưa có **comparative analysis** với nhiều giá trị k, l, ε

**Bước 5 (Gaussian):** ✅ Đúng hướng
- Thêm Gaussian mechanism cho phép so sánh Laplace vs Gaussian
- Nhưng chưa có benchmark cụ thể

**THIẾU HOÀN TOÀN trong 7 bước:**
- **Skew handling:** Mondrian có thể tạo equivalence classes không đều
  → Cần đo distribution của class sizes
- **Latency benchmark:** Thời gian xử lý mỗi step (spark_clean, k-anon, l-div, dp)
- **Throughput:** Bao nhiêu records/second ở mỗi step
- **Comparative table:** k vs NCP vs Accuracy cross-tabulation

---

### 🔧 CẦN BỔ SUNG THÊM VÀO CHIẾN LƯỢC

**Thêm Bước 8: Comparative Analysis (ngoài 7 bước hiện tại)**

```python
# preprocess-svc/app/core/benchmark.py

import time
import pandas as pd

def run_privacy_benchmark(df: pd.DataFrame) -> dict:
    """
    Chạy benchmark với nhiều cấu hình khác nhau.
    Kết quả dùng để vẽ privacy-utility tradeoff curves.
    """
    results = []

    # K-Anonymity benchmark: k = 5, 10, 20, 50
    for k in [5, 10, 20, 50]:
        t0 = time.time()
        anon_df, ncp, _ = anonymize_adult_dataframe(df, k=k)
        elapsed = time.time() - t0
        acc = measure_classification_accuracy(anon_df)["accuracy"]
        results.append({
            "method": "k_anonymity", "param": k,
            "ncp": ncp, "accuracy": acc,
            "records_kept": len(anon_df),
            "latency_sec": elapsed,
        })

    # DP benchmark: epsilon = 0.01, 0.1, 0.3, 0.5, 1.0, 5.0
    for eps in [0.01, 0.1, 0.3, 0.5, 1.0, 5.0]:
        t0 = time.time()
        dp_df = apply_laplace_dp(df, epsilon=eps)
        elapsed = time.time() - t0
        err = measure_dp_query_error(df, dp_df, eps)
        acc = measure_classification_accuracy(dp_df)["accuracy"]
        results.append({
            "method": "differential_privacy", "param": eps,
            "avg_error_pct": err["avg_relative_error_pct"],
            "accuracy": acc,
            "latency_sec": elapsed,
        })

    # Laplace vs Gaussian comparison (same epsilon=0.5)
    for mechanism in ["laplace", "gaussian"]:
        t0 = time.time()
        dp_df = apply_dp(df, epsilon=0.5, mechanism=mechanism)
        elapsed = time.time() - t0
        acc = measure_classification_accuracy(dp_df)["accuracy"]
        results.append({
            "method": f"dp_{mechanism}", "param": 0.5,
            "accuracy": acc, "latency_sec": elapsed,
        })

    return pd.DataFrame(results).to_dict(orient="records")
```

**Endpoint mới `/benchmark`:**
```python
@router.post("/benchmark")
async def run_benchmark(version_id: str) -> dict:
    """Chạy comparative analysis và trả về kết quả."""
    ...
```

**Cần có bảng so sánh như này trong báo cáo:**

| Method | Param | NCP | Accuracy | Latency(s) | Throughput(rec/s) |
|---|---|---|---|---|---|
| K-Anonymity | k=5 | 0.08 | 84.2% | 12.3 | 2,647 |
| K-Anonymity | k=10 | 0.18 | 83.0% | 14.1 | 2,309 |
| K-Anonymity | k=20 | 0.31 | 81.5% | 18.7 | 1,741 |
| K-Anonymity | k=50 | 0.52 | 78.3% | 31.2 | 1,043 |
| DP Laplace | ε=0.1 | — | 79.8% | 2.1 | 15,505 |
| DP Laplace | ε=0.3 | — | 80.7% | 2.0 | 16,281 |
| DP Laplace | ε=1.0 | — | 83.5% | 1.9 | 17,138 |
| DP Gaussian | ε=0.5 | — | 82.1% | 2.2 | 14,800 |

---

## 04. Evaluation & Testing — 15%

### Yêu cầu Excellence:
- Rigorous test cases
- Quantitative performance evaluation (Latency, Throughput, Accuracy)
- Supported by charts

---

### ⚠️ CHIẾN LƯỢC 7 BƯỚC: ĐI ĐÚNG HƯỚNG 60%

**Bước 6 (Unit Tests) — ĐÚN HƯỚNG:**
- 35+ test cases cho DP mechanisms, utility evaluator, budget accountant
- Dùng pytest + fixtures → đúng chuẩn
- Test cả edge cases (insufficient data, budget exceeded, invalid epsilon)

**NHƯNG CÒN THIẾU:**

**1. Không có Performance Tests (Latency/Throughput)**
- Rubric yêu cầu: "Quantitative performance evaluation (Latency, Throughput)"
- 7 bước chỉ có unit tests (correctness), không có performance tests

```python
# Cần thêm: tests/test_performance.py
import time
import pytest

@pytest.mark.slow
def test_k_anonymity_latency(adult_sample_df):
    """K-Anonymity phải xử lý 32k records trong < 60 giây."""
    t0 = time.time()
    result, ncp, _ = anonymize_adult_dataframe(adult_sample_df, k=10)
    elapsed = time.time() - t0
    assert elapsed < 60.0, f"K-Anonymity too slow: {elapsed:.1f}s"
    throughput = len(adult_sample_df) / elapsed
    print(f"Throughput: {throughput:.0f} records/sec")

@pytest.mark.slow
def test_dp_laplace_throughput(adult_sample_df):
    """DP Laplace phải xử lý > 10k records/sec."""
    mechanism = LaplaceNoiseMechanism(epsilon=0.3)
    t0 = time.time()
    noisy = mechanism.apply(adult_sample_df["age"].values, sensitivity=100.0)
    elapsed = time.time() - t0
    throughput = len(adult_sample_df) / elapsed
    assert throughput > 10_000
```

**2. Không có Charts trong Tests**
- Rubric: "supported by charts"
- Cần tạo plots từ test results và lưu vào docs/charts/

**3. Không có Integration Tests**
- Chỉ unit test từng component riêng lẻ
- Cần test full pipeline: upload → clean → anonymize → dp → kafka event

**4. Thiếu Statistical Validation**
- Test Laplace noise có đúng phân phối không? (KS test)
- Test k-anonymity có đạt k=10 thật không? (verify mọi equivalence class size >= 10)

```python
# Cần thêm:
def test_k_anonymity_guarantee(k_anon_df, k=10):
    """Verify THỰC SỰ mọi equivalence class có size >= k."""
    # Nhóm theo QI columns
    qi_cols = ["age", "workclass", "education_num", "marital_status",
               "occupation", "race", "sex", "native_country"]
    for qi_cols_present in [c for c in qi_cols if c in k_anon_df.columns]:
        group_sizes = k_anon_df.groupby(qi_cols_present).size()
        assert (group_sizes >= k).all(), f"k-anonymity violated! Min size: {group_sizes.min()}"

def test_laplace_noise_distribution(adult_sample_df):
    """Verify Laplace noise có mean ≈ 0 và variance ≈ 2*(sensitivity/epsilon)^2."""
    from scipy import stats
    m = LaplaceNoiseMechanism(epsilon=1.0)
    noise = [m.apply(0.0, sensitivity=1.0) for _ in range(10000)]
    # KS test against Laplace distribution
    stat, p_value = stats.kstest(noise, "laplace", args=(0, 1.0))
    assert p_value > 0.05, f"Noise not Laplace distributed! p={p_value:.4f}"
```

---

### 🔧 CẦN BỔ SUNG VÀO BƯỚC 6

```
Thêm vào chiến lược:
- tests/test_performance.py   (latency + throughput benchmarks)
- tests/test_statistical.py   (KS test cho noise distribution)
- tests/test_integration.py   (full pipeline test)
- tests/test_guarantees.py    (verify k-anonymity/l-diversity đạt đúng k, l)
- docs/charts/                (lưu plots từ notebook)
```

---

## 05. Report & Presentation — 15%

### Yêu cầu Excellence:
- Academic writing style
- Confident delivery
- Ability to defend technical choices during Q&A

---

### ⚠️ ĐÁNH GIÁ

**Tài liệu hiện có:**
- README.md — functional, không academic
- `dp_mechanisms/README.md` và `IMPLEMENTATION_SUMMARY.md` — tốt
- 11 file docs/ mới tạo — technical, step-by-step

**Thiếu:**
- Không có báo cáo dạng academic (Introduction, Related Work, Methodology, Results, Conclusion)
- Không có references/citations học thuật
- Không có abstract

**Gợi ý câu hỏi Q&A mà giáo viên có thể hỏi và cần chuẩn bị:**

```
Q1: Tại sao chọn Mondrian thay vì OLA (Optimal Lattice Anonymization)?
A: Mondrian: O(n log n), OLA: O(2^|QI| × n). Với 8 QI columns và 32k records,
   OLA không feasible. Mondrian là heuristic tốt nhất cho dataset size này.

Q2: ε=0.3 là "đủ" privacy không?
A: Theo NIST guidelines (NIST Privacy Framework), ε < 1 được coi là "high privacy".
   ε=0.3 nằm trong HIGH privacy zone (0.1 < ε ≤ 0.5). Tuy nhiên không có
   "đủ" tuyệt đối — tùy threat model.

Q3: Tại sao không dùng thư viện DP có sẵn như Google DP hoặc OpenDP?
A: Dùng custom implementation để hiểu underlying math (sensitivity, composition).
   Trong production, recommend Google DP library (C++ backend, audited).

Q4: K-anonymity có thể bị tấn công gì?
A: Homogeneity attack (tất cả SA trong 1 class giống nhau) → L-diversity giải quyết.
   Background knowledge attack → T-closeness giải quyết.
   Kết hợp cả 3 cho defense-in-depth.

Q5: Kafka có thực sự cần không? HTTP đơn giản hơn?
A: Kafka giải quyết: (1) async decoupling, (2) retry khi consumer chết,
   (3) at-least-once delivery guarantee, (4) natural audit log,
   (5) fan-out (nhiều consumer cùng subscribe). HTTP synchronous coupling
   không scale và không fault-tolerant.

Q6: Tại sao AES-256-GCM thay vì AES-256-CBC?
A: GCM = Authenticated Encryption with Associated Data (AEAD).
   CBC: encryption only, không detect tampering.
   GCM: encryption + authentication tag → phát hiện file bị sửa đổi.
   NIST SP 800-38D recommends GCM cho data-at-rest.

Q7: Spark chạy local mode có đúng không khi nói "Big Data"?
A: local[*] sử dụng tất cả CPU cores — phù hợp cho single-node deployment.
   Để truly distributed, cần Spark cluster (YARN/Kubernetes).
   Project demo với local mode, production sẽ deploy trên cluster.
```

---

## TỔNG KẾT: CẦN BỔ SUNG GÌ VÀO 7 BƯỚC

### Bổ sung vào Bước 1 (Bug Fixes):
- Thêm Dockerfile cho preprocess-svc
- Thêm `local[*]` config cho Spark session

### Bổ sung vào Bước 2 (Utility Evaluator):
- Thêm latency measurement cho mỗi step

### Bổ sung vào Bước 4 (Encryption-svc):
- Thêm MongoDB để lưu privacy audit logs (= NoSQL component rõ ràng)

### Bổ sung Bước 8 (THÊM MỚI — Benchmark):
- `benchmark.py`: chạy k={5,10,20,50} và ε={0.01,0.1,0.3,1.0,5.0}
- Endpoint `POST /benchmark`
- Comparative table trong docs
- Charts lưu vào `docs/charts/`

### Bổ sung Bước 9 (THÊM MỚI — Báo Cáo Academic):
- `REPORT.md` dạng academic (2000+ từ)
- Sections: Abstract, Introduction, Related Work, System Design,
  Implementation, Evaluation, Conclusion, References
- Cite papers: Dwork 2006, Sweeney 2002 (k-anonymity), Machanavajjhala 2007 (l-diversity)

---

## ĐIỂM DỰ ĐOÁN SAU KHI HOÀN THÀNH ĐỦ

| Tiêu chí | Trọng số | Điểm/10 | Điểm có trọng số |
|---|---|---|---|
| System Analysis & Design | 20% | 8.0 | 1.60 |
| Technical Implementation | 30% | 8.5 | 2.55 |
| Optimization & Research Depth | 20% | 6.5 | 1.30 |
| Evaluation & Testing | 15% | 7.0 | 1.05 |
| Report & Presentation | 15% | 7.5 | 1.125 |
| **TỔNG** | **100%** | — | **7.625/10** |

> **Lưu ý:** Nếu không thêm Bước 8 (Benchmark) và Bước 9 (Academic Report),
> điểm "Optimization & Research Depth" và "Report & Presentation" sẽ thấp hơn
> đáng kể (~5.0–5.5), kéo tổng điểm xuống ~6.5–7.0/10.
