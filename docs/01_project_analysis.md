# Phân Tích Project: `credit-score-apps` — Nhánh `main` & `ltlong`

> **Mục tiêu yêu cầu:** Privacy-Preserving Big Data Processing — Bảo vệ thông tin cá nhân nhạy cảm trong khi duy trì tiện ích phân tích dữ liệu quy mô lớn.

---

## 1. Dataset Đang Dùng

### ✅ Adult Census Income Dataset (UCI ML Repository)

Project dùng **Adult Census Income** — đúng với dataset gợi ý trong đề bài.

| Thuộc tính | Chi tiết |
|---|---|
| **Tên file gốc** | `adult.data`, `adult.test` (hoặc `.csv`) |
| **Số cột gốc** | 15 cột |
| **Cột bị loại** | `fnlwgt` (bị drop khi cleaning) |
| **Cột giữ lại** | `age`, `workclass`, `education`, `education-num`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `capital-gain`, `capital-loss`, `hours-per-week`, `native-country`, `income` |
| **Nhãn (target)** | `income`: binary `0` (≤50K) / `1` (>50K) |
| **Cột Quasi-Identifier (QI)** | `age`, `workclass`, `education-num`, `marital-status`, `occupation`, `race`, `sex`, `native-country` |
| **Cột Sensitive Attribute (SA)** | `income` |

**Lý do phù hợp với chủ đề Inter-bank Credit Scoring:**
- `income` là mục tiêu phân tích tín dụng
- Các thuộc tính nhân khẩu học (`race`, `sex`, `age`) là thông tin nhạy cảm cần bảo vệ
- Dataset có kích thước vừa đủ để thử nghiệm kỹ thuật anonymization

---

## 2. Kiến Trúc Hệ Thống Đã Xây Dựng

```
credit-score-apps/
└── preprocess-svc/          # Microservice duy nhất
    ├── app/
    │   ├── main.py          # FastAPI application
    │   ├── api/
    │   │   └── routes.py    # API endpoints
    │   └── core/
    │       ├── cleaner.py                   # Pandas-based cleaner
    │       ├── spark_cleaner.py             # Spark-based cleaner + generalization
    │       ├── kafka_producer.py            # Kafka event publisher
    │       ├── minio_client.py              # MinIO storage client
    │       ├── spark_session.py             # PySpark session manager
    │       ├── anonymize_k_anonymity.py     # K-anonymity pipeline
    │       ├── anonymize_l_diversity.py     # L-diversity pipeline
    │       ├── basic_mondrian_adapter.py    # Mondrian k-anonymity adapter
    │       ├── l_diversity_adapter.py       # Mondrian l-diversity adapter
    │       ├── dp_anonymization_adapter.py  # DP pipeline adapter
    │       ├── basic_mondrian/              # Mondrian algorithm core
    │       ├── mondrian_l_diversity/        # Mondrian L-diversity core
    │       ├── anonymization_shared/        # Shared: GenTree, NumRange
    │       └── dp_mechanisms/              # Differential Privacy module
    │           ├── laplace_mechanism.py
    │           ├── exponential_mechanism.py
    │           ├── above_threshold.py
    │           ├── dp_anonymization_integration.py
    │           ├── dp_utils.py
    │           └── config.py
    ├── docker-compose.yml   # Kafka setup
    └── requirements.txt
```

**Infrastructure Stack:**
- **API:** FastAPI + Uvicorn
- **Storage:** MinIO (S3-compatible) — landing-zone + clean-zone buckets
- **Messaging:** Apache Kafka (event-driven)
- **Processing:** PySpark (big data) + Pandas (anonymization)
- **Format:** Parquet (columnar storage)

---

## 3. Những Gì Đã Làm (Theo Commit History)

### Nhánh `ltlong` (feature branch)

| Commit | Nội dung |
|---|---|
| `f835232` | Initial commit — cấu trúc cơ bản |
| `bbd3dfd` | Kafka event sau preprocessing |
| `adc9297` | README |
| `d360479` | **K-anonymity + L-diversity** (Mondrian algorithm) |
| `55ded3f` | **Spark cleaner** — thay Pandas bằng Spark |
| `6cad9c5` | `spark_clean_and_upload` — full pipeline |
| `9af3d5f` | Thêm thư viện vào requirements |
| `c7762b9` | **DP mechanism** — Laplace, Exponential, Above Threshold |

### Nhánh `main` (production)

| Commit | Nội dung |
|---|---|
| `6f32b1e` | Merge từ ltlong |
| `c5d3270` | Refactor: bỏ bước pre-process thừa |

---

## 4. Chi Tiết Từng Kỹ Thuật Đã Implement

### 4.1 Data Cleaning + Generalization (Spark)
- ✅ Đọc file `adult.data`/`adult.test` bằng Spark với schema định sẵn
- ✅ Trim whitespace, xử lý "?" → null
- ✅ Drop `fnlwgt`
- ✅ **Generalization:** Thêm cột `continent_code` từ `native-country` (dùng `pycountry` + fuzzy matching)
- ⚠️ `age_group` generalization bị **comment out** (S6 trong `spark_cleaner.py`)
- ⚠️ Income mapping sang binary bị **comment out** (S4) — income vẫn là string `<=50K`/`>50K`

### 4.2 K-Anonymity (Basic Mondrian)
- ✅ Implement thuật toán **Basic Mondrian** cho k-anonymity
- ✅ Support cả QI numeric (age, education-num) và categorical (workclass, occupation, race, sex, native-country)
- ✅ Build generalization tree tự động từ data (không cần file external)
- ✅ Upload kết quả lên MinIO: `adult_anon_k10.parquet`
- ✅ k mặc định = 10

### 4.3 L-Diversity (Mondrian L-Diversity)
- ✅ Implement **Mondrian L-Diversity** — đảm bảo mỗi equivalence class có ít nhất `l` giá trị SA khác nhau
- ✅ Cùng QI/SA setup như K-Anonymity
- ✅ Upload: `adult_anon_l2.parquet`
- ✅ l mặc định = 2

### 4.4 Differential Privacy
- ✅ **Laplace Mechanism** — (ε,0)-DP cho numerical columns
- ✅ **Exponential Mechanism** — DP selection cho categorical
- ✅ **Above Threshold** — (ε,δ)-DP cho threshold queries
- ✅ **DPAnonymizationIntegration** — class tổng hợp
- ✅ **Config + Privacy Levels:** VERY_HIGH (ε=0.01) → VERY_LOW (ε=5.0)
- ✅ Upload: `adult_dp_e0_30.parquet`
- ⚠️ **Bug nghiêm trọng:** `dp_anonymization_integration.py` line 71 có `breakpoint()` — sẽ block production!
- ⚠️ DP apply trên **clean data trực tiếp**, không apply sau k-anonymity/l-diversity

### 4.5 Pipeline Flow (routes.py)
```
Upload file
  → MinIO landing-zone
  → Spark Clean + Generalize (spark_clean_and_upload)
  → K-Anonymity Mondrian → MinIO (adult_anon_k10.parquet)
  → L-Diversity Mondrian → MinIO (adult_anon_l2.parquet)
  → Differential Privacy → MinIO (adult_dp_e0_30.parquet)
  → Kafka event (data-cleaned-topic)
```
- ⚠️ DP được apply song song với K-Anon/L-Div, không phải **sau** chúng (chưa đúng workflow Defense-in-Depth)
- ⚠️ Import `apply_dp_protection_and_upload` trong `routes.py` bị **thiếu** (không có dòng import!)

---

## 5. Những Gì Còn Thiếu

Xem file: [missing_requirements.md](./missing_requirements.md)

---

## 6. Tóm Tắt Đánh Giá

| Yêu cầu | Trạng thái | Chất lượng |
|---|---|---|
| Differential Privacy (ε-privacy) | ✅ Có code | ⚠️ Có bug, chưa tích hợp đúng |
| K-Anonymity | ✅ Đã implement | ✅ Tốt |
| L-Diversity | ✅ Đã implement | ✅ Tốt |
| Privacy-Utility Tradeoff Evaluation | ❌ Chưa có | — |
| Big Data Processing (Spark) | ✅ Có Spark | ⚠️ Chỉ ở bước cleaning, chưa scale anonymization |
| Kafka Event-Driven | ✅ Đã có | ✅ OK |
| Documentation | ✅ Có README + DP docs | ⚠️ Thiếu metrics/evaluation |
| Tests | ❌ Chưa chạy được | — |
