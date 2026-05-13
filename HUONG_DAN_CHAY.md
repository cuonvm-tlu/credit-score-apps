# 🚀 Hướng dẫn chạy toàn bộ hệ thống từ số 0

## Tổng quan kiến trúc

```
[Bạn upload file]
       │
       ▼
 preprocess-svc (FastAPI :8000)
  ├─ lưu raw file → MinIO (landing-zone)
  ├─ clean data   → MinIO (clean-zone)
  └─ publish event → Kafka [data-cleaned-topic]
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    anonymize-svc (FastAPI :8001)    analytic-service (script)
    ├─ k/l/DP anonymization          └─ train RF model (từ clean data)
    ├─ lưu → MinIO (anonymize-zone)      → MinIO (model-zone)
    └─ publish → [data-anonymized-topic]
                         │
                         ▼
              analytic-service (script)
              └─ train RF model (từ anonymized data)
                  → MinIO (model-zone/anon/)
```

**Bạn cần mở 5 terminal riêng biệt.**

---

## Bước 1 — Khởi động MinIO (lưu trữ file)

> [!IMPORTANT]
> MinIO là "ổ cứng" dùng chung. **Phải chạy trước tất cả.**

```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=admin" \
  -e "MINIO_ROOT_PASSWORD=password" \
  -v /tmp/minio-data:/data \
  minio/minio server /data --console-address ":9001"
```

Kiểm tra MinIO đã lên chưa: mở trình duyệt → http://127.0.0.1:9001
- User: `admin` / Password: `password`

> [!TIP]
> Lần sau chỉ cần `docker start minio` (không cần chạy lại lệnh dài).

---

## Bước 2 — Khởi động Kafka (message broker)

> [!IMPORTANT]
> Kafka là "đường ống" truyền event giữa các service. **Phải chạy trước khi chạy service nào.**

```bash
# Dùng docker-compose có sẵn trong repo
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/anonymize-svc
docker compose up -d
```

Kiểm tra Kafka đã lên chưa:
```bash
docker ps | grep kafka
# Phải thấy: local-kafka ... Up
```

> [!TIP]
> Lần sau chỉ cần `docker compose up -d` trong thư mục `anonymize-svc`.

---

## Bước 3 — Cài đặt và chạy `preprocess-svc`

**Mở Terminal 1:**

```bash
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/preprocess-svc

# Tạo môi trường ảo (chỉ làm 1 lần)
python3 -m venv venv

# Kích hoạt venv
source venv/bin/activate

# Cài thư viện (chỉ làm 1 lần)
pip install -r requirements.txt

# Chạy service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

✅ Thành công khi thấy:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Bước 4 — Cài đặt và chạy `anonymize-svc`

**Mở Terminal 2:**

```bash
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/anonymize-svc

# Tạo môi trường ảo (chỉ làm 1 lần)
python3 -m venv venv

# Kích hoạt venv
source venv/bin/activate

# Cài thư viện (chỉ làm 1 lần)
pip install -r requirements.txt

# Chạy service
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

✅ Thành công khi thấy:
```
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO  [app.core.kafka_anonymize_worker] Kafka anonymization consumer started
```

---

## Bước 5 — Cài đặt và chạy `analytic-service`

**Mở Terminal 3:**

```bash
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/analytic-service

# Tạo môi trường ảo (chỉ làm 1 lần)
python3 -m venv venv

# Kích hoạt venv
source venv/bin/activate

# Cài thư viện (chỉ làm 1 lần)
pip install -r requirements.txt

# Chạy service
python -m app.main
```

✅ Thành công khi thấy:
```
INFO [app.main] Starting Analytic Service...
INFO [app.core.kafka_anonymize_consumer] Anonymize-training Kafka consumer started
INFO [app.core.kafka_consumer] Subscribed to topic: data-cleaned-topic
INFO [app.main] Both Kafka consumers are running. Press Ctrl+C to stop.
```

---

## Bước 6 — Upload file để test toàn bộ luồng

**Mở Terminal 4** (cần file `adult.data` hoặc `adult.test`):

```bash
# Upload file lên preprocess-svc
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@/đường/dẫn/đến/adult.data"
```

**Kết quả mong đợi trong các terminal:**

| Terminal | Sẽ thấy log gì |
|---|---|
| **preprocess-svc** | `Downloaded → cleaned → Kafka event published` |
| **anonymize-svc** | `DATA_CLEANING_COMPLETED received → anonymizing → published DATA_ANNONIMIZING_COMPLETED` |
| **analytic-service** | `Training model từ clean data…` rồi `Training model từ anonymized data…` |

---

## Kiểm tra kết quả trên MinIO

Mở http://127.0.0.1:9001 → Buckets:

| Bucket | Chứa gì |
|---|---|
| `landing-zone` | File gốc bạn upload |
| `clean-zone` | File đã clean (`*_clean.parquet`) |
| `anonymize-zone` | File đã ẩn danh (`*_anon_k5.parquet`, `*_anon_l2.parquet`, v.v.) |
| `model-zone` | Model đã train (`.joblib`) |

Cụ thể trong `model-zone`:
```
model-zone/
└── {version_id}/
    ├── rf_model.joblib          ← train từ clean data
    └── anon/
        ├── adult_anon_k5.joblib
        ├── adult_anon_k10.joblib
        ├── adult_anon_k20.joblib
        ├── adult_anon_k50.joblib
        ├── adult_anon_l2.joblib
        └── adult_dp_*.joblib    ← train từ anonymized data
```

---

## Lần sau chỉ cần

```bash
# Terminal 0: Khởi động infrastructure
docker start minio
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/anonymize-svc && docker compose up -d

# Terminal 1: preprocess-svc
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/preprocess-svc
source venv/bin/activate && uvicorn app.main:app --port 8000 --reload

# Terminal 2: anonymize-svc
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/anonymize-svc
source venv/bin/activate && uvicorn app.main:app --port 8001 --reload

# Terminal 3: analytic-service
cd /home/tienpv16/Desktop/Workspace/credit-score-apps/analytic-service
source venv/bin/activate && python -m app.main
```

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách fix |
|---|---|---|
| `Connection refused 9092` | Kafka chưa chạy | `docker compose up -d` trong thư mục `anonymize-svc` |
| `Connection refused 9000` | MinIO chưa chạy | `docker start minio` |
| `ModuleNotFoundError` | Chưa cài thư viện | `pip install -r requirements.txt` |
| `Target column 'income' not found` | File anonymized thiếu cột | Bình thường, analytic-svc sẽ skip file đó |
| Port 8000 đã bị chiếm | Service khác đang dùng | Đổi port: `--port 8002` |
