# Kiến Trúc Kafka Event-Driven: Tại Sao & Làm Gì Tiếp?

> **Bối cảnh:** Leader muốn: `Preprocess → Send Event → Consume → Mã hoá dữ liệu`  
> **Screenshot đính kèm:** Kafka consumer đang nhận event `DATA_CLEANING_COMPLETED` từ topic `data-cleaned-topic`

---

## 1. Hiện Tại Đang Có Gì?

### Event đang được bắn (kafka_producer.py):

```json
{
  "event_type": "DATA_CLEANING_COMPLETED",
  "status": "success",
  "version_id": "2026:04:08:14:59",
  "clean_file_paths": [
    "clean-zone/2026:04:08:14:59/adult_clean.parquet",
    "clean-zone/2026:04:08:14:59/adult_anon_k10.parquet",
    "clean-zone/2026:04:08:14:59/adult_anon_l2.parquet",
    "clean-zone/2026:04:08:14:59/adult_dp_e0_30.parquet"
  ]
}
```

### Flow hiện tại (một service làm tất cả):

```
Client (curl /upload)
    │
    ▼
[preprocess-svc]
    ├── Upload raw → MinIO landing-zone
    ├── Spark clean + generalize
    ├── K-Anonymity (Mondrian)
    ├── L-Diversity (Mondrian)
    ├── Differential Privacy (Laplace)
    └── Publish event → Kafka (data-cleaned-topic)
                                 │
                           [Không ai lắng nghe! ← vấn đề hiện tại]
```

**Vấn đề:** Event bắn ra nhưng **không có consumer nào** xử lý tiếp. Dữ liệu sau khi anonymize xong thì... dừng lại ở MinIO.

---

## 2. Tại Sao Cần Kafka? (Lý Do Kiến Trúc)

### 2.1 Kafka giải quyết vấn đề gì?

Trong hệ thống **Privacy-Preserving Big Data**, dữ liệu đi qua nhiều giai đoạn xử lý độc lập:

| Giai đoạn | Trách nhiệm | Service |
|---|---|---|
| Ingest | Upload file thô | preprocess-svc (đã có) |
| Clean & Anonymize | Spark + Mondrian + DP | preprocess-svc (đã có) |
| **Encrypt** | Mã hoá tầng lưu trữ | **encryption-svc (chưa có)** |
| **Serve** | Cung cấp API cho model ML | **serving-svc (chưa có)** |
| **Audit** | Log ai truy cập, bao nhiêu epsilon | **audit-svc (chưa có)** |

**Kafka** là "bus" kết nối các service này mà **không cần chúng biết nhau**:
- `preprocess-svc` chỉ cần bắn event, không cần quan tâm ai nhận
- `encryption-svc` chỉ cần lắng nghe topic, không cần biết ai gửi
- Thêm service mới → subscribe topic là xong, không sửa code cũ

### 2.2 Tại sao không gọi thẳng HTTP?

```
❌ Không dùng Kafka (tight coupling):
preprocess-svc → HTTP POST → encryption-svc
preprocess-svc → HTTP POST → audit-svc
preprocess-svc → HTTP POST → serving-svc
# Nếu encryption-svc chết → preprocess-svc lỗi!

✅ Dùng Kafka (loose coupling):
preprocess-svc → Kafka → encryption-svc (tự xử lý khi sống lại)
                       → audit-svc
                       → serving-svc
# Mỗi service độc lập, lỗi không lan sang nhau
```

### 2.3 Lợi ích cụ thể với Privacy-Preserving:

1. **Audit trail tự động**: Mọi event đều có timestamp, version_id → dễ audit "dữ liệu nào đã được xử lý"
2. **Retry tự động**: Nếu encryption lỗi → Kafka giữ message, consumer retry sau
3. **Parallel processing**: Nhiều consumer cùng lúc (encrypt + audit + serve) không block nhau
4. **Scale độc lập**: Encryption service cần nhiều CPU → scale riêng, không ảnh hưởng preprocess

---

## 3. Kiến Trúc Leader Muốn Đạt Được

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│  Client                                                         │
│    │                                                            │
│    ▼                                                            │
│  [preprocess-svc]  ──── Kafka ────────────────────────────────► │
│   • Upload raw             │                                    │
│   • Spark clean            │         ┌──────────────────────┐  │
│   • K-Anonymity            ├────────►│  encryption-svc      │  │
│   • L-Diversity            │         │  • Đọc file từ MinIO │  │
│   • DP noise               │         │  • AES-256 encrypt   │  │
│   • Publish event          │         │  • Upload encrypted  │  │
│                            │         └──────────────────────┘  │
│                            │                                    │
│                            │         ┌──────────────────────┐  │
│                            ├────────►│  audit-svc           │  │
│                            │         │  • Ghi log event     │  │
│                            │         │  • Track epsilon used│  │
│                            │         │  • Compliance record │  │
│                            │         └──────────────────────┘  │
│                            │                                    │
│                            │         ┌──────────────────────┐  │
│                            └────────►│  serving-svc (future)│  │
│                                      │  • Expose anonymized │  │
│                                      │  • API cho ML model  │  │
│                                      └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Bước Tiếp Theo: Implement `encryption-svc`

### 4.1 Vai trò của encryption-svc

Sau khi dữ liệu đã được:
- ✅ Cleaned (Spark)
- ✅ K-anonymized (Mondrian)
- ✅ L-diversified (Mondrian)
- ✅ DP-protected (Laplace noise)

→ **Vẫn cần mã hoá tầng lưu trữ** vì:
- File parquet trong MinIO vẫn readable nếu ai đó có quyền truy cập MinIO
- Encryption đảm bảo **data-at-rest security**
- Phù hợp với tiêu chuẩn: GDPR, PCI-DSS, HIPAA

### 4.2 Code mẫu `encryption-svc`

**Cấu trúc thư mục:**

```
encryption-svc/
├── app/
│   ├── main.py              # Entry point
│   ├── kafka_consumer.py    # Lắng nghe data-cleaned-topic
│   ├── encryptor.py         # Logic mã hoá AES-256-GCM
│   └── minio_client.py      # Download/upload MinIO
├── requirements.txt
└── Dockerfile
```

**`kafka_consumer.py`** — Lắng nghe event từ preprocess-svc:

```python
import json
import logging
from confluent_kafka import Consumer, KafkaError

from app.encryptor import encrypt_parquet_file
from app.minio_client import download_file, upload_file

logger = logging.getLogger(__name__)

KAFKA_CONFIG = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'encryption-service-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': False,   # Manual commit — chỉ commit sau khi encrypt thành công
}

TOPIC = 'data-cleaned-topic'


def start_consumer():
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([TOPIC])
    logger.info(f"[encryption-svc] Listening on topic: {TOPIC}")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    logger.error(f"Kafka error: {msg.error()}")
                continue

            event = json.loads(msg.value().decode('utf-8'))
            logger.info(f"Received: {event['event_type']} | version: {event['version_id']}")

            if event['event_type'] == 'DATA_CLEANING_COMPLETED':
                _handle_cleaning_completed(event)
                consumer.commit(msg)   # Commit sau khi xử lý thành công

    except KeyboardInterrupt:
        logger.info("Shutting down consumer...")
    finally:
        consumer.close()


def _handle_cleaning_completed(event: dict):
    """Download file anonymized → encrypt → upload lên encrypted-zone."""
    version_id = event['version_id']
    file_paths = event['clean_file_paths']

    for file_path in file_paths:
        # Chỉ encrypt file anonymized, không encrypt file clean thô
        if any(tag in file_path for tag in ['_anon_', '_dp_', '_ldiv_']):
            bucket, key = file_path.split('/', 1)
            local_path = download_file(bucket, key)
            encrypted_local = encrypt_parquet_file(local_path)
            encrypted_key = key.replace('.parquet', '_encrypted.enc')
            upload_file(encrypted_local, bucket='encrypted-zone', key=encrypted_key)
            logger.info(f"Encrypted: {file_path} → encrypted-zone/{encrypted_key}")

    logger.info(f"[{version_id}] Encryption complete")
```

**`encryptor.py`** — AES-256-GCM:

```python
import os
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Key 32 bytes = AES-256. Trong production: lấy từ KMS (AWS KMS / HashiCorp Vault)
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY', os.urandom(32).hex())


def encrypt_parquet_file(input_path: str) -> str:
    """
    Mã hoá file bằng AES-256-GCM (authenticated encryption).
    
    Tại sao AES-256-GCM?
    - AES-256: Chuẩn mã hoá mạnh nhất (NIST approved)
    - GCM mode: Phát hiện file bị tamper (chỉnh sửa trái phép)
    - Phù hợp GDPR data-at-rest requirements
    
    Format output: [12 bytes nonce][ciphertext+auth_tag]
    """
    key = bytes.fromhex(ENCRYPTION_KEY)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)   # Random, KHÔNG BAO GIỜ tái sử dụng nonce với cùng key!

    with open(input_path, 'rb') as f:
        plaintext = f.read()

    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    output_path = Path(input_path).with_suffix('.enc')
    with open(output_path, 'wb') as f:
        f.write(nonce + ciphertext)   # Lưu nonce kèm ciphertext để giải mã sau

    return str(output_path)


def decrypt_parquet_file(encrypted_path: str) -> str:
    """Giải mã file. Dùng khi serving-svc cần đọc data cho ML model."""
    key = bytes.fromhex(ENCRYPTION_KEY)
    aesgcm = AESGCM(key)

    with open(encrypted_path, 'rb') as f:
        data = f.read()

    nonce = data[:12]
    ciphertext = data[12:]
    plaintext = aesgcm.decrypt(nonce, ciphertext, associated_data=None)

    output_path = Path(encrypted_path).with_suffix('.parquet')
    with open(output_path, 'wb') as f:
        f.write(plaintext)

    return str(output_path)
```

---

## 5. Cập Nhật Event Schema — Preprocess-Svc Nên Bắn Thêm Info

Hiện tại event chỉ có `clean_file_paths`. Nên bổ sung để downstream consumer biết context:

```python
# kafka_producer.py — cập nhật message:
message = {
    "event_type": "DATA_CLEANING_COMPLETED",
    "status": "success",
    "version_id": version_folder,
    "clean_file_paths": clean_file_paths,

    # THÊM MỚI — downstream cần biết:
    "privacy_metadata": {
        "k_anonymity": {"k": 10, "ncp": 0.23},
        "l_diversity": {"l": 2, "ncp": 0.31},
        "differential_privacy": {"epsilon": 0.3, "mechanism": "Laplace"},
    },
    "dataset": "adult_census_income",
    "schema_version": "1.1",
}
```

---

## 6. Toàn Bộ Flow Mới (Đầy Đủ)

```
Step 1: Client upload adult.data
        │
        ▼
Step 2: preprocess-svc
        ├── MinIO landing-zone: adult.data
        ├── Spark clean → adult_clean.parquet
        ├── K-Anonymity → adult_anon_k10.parquet
        ├── L-Diversity → adult_anon_l2.parquet
        ├── DP protect  → adult_dp_e0_30.parquet
        └── Kafka: DATA_CLEANING_COMPLETED {version, file_paths, privacy_metadata}
                │
        ┌───────┴────────┐
        ▼                ▼
Step 3: encryption-svc   audit-svc  (song song, cùng subscribe 1 topic)
        ├── AES-256-GCM  ├── Ghi log compliance
        ├── encrypted-zone ├── Track epsilon budget
        └── ENCRYPTION_COMPLETED event └── DB record
```

---

## 7. Thứ Tự Làm Tiếp

### Ưu tiên cao:

```
[1] Tạo encryption-svc/ (Kafka consumer + AES-256-GCM encryptor)
[2] Cập nhật docker-compose.yml thêm encryption-svc
[3] Thêm MinIO bucket: encrypted-zone
[4] Cập nhật event schema trong kafka_producer.py (thêm privacy_metadata)
[5] Test end-to-end: upload → clean → Kafka → encrypt → verify
```

### Ưu tiên trung bình:

```
[6] Fix bug preprocess-svc (xem 02_missing_requirements.md)
[7] Implement utility_evaluator.py (xem 03_implementation_plan.md)
[8] Thêm privacy_report vào API response
```

### Ưu tiên thấp:

```
[9]  audit-svc — ghi log compliance
[10] serving-svc — decrypt + serve cho ML model
[11] Jupyter notebook — visualization privacy-utility tradeoff
```

---

## 8. Tóm Tắt: Kafka Có Tác Dụng Gì?

| Không có Kafka | Có Kafka |
|---|---|
| preprocess-svc gọi encryption trực tiếp → tightly coupled | preprocess-svc bắn event → encryption tự xử lý |
| Nếu encryption down → upload lỗi ngay | Nếu encryption down → retry tự động khi sống lại |
| Thêm service mới phải sửa preprocess-svc | Thêm service mới chỉ cần subscribe topic |
| Không có audit trail | Mọi event được track tự nhiên (Kafka log = immutable) |
| Không scale được | Scale từng service độc lập |

**Kafka không phải tính năng privacy, nhưng là foundation để:**
1. Tách biệt các bước xử lý (clean / anonymize / encrypt / serve)
2. Đảm bảo reliability (retry, at-least-once delivery)
3. Audit trail tự nhiên
4. Scale to Big Data thực sự

---

## References

- [Confluent Kafka Python Client](https://github.com/confluentinc/confluent-kafka-python)
- [cryptography library — AESGCM](https://cryptography.io/en/latest/hazmat/primitives/aead/#cryptography.hazmat.primitives.ciphers.aead.AESGCM)
- [NIST AES-256](https://csrc.nist.gov/publications/detail/fips/197/final)
- [GDPR Data-at-Rest Encryption](https://gdpr.eu/encryption/)
