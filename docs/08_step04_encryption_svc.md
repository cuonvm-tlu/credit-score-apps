# BƯỚC 4 — Tạo `encryption-svc` (Kafka Consumer + AES-256-GCM)

> **Thư mục mới:** `encryption-svc/`
> **Mục tiêu:** Consume event Kafka → download file anonymized → mã hoá AES-256-GCM → upload encrypted-zone

---

## 4.1 Tạo Cấu Trúc Thư Mục

```bash
mkdir -p encryption-svc/app
touch encryption-svc/app/__init__.py
touch encryption-svc/app/main.py
touch encryption-svc/app/kafka_consumer.py
touch encryption-svc/app/encryptor.py
touch encryption-svc/app/minio_client.py
touch encryption-svc/requirements.txt
touch encryption-svc/Dockerfile
touch encryption-svc/.env.example
```

Cấu trúc sau khi tạo:
```
encryption-svc/
├── app/
│   ├── __init__.py
│   ├── main.py            # Entry point: gọi start_consumer()
│   ├── kafka_consumer.py  # Poll Kafka, parse event, gọi encrypt
│   ├── encryptor.py       # AES-256-GCM encrypt/decrypt
│   └── minio_client.py    # Download/upload MinIO
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## 4.2 `requirements.txt`

```
confluent-kafka==2.3.0
boto3==1.34.70
cryptography>=42.0.0
python-dotenv==1.0.1
```

---

## 4.3 `.env.example`

```bash
# Kafka
KAFKA_BOOTSTRAP_SERVERS=127.0.0.1:9092
KAFKA_GROUP_ID=encryption-service-group
KAFKA_TOPIC=data-cleaned-topic

# MinIO
MINIO_ENDPOINT=http://127.0.0.1:9000
MINIO_ACCESS_KEY=admin
MINIO_SECRET_KEY=password
MINIO_REGION=us-east-1
ENCRYPTED_BUCKET=encrypted-zone

# Encryption
# QUAN TRỌNG: Trong production dùng KMS, KHÔNG hardcode!
# Tạo key: python3 -c "import os; print(os.urandom(32).hex())"
ENCRYPTION_KEY=your_64_char_hex_key_here
```

---

## 4.4 `app/encryptor.py`

```python
"""
encryptor.py
Mã hoá / giải mã file bằng AES-256-GCM.

AES-256-GCM là Authenticated Encryption:
  - AES-256: Khối mã hoá 256-bit (NIST FIPS 197)
  - GCM mode: Galois/Counter Mode - phát hiện tamper
  - Nonce 12 bytes random mỗi file (KHÔNG tái sử dụng!)

Format file encrypted:
  [12 bytes nonce] [n bytes ciphertext + 16 bytes auth_tag]
"""

import logging
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)


def _get_key() -> bytes:
    """Lấy encryption key từ env. Production: dùng AWS KMS / Vault."""
    hex_key = os.environ.get("ENCRYPTION_KEY", "")
    if len(hex_key) != 64:
        raise EnvironmentError(
            "ENCRYPTION_KEY must be 64 hex chars (32 bytes = AES-256). "
            "Generate: python3 -c \"import os; print(os.urandom(32).hex())\""
        )
    return bytes.fromhex(hex_key)


def encrypt_file(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Mã hoá file bằng AES-256-GCM.

    Args:
        input_path:  Đường dẫn file cần mã hoá
        output_path: Đường dẫn file output (None = input_path + ".enc")

    Returns:
        Đường dẫn file đã mã hoá

    Raises:
        FileNotFoundError: Nếu input_path không tồn tại
        EnvironmentError:  Nếu ENCRYPTION_KEY không hợp lệ
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    key   = _get_key()
    aesgcm = AESGCM(key)
    nonce  = os.urandom(12)   # 96-bit random nonce

    with open(input_path, "rb") as f:
        plaintext = f.read()

    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)

    if output_path is None:
        output_path = input_path + ".enc"

    with open(output_path, "wb") as f:
        f.write(nonce + ciphertext)   # nonce prepended for decryption

    original_size  = len(plaintext)
    encrypted_size = len(nonce) + len(ciphertext)
    logger.info(
        f"Encrypted {input_path} → {output_path} "
        f"({original_size} → {encrypted_size} bytes, overhead={encrypted_size - original_size}B)"
    )
    return output_path


def decrypt_file(encrypted_path: str, output_path: Optional[str] = None) -> str:
    """
    Giải mã file đã mã hoá bởi encrypt_file().

    Args:
        encrypted_path: Đường dẫn file .enc
        output_path:    Đường dẫn file output (None = bỏ suffix .enc)

    Returns:
        Đường dẫn file đã giải mã

    Raises:
        cryptography.exceptions.InvalidTag: Nếu file bị tamper
    """
    if not Path(encrypted_path).exists():
        raise FileNotFoundError(f"Encrypted file not found: {encrypted_path}")

    key    = _get_key()
    aesgcm = AESGCM(key)

    with open(encrypted_path, "rb") as f:
        data = f.read()

    nonce      = data[:12]
    ciphertext = data[12:]
    plaintext  = aesgcm.decrypt(nonce, ciphertext, associated_data=None)

    if output_path is None:
        output_path = encrypted_path.removesuffix(".enc")

    with open(output_path, "wb") as f:
        f.write(plaintext)

    logger.info(f"Decrypted {encrypted_path} → {output_path} ({len(plaintext)} bytes)")
    return output_path


# Thêm import Optional
from typing import Optional
```

---

## 4.5 `app/minio_client.py`

```python
"""
minio_client.py (encryption-svc)
Download / upload files từ MinIO cho encryption service.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)

_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",    "http://127.0.0.1:9000")
_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY",   "admin")
_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY",   "password")
_REGION     = os.environ.get("MINIO_REGION",       "us-east-1")
_ENC_BUCKET = os.environ.get("ENCRYPTED_BUCKET",   "encrypted-zone")


def get_client():
    return boto3.client(
        "s3",
        endpoint_url=_ENDPOINT,
        aws_access_key_id=_ACCESS_KEY,
        aws_secret_access_key=_SECRET_KEY,
        region_name=_REGION,
        config=Config(signature_version="s3v4"),
    )


def ensure_bucket(client, bucket_name: str) -> None:
    try:
        client.head_bucket(Bucket=bucket_name)
    except Exception:
        client.create_bucket(Bucket=bucket_name)
        logger.info(f"Created bucket: {bucket_name}")


def download_to_temp(bucket: str, key: str, suffix: str = ".parquet") -> str:
    """Download object từ MinIO → temp file, trả về local path."""
    client = get_client()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        client.download_fileobj(Bucket=bucket, Key=key, Fileobj=tmp)
    finally:
        tmp.close()
    logger.debug(f"Downloaded s3://{bucket}/{key} → {tmp.name}")
    return tmp.name


def upload_file(local_path: str, bucket: str, key: str) -> None:
    """Upload local file lên MinIO."""
    client = get_client()
    ensure_bucket(client, bucket)
    file_size = Path(local_path).stat().st_size
    with open(local_path, "rb") as f:
        client.put_object(
            Bucket=bucket, Key=key,
            Body=f, ContentLength=file_size,
            ContentType="application/octet-stream",
        )
    logger.info(f"Uploaded {local_path} → s3://{bucket}/{key} ({file_size} bytes)")
```

---

## 4.6 `app/kafka_consumer.py`

```python
"""
kafka_consumer.py (encryption-svc)
Lắng nghe data-cleaned-topic, xử lý DATA_CLEANING_COMPLETED event.

Flow:
  1. Poll event từ Kafka
  2. Parse event_type, version_id, clean_file_paths
  3. Với mỗi file anonymized (chứa _anon_ hoặc _dp_):
     a. Download từ MinIO clean-zone
     b. Encrypt bằng AES-256-GCM
     c. Upload lên MinIO encrypted-zone
  4. Commit offset (manual) chỉ sau khi encrypt thành công
"""

import json
import logging
import os
import time
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, KafkaException

from app.encryptor import encrypt_file
from app.minio_client import download_to_temp, upload_file

logger = logging.getLogger(__name__)

_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9092")
_GROUP_ID  = os.environ.get("KAFKA_GROUP_ID",          "encryption-service-group")
_TOPIC     = os.environ.get("KAFKA_TOPIC",              "data-cleaned-topic")
_ENC_BUCKET = os.environ.get("ENCRYPTED_BUCKET",        "encrypted-zone")

# File tags cần encrypt (bỏ qua file clean thô adult_clean.parquet)
_ENCRYPT_TAGS = ("_anon_k", "_anon_l", "_ldiv", "_dp_")


def _should_encrypt(file_path: str) -> bool:
    return any(tag in file_path for tag in _ENCRYPT_TAGS)


def _build_encrypted_key(clean_key: str) -> str:
    """
    Tạo key cho encrypted-zone từ clean-zone key.
    Ví dụ: "2026-04-08_14-30/adult_anon_k10.parquet"
         → "2026-04-08_14-30/adult_anon_k10.parquet.enc"
    """
    return clean_key + ".enc"


def _handle_event(event: dict) -> list:
    """
    Xử lý 1 DATA_CLEANING_COMPLETED event.

    Returns:
        List các encrypted MinIO paths đã upload thành công
    """
    version_id   = event.get("version_id", "unknown")
    file_paths   = event.get("clean_file_paths", [])
    encrypted_ok = []

    logger.info(f"[{version_id}] Processing {len(file_paths)} files...")

    for file_path in file_paths:
        if not _should_encrypt(file_path):
            logger.debug(f"Skip (not anonymized): {file_path}")
            continue

        try:
            # Parse bucket/key từ path "clean-zone/version/file.parquet"
            parts  = file_path.split("/", 1)
            if len(parts) != 2:
                logger.warning(f"Cannot parse path: {file_path}")
                continue

            bucket, key = parts

            # 1. Download → temp
            local_path = download_to_temp(bucket=bucket, key=key, suffix=".parquet")

            # 2. Encrypt → .enc file
            enc_local_path = local_path + ".enc"
            encrypt_file(input_path=local_path, output_path=enc_local_path)

            # 3. Upload lên encrypted-zone
            enc_key = _build_encrypted_key(key)
            upload_file(
                local_path=enc_local_path,
                bucket=_ENC_BUCKET,
                key=enc_key,
            )

            result_path = f"{_ENC_BUCKET}/{enc_key}"
            encrypted_ok.append(result_path)
            logger.info(f"✓ Encrypted: {file_path} → {result_path}")

        except Exception as e:
            logger.error(f"✗ Failed to encrypt {file_path}: {e}", exc_info=True)

    logger.info(f"[{version_id}] Done: {len(encrypted_ok)}/{len(file_paths)} encrypted")
    return encrypted_ok


def start_consumer(max_retries: int = 5) -> None:
    """
    Start Kafka consumer loop.

    Args:
        max_retries: Số lần retry khi Kafka unavailable
    """
    conf = {
        "bootstrap.servers":  _BOOTSTRAP,
        "group.id":           _GROUP_ID,
        "auto.offset.reset":  "earliest",
        "enable.auto.commit": False,   # Manual commit!
    }

    consumer = Consumer(conf)
    consumer.subscribe([_TOPIC])
    logger.info(f"[encryption-svc] Started | topic={_TOPIC} | group={_GROUP_ID}")

    retries = 0
    try:
        while True:
            msg = consumer.poll(timeout=2.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.debug("Reached end of partition")
                    continue
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    logger.warning(f"Topic not found yet, retry {retries}/{max_retries}")
                    retries += 1
                    if retries > max_retries:
                        raise KafkaException(msg.error())
                    time.sleep(5)
                    continue
                else:
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

            retries = 0   # Reset retry counter on success

            try:
                event = json.loads(msg.value().decode("utf-8"))
                event_type = event.get("event_type", "")

                if event_type == "DATA_CLEANING_COMPLETED":
                    _handle_event(event)
                else:
                    logger.debug(f"Ignored event: {event_type}")

                # Commit CHỈ sau khi xử lý xong
                consumer.commit(message=msg)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in message: {e}")
                consumer.commit(message=msg)   # Skip bad message

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    finally:
        consumer.close()
        logger.info("Consumer closed")
```

---

## 4.7 `app/main.py`

```python
"""
main.py (encryption-svc)
Entry point. Load .env và start consumer.
"""

import logging
import os

from dotenv import load_dotenv

load_dotenv()   # Load .env file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from app.kafka_consumer import start_consumer

if __name__ == "__main__":
    start_consumer()
```

---

## 4.8 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["python", "-m", "app.main"]
```

---

## 4.9 Cập Nhật `docker-compose.yml` (Root Level)

Tạo file `credit-score-apps/docker-compose.yml` orchestrate cả 2 services:

```yaml
version: "3.8"

services:
  # ── Infrastructure ────────────────────────────────────────────────────────
  kafka:
    image: apache/kafka:3.7.0
    container_name: local-kafka
    ports:
      - "9092:9092"
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://127.0.0.1:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@localhost:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS: 0
    healthcheck:
      test: ["CMD-SHELL", "/opt/kafka/bin/kafka-broker-api-versions.sh --bootstrap-server localhost:9092 > /dev/null 2>&1"]
      interval: 10s
      timeout: 5s
      retries: 5

  minio:
    image: minio/minio:latest
    container_name: local-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: password
    command: server /data --console-address ":9001"
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 3

  # ── Application Services ──────────────────────────────────────────────────
  preprocess-svc:
    build: ./preprocess-svc
    container_name: preprocess-svc
    ports:
      - "8000:8000"
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      MINIO_ENDPOINT: http://minio:9000
      MINIO_ACCESS_KEY: admin
      MINIO_SECRET_KEY: password
    depends_on:
      kafka:
        condition: service_healthy
      minio:
        condition: service_healthy
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  encryption-svc:
    build: ./encryption-svc
    container_name: encryption-svc
    environment:
      KAFKA_BOOTSTRAP_SERVERS: kafka:9092
      KAFKA_GROUP_ID: encryption-service-group
      KAFKA_TOPIC: data-cleaned-topic
      MINIO_ENDPOINT: http://minio:9000
      MINIO_ACCESS_KEY: admin
      MINIO_SECRET_KEY: password
      ENCRYPTED_BUCKET: encrypted-zone
      # PRODUCTION: Dùng Docker secrets hoặc env từ KMS
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
    depends_on:
      kafka:
        condition: service_healthy
      minio:
        condition: service_healthy

volumes:
  minio_data:
```

---

## 4.10 Tạo `.env` ở Root Level

```bash
# credit-score-apps/.env
# Tạo key: python3 -c "import os; print(os.urandom(32).hex())"
ENCRYPTION_KEY=a1b2c3d4e5f6789012345678901234567890123456789012345678901234abcd
```

---

## 4.11 Chạy End-to-End Test

```bash
# Bước 1: Tạo .env với key thật
python3 -c "import os; print('ENCRYPTION_KEY=' + os.urandom(32).hex())" > .env

# Bước 2: Start tất cả services
docker-compose up -d

# Bước 3: Đợi services healthy
docker-compose ps

# Bước 4: Upload file test
curl -X POST "http://localhost:8000/upload" \
  -F "files=@preprocess-svc/adult.data"

# Bước 5: Kiểm tra Kafka consumer nhận event
docker logs encryption-svc -f

# Bước 6: Kiểm tra MinIO có file encrypted
# Truy cập http://localhost:9001 (user: admin, pass: password)
# → bucket encrypted-zone → tìm file *.parquet.enc

# Bước 7: Verify encrypt/decrypt
docker exec -it encryption-svc python3 -c "
from app.encryptor import encrypt_file, decrypt_file
# Test với file parquet bất kỳ
enc = encrypt_file('/tmp/test.parquet')
dec = decrypt_file(enc)
print(f\'Encrypted: {enc}, Decrypted: {dec}\')
"
```

---

## 4.12 Commit Bước 4

```bash
git add encryption-svc/
git add docker-compose.yml
git add .env.example
git commit -m "feat: add encryption-svc with AES-256-GCM, Kafka consumer, root docker-compose"
```

---

## 4.13 Checklist Bước 4

- [ ] Tạo `encryption-svc/` với đầy đủ cấu trúc
- [ ] `encryptor.py`: AES-256-GCM encrypt/decrypt
- [ ] `minio_client.py`: download/upload
- [ ] `kafka_consumer.py`: poll, parse event, gọi encrypt
- [ ] `main.py`: entry point
- [ ] `Dockerfile` cho encryption-svc
- [ ] Root `docker-compose.yml` orchestrate cả kafka + minio + 2 services
- [ ] `.env` với ENCRYPTION_KEY thật
- [ ] End-to-end test: upload → clean → Kafka → encrypt → verify
- [ ] Commit

**Tiếp theo:** [Bước 5 — Gaussian Mechanism](./09_step05_gaussian_mechanism.md)
