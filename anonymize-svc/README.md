# Anonymization Service

A FastAPI-based worker service for anonymizing Adult Census Income data. It consumes cleaned-data Kafka messages, runs k-anonymity, l-diversity, and differential privacy (DP), writes outputs to MinIO, and publishes anonymization completion events.

## Features

- **Kafka Consumer Worker**: Consumes `DATA_CLEANING_COMPLETED` events from `data-cleaned-topic`.
- **K-anonymity + L-diversity + DP**: Applies anonymization for eligible `*_clean.parquet` files.
- **MinIO Integration**: Reads inputs from `clean-zone` and writes anonymized outputs to `anonymize-zone`.
- **Kafka Producer**: Publishes `DATA_ANNONIMIZING_COMPLETED` events to `data-anonymized-topic`.
- **Health Endpoint**: Exposes `GET /health` for liveness checks.

## Prerequisites

- Python 3.8+
- MinIO server running locally on `http://127.0.0.1:9000`
- Kafka broker running locally on `127.0.0.1:9092`
- Docker and Docker Compose (for Kafka)

## Setup

### 1. Clone or Navigate to the Service Directory

```bash
cd /path/to/anonymize-svc
```

### 2. Install Python Dependencies

Create a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start MinIO

Ensure MinIO is running locally. If not, start it using Docker or your preferred method:

```bash
# Example using Docker
docker run -d -p 9000:9000 -p 9001:9001 --name minio \
  -e "MINIO_ACCESS_KEY=admin" \
  -e "MINIO_SECRET_KEY=password" \
  -v /tmp/minio:/data \
  minio/minio server /data --console-address ":9001"
```

### 4. Start Kafka

Use Docker Compose to start the Kafka broker:

```bash
docker-compose up -d
```

This will start Kafka on `localhost:9092`.

### 5. Run the FastAPI Application

```bash
uvicorn app.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

## API Usage

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "ok",
  "service": "anonymize-svc"
}
```

## Kafka Events

### 1) Input event (consumed)

The service consumes this event from `data-cleaned-topic`:

```json
{
  "event_type": "DATA_CLEANING_COMPLETED",
  "status": "success",
  "version_id": "2026:04:08:14:30",
  "clean_file_paths": ["clean-zone/2026:04:08:14:30/adult_clean.parquet"]
}
```

### 2) Output event (published)

After running K-anonymity + L-diversity + DP, the service publishes to `data-anonymized-topic`:

```json
{
  "event_type": "DATA_ANNONIMIZING_COMPLETED",
  "status": "success",
  "version_id": "2026-05-06_21-30-00",
  "annonimize_file_paths": [
    "anonymize-zone/2026-05-06_21-30-00/adult_anon_k10.parquet",
    "anonymize-zone/2026-05-06_21-30-00/adult_anon_l2.parquet",
    "anonymize-zone/2026-05-06_21-30-00/adult_dp_e0_30_src2026-05-06_21-30-00_run20260506213112.parquet"
  ]
}
```

### Anonymize flow (short)

1. `preprocess-svc` sends `DATA_CLEANING_COMPLETED` -> `data-cleaned-topic`.
2. `anonymize-svc` worker consumes message and runs:
   - K-anonymity (k=10)
   - L-diversity (l=2)
  - Differential Privacy (epsilon=0.3)
3. Worker uploads anonymized files -> `anonymize-zone`.
4. Worker sends `DATA_ANNONIMIZING_COMPLETED` -> `data-anonymized-topic`.

## Configuration

- **MinIO**: Endpoint `http://127.0.0.1:9000`, Access Key `minioadmin`, Secret Key `minioadmin`, Region `us-east-1`
- **Kafka**: Bootstrap servers `127.0.0.1:9092`, Topics `data-cleaned-topic` and `data-anonymized-topic`

## Development

- The service uses timestamp-based versioning for file organization.
- Buckets `landing-zone`, `clean-zone`, and `anonymize-zone` are created automatically if they don't exist.
- Worker skips non-eligible paths (e.g. DP outputs) and only anonymizes `*_clean.parquet`.

## Stopping Services

```bash
# Stop Kafka
docker-compose down

# Stop MinIO (if using Docker)
docker stop minio
```