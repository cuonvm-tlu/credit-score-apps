# Analytic Service

An event-driven microservice for training machine learning models on credit scoring data. It operates **two Kafka consumers** running in parallel:

1. **Clean-data consumer** — consumes `data-cleaned-topic` (from `preprocess-svc`), downloads cleaned Parquet files from `clean-zone`, trains a Random Forest classifier, and uploads the model to `model-zone`.
2. **Anonymize-data consumer** — consumes `data-anonymized-topic` (from `anonymize-svc`), downloads anonymized Parquet files from `anonymize-zone`, trains a Random Forest classifier **per anonymized variant** (k5, k10, l-diversity, DP…), and uploads each model to `model-zone`.

## Event Flow

```
preprocess-svc
  │  publishes DATA_CLEANING_COMPLETED
  └─► [data-cleaned-topic] ──► analytic-service (consumer 1)
                                  └─► train RF → model-zone/{version_id}/rf_model.joblib

anonymize-svc
  │  listens data-cleaned-topic, runs k/l/DP anonymization
  │  publishes DATA_ANNONIMIZING_COMPLETED
  └─► [data-anonymized-topic] ──► analytic-service (consumer 2)
                                    └─► train RF per variant
                                        → model-zone/{version_id}/anon/{stem}.joblib
```

## Features

- **Dual Consumer**: Two independent Kafka consumers running as daemon threads.
- **Data Download**: Downloads Parquet files from MinIO `clean-zone` or `anonymize-zone`.
- **Model Training**: Trains a Random Forest classifier with one-hot encoding for categorical features.
- **Model Upload**: Saves trained models to MinIO `model-zone` with versioning.
- **Graceful Shutdown**: SIGINT/SIGTERM stops both consumers cleanly.
- **Error Handling**: Per-file error isolation — one failed file does not stop the rest.

## Prerequisites

- Python 3.9+
- MinIO server running locally on `http://127.0.0.1:9000`
- Kafka broker running locally on `127.0.0.1:9092`
- `preprocess-svc` publishing to `data-cleaned-topic`
- `anonymize-svc` publishing to `data-anonymized-topic`

## Setup

### 1. Install Dependencies

```bash
cd analytic-service
pip install -r requirements.txt
```

### 2. Ensure Services are Running

- MinIO: `http://127.0.0.1:9000`
- Kafka: `127.0.0.1:9092`
- `preprocess-svc` (to publish clean-data messages)
- `anonymize-svc` (to publish anonymized-data messages)

## Running the Service

```bash
python -m app.main
```

The service will continuously poll both Kafka topics and process them.

## Configuration

| Setting | Value |
|---|---|
| Kafka bootstrap servers | `127.0.0.1:9092` |
| Consumer group (clean) | `analytic-training-group` |
| Consumer group (anonymize) | `analytic-anonymize-training-group` |
| Topic (clean) | `data-cleaned-topic` |
| Topic (anonymize) | `data-anonymized-topic` |
| MinIO endpoint | `http://127.0.0.1:9000` |
| MinIO credentials | `admin` / `password` |
| MinIO source (clean) | `clean-zone` |
| MinIO source (anonymize) | `anonymize-zone` |
| MinIO model destination | `model-zone` |

## Model Key Naming

| Source | MinIO model key |
|---|---|
| Clean data | `{version_id}/rf_model.joblib` |
| Anonymized (k=5) | `{version_id}/anon/{stem}_anon_k5.joblib` |
| Anonymized (k=10) | `{version_id}/anon/{stem}_anon_k10.joblib` |
| Anonymized (l-div) | `{version_id}/anon/{stem}_anon_l2.joblib` |
| Anonymized (DP) | `{version_id}/anon/{stem}_dp_*.joblib` |

## Development

- Models are versioned using the `version_id` from Kafka messages.
- The `model-zone` bucket is created automatically if it doesn't exist.
- All logs are written to stdout for easy monitoring/Docker log collection.