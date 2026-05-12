# Analytic Service

An event-driven microservice for training machine learning models on preprocessed credit scoring data. It consumes Kafka messages from the preprocessing service, downloads cleaned data from MinIO, trains a Random Forest classifier, and uploads the trained model back to MinIO.

## Features

- **Event-Driven**: Triggered by Kafka messages from the preprocessing service.
- **Data Download**: Downloads Parquet files from MinIO `clean-zone`.
- **Model Training**: Trains a Random Forest classifier with one-hot encoding for categorical features.
- **Model Upload**: Saves trained models to MinIO `model-zone` with versioning.
- **Error Handling**: Graceful handling of invalid messages and connection issues.

## Prerequisites

- Python 3.9+
- MinIO server running locally on `http://127.0.0.1:9000`
- Kafka broker running locally on `127.0.0.1:9092`
- Preprocessing service publishing to `data-cleaned-topic`

## Setup

### 1. Install Dependencies

```bash
cd analytic-service
pip install -r requirements.txt
```

### 2. Ensure Services are Running

- MinIO: `http://127.0.0.1:9000`
- Kafka: `127.0.0.1:9092`
- Preprocessing service (to publish messages)

## Running the Service

```bash
python app/main.py
```

The service will continuously poll for Kafka messages and process them.

## Configuration

- **Kafka**: Bootstrap servers `127.0.0.1:9092`, Topic `data-cleaned-topic`, Group ID `analytic-training-group`
- **MinIO**: Endpoint `http://127.0.0.1:9000`, Access Key `admin`, Secret Key `password`, Region `us-east-1`

## Workflow

1. Receives Kafka message with cleaned file paths.
2. Downloads Parquet file from `clean-zone`.
3. Trains Random Forest model on the data.
4. Uploads `rf_model.joblib` to `model-zone/{version_id}/`.

## Development

- Models are versioned using the `version_id` from Kafka messages.
- The `model-zone` bucket is created automatically if it doesn't exist.
- Logs are printed to console for monitoring.