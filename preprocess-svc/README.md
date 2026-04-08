# Preprocessing Service

A FastAPI-based microservice for preprocessing Adult Census Income dataset files. It uploads raw files to MinIO, cleans the data using Pandas, saves cleaned Parquet files to MinIO, and publishes Kafka events for downstream processing.

## Features

- **File Upload**: Accepts multiple file uploads via POST `/upload` endpoint.
- **Data Cleaning**: Applies specific cleaning rules to data files (adult.data, adult.test, or .csv files).
- **MinIO Integration**: Stores raw files in `landing-zone` and cleaned Parquet files in `clean-zone` with timestamp-based versioning.
- **Kafka Messaging**: Publishes events to `data-cleaned-topic` after successful cleaning.
- **Event-Driven**: Triggers downstream services via Kafka messages.

## Prerequisites

- Python 3.8+
- MinIO server running locally on `http://127.0.0.1:9000`
- Kafka broker running locally on `127.0.0.1:9092`
- Docker and Docker Compose (for Kafka)

## Setup

### 1. Clone or Navigate to the Project Directory

```bash
cd /path/to/preprocess-svc
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

### Upload Files

**Endpoint**: `POST /upload`

**Description**: Upload multiple files. Raw files are saved to MinIO `landing-zone`. Data files are cleaned and saved as Parquet to `clean-zone`. A Kafka message is published upon successful cleaning.

**Request**:
- `files`: List of files to upload (multipart/form-data)

**Response**:
```json
{
  "message": "Upload completed",
  "landing_zone_paths": ["landing-zone/2026:04:08:14:30/adult.data"],
  "clean_zone_paths": ["clean-zone/2026:04:08:14:30/adult_clean.parquet"]
}
```

**Example using curl**:
```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -F "files=@adult.data" \
  -F "files=@adult.test"
```

## Data Cleaning Rules

For data files (adult.data, adult.test, or .csv):
- Assigns 15 column names: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income
- Converts "?" to NaN and drops rows with missing values
- Drops 'fnlwgt' column
- Strips whitespace from string columns
- Maps 'income' to binary (0 for <=50K, 1 for >50K)
- Saves as Parquet

## Kafka Events

After cleaning, a message is sent to `data-cleaned-topic`:

```json
{
  "event_type": "DATA_CLEANING_COMPLETED",
  "status": "success",
  "version_id": "2026:04:08:14:30",
  "clean_file_paths": ["clean-zone/2026:04:08:14:30/adult_clean.parquet"]
}
```

## Configuration

- **MinIO**: Endpoint `http://127.0.0.1:9000`, Access Key `admin`, Secret Key `password`, Region `us-east-1`
- **Kafka**: Bootstrap servers `127.0.0.1:9092`, Topic `data-cleaned-topic`

## Development

- The service uses timestamp-based versioning for file organization.
- Buckets `landing-zone` and `clean-zone` are created automatically if they don't exist.
- Error handling ensures Kafka failures don't crash the upload process.

## Stopping Services

```bash
# Stop Kafka
docker-compose down

# Stop MinIO (if using Docker)
docker stop minio
```