import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from confluent_kafka import Consumer, KafkaError, KafkaException

from app.core.minio_client import download_parquet_from_minio, ensure_bucket, get_minio_client
from app.core.model_trainer import train_and_save_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': '127.0.0.1:9092',
    'group.id': 'analytic-training-group',
    'auto.offset.reset': 'earliest'
}

TOPIC = 'data-cleaned-topic'


def process_message(message: Dict[str, Any]) -> None:
    """Process a Kafka message to download data, train model, and upload."""
    try:
        event_type = message.get('event_type')
        if event_type != 'DATA_CLEANING_COMPLETED':
            logger.warning(f"Ignoring message with event_type: {event_type}")
            return

        version_id = message.get('version_id')
        clean_file_paths = message.get('clean_file_paths', [])

        if not version_id or not clean_file_paths:
            logger.error("Invalid message: missing version_id or clean_file_paths")
            return

        # Ensure model-zone bucket exists
        client = get_minio_client()
        ensure_bucket(client, "model-zone")

        # Process each file path
        for file_path in clean_file_paths:
            if not file_path.startswith('clean-zone/'):
                logger.warning(f"Skipping invalid file path: {file_path}")
                continue

            # Extract bucket and key
            bucket = 'clean-zone'
            key = file_path[len('clean-zone/'):]

            # Download to temp file
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                download_parquet_from_minio(client, bucket, key, temp_path)
                logger.info(f"Downloaded {file_path} to {temp_path}")

                # Train and save model
                train_and_save_model(temp_path, version_id)
                logger.info(f"Trained and uploaded model for version {version_id}")

            finally:
                Path(temp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.error(f"Error processing message: {e}")


def consume_messages() -> None:
    """Consume messages from Kafka and process them."""
    consumer = Consumer(KAFKA_CONFIG)

    try:
        consumer.subscribe([TOPIC])
        logger.info(f"Subscribed to topic: {TOPIC}")

        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info('Reached end of partition')
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                try:
                    message = json.loads(msg.value().decode('utf-8'))
                    logger.info(f"Received message: {message}")
                    process_message(message)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")

    except KeyboardInterrupt:
        logger.info("Consumer interrupted")
    finally:
        consumer.close()