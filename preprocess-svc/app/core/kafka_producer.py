import json
import logging
from typing import List

from confluent_kafka import Producer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': '127.0.0.1:9092',
}

TOPIC = 'data-cleaned-topic'


def delivery_report(err, msg):
    """Callback for message delivery reports."""
    if err is not None:
        logger.error(f'Message delivery failed: {err}')
    else:
        logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')


def send_cleaning_success_event(version_folder: str, clean_file_paths: List[str]) -> None:
    """
    Send a Kafka message indicating that data cleaning has completed successfully.

    Args:
        version_folder (str): The timestamp-based version folder (e.g., '2026:04:08:14:30').
        clean_file_paths (List[str]): List of MinIO paths for the cleaned files.
    """
    try:
        producer = Producer(KAFKA_CONFIG)

        message = {
            "event_type": "DATA_CLEANING_COMPLETED",
            "status": "success",
            "version_id": version_folder,
            "clean_file_paths": clean_file_paths
        }

        # Produce the message
        producer.produce(
            TOPIC,
            value=json.dumps(message).encode('utf-8'),
            callback=delivery_report
        )

        # Wait for any outstanding messages to be delivered
        producer.flush()

        logger.info(f"Sent Kafka message for version {version_folder}")

    except Exception as e:
        logger.error(f"Failed to send Kafka message: {e}")
        # Do not raise exception to avoid crashing the FastAPI request