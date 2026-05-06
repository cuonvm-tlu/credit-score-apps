import json
import logging
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple

from confluent_kafka import Consumer, Producer

from app.core.anonymize_k_anonymity import anonymize_cleaned_adult_k_anonymity_and_upload
from app.core.anonymize_l_diversity import anonymize_cleaned_adult_l_diversity_and_upload
from app.core.minio_client import ensure_bucket, get_minio_client

logger = logging.getLogger(__name__)

KAFKA_CONFIG = {"bootstrap.servers": "127.0.0.1:9092"}
CONSUMER_CONFIG = {
    **KAFKA_CONFIG,
    "group.id": "preprocess-svc-anonymize-worker",
    "auto.offset.reset": "earliest",
}

CLEANING_TOPIC = "data-cleaned-topic"
ANONYMIZED_TOPIC = "data-anonymized-topic"
ANONYMIZE_BUCKET = "anonymize-zone"

_worker_thread: Optional[Thread] = None
_stop_event = Event()


def start_anonymization_consumer() -> None:
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return

    _stop_event.clear()
    _worker_thread = Thread(target=_consume_forever, name="kafka-anonymize-worker", daemon=True)
    _worker_thread.start()
    logger.info("Kafka anonymization consumer started.")


def stop_anonymization_consumer() -> None:
    if not _worker_thread:
        return

    _stop_event.set()
    _worker_thread.join(timeout=5)
    logger.info("Kafka anonymization consumer stopped.")


def _consume_forever() -> None:
    consumer = Consumer(CONSUMER_CONFIG)
    consumer.subscribe([CLEANING_TOPIC])
    logger.info("Subscribed to topic: %s", CLEANING_TOPIC)

    try:
        while not _stop_event.is_set():
            message = consumer.poll(timeout=1.0)
            if message is None:
                continue
            if message.error():
                logger.error("Kafka consume error: %s", message.error())
                continue

            _handle_cleaning_completed_message(message.value())
    except Exception:
        logger.exception("Kafka anonymization worker crashed.")
    finally:
        consumer.close()


def _handle_cleaning_completed_message(raw_value: bytes) -> None:
    try:
        payload = json.loads(raw_value.decode("utf-8"))
    except Exception:
        logger.exception("Invalid Kafka payload for anonymization.")
        return

    if payload.get("event_type") != "DATA_CLEANING_COMPLETED":
        return
    if payload.get("status") != "success":
        return

    version_id = payload.get("version_id")
    clean_paths = payload.get("clean_file_paths", [])
    if not version_id or not isinstance(clean_paths, list) or not clean_paths:
        logger.warning("Skip message due to missing version_id or clean_file_paths.")
        return

    try:
        output_paths = _run_anonymization(clean_paths)
        _send_anonymizing_completed_event(version_id, output_paths)
    except Exception:
        logger.exception("Anonymization processing failed for version_id=%s", version_id)


def _run_anonymization(clean_paths: List[str]) -> List[str]:
    client = get_minio_client()
    ensure_bucket(client, ANONYMIZE_BUCKET)
    anonymized_paths: List[str] = []

    for clean_path in clean_paths:
        bucket_and_key = _split_bucket_and_key(clean_path)
        if not bucket_and_key:
            logger.warning("Skip invalid clean path: %s", clean_path)
            continue
        clean_bucket, clean_key = bucket_and_key

        k_path = anonymize_cleaned_adult_k_anonymity_and_upload(
            client=client,
            clean_bucket=clean_bucket,
            clean_object_key=clean_key,
            anonymize_bucket=ANONYMIZE_BUCKET,
            k=10,
        )
        anonymized_paths.append(k_path)

        l_path = anonymize_cleaned_adult_l_diversity_and_upload(
            client=client,
            clean_bucket=clean_bucket,
            clean_object_key=clean_key,
            anonymize_bucket=ANONYMIZE_BUCKET,
            l_value=2,
        )
        anonymized_paths.append(l_path)

    if not anonymized_paths:
        raise ValueError("No anonymized outputs generated.")

    return anonymized_paths


def _split_bucket_and_key(path: str) -> Optional[Tuple[str, str]]:
    if "/" not in path:
        return None
    bucket, key = path.split("/", 1)
    if not bucket or not key:
        return None
    return bucket, key


def _send_anonymizing_completed_event(version_id: str, anonymized_paths: List[str]) -> None:
    producer = Producer(KAFKA_CONFIG)
    event: Dict[str, Any] = {
        "event_type": "DATA_ANNONIMIZING_COMPLETED",
        "status": "success",
        "version_id": version_id,
        "annonimize_file_paths": anonymized_paths,
    }

    producer.produce(ANONYMIZED_TOPIC, value=json.dumps(event).encode("utf-8"))
    producer.flush()
    logger.info("Published DATA_ANNONIMIZING_COMPLETED for version_id=%s", version_id)
