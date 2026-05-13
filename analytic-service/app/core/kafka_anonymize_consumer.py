"""
kafka_anonymize_consumer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Consume events từ ``data-anonymized-topic`` do anonymize-svc publish sau khi
hoàn thành k-anonymity / l-diversity / DP.

Event schema (xem anonymize-svc/app/core/kafka_anonymize_worker.py):
    {
        "event_type": "DATA_ANNONIMIZING_COMPLETED",   # typo giữ nguyên theo upstream
        "status":     "success",
        "version_id": "<timestamp_folder>",
        "annonimize_file_paths": [                     # typo giữ nguyên theo upstream
            "anonymize-zone/<version_id>/<stem>_anon_k5.parquet",
            "anonymize-zone/<version_id>/<stem>_anon_k10.parquet",
            ...
        ]
    }

Với mỗi file .parquet trong ``annonimize_file_paths``:
  1. Download từ MinIO bucket ``anonymize-zone``
  2. Train RandomForest (dùng lại train_and_save_model)
  3. Upload model lên ``model-zone/<version_id>/anon/<filename_stem>.joblib``
"""

import json
import logging
import tempfile
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Optional

from confluent_kafka import Consumer, KafkaError, KafkaException

from app.core.minio_client import (
    download_parquet_from_minio,
    ensure_bucket,
    get_minio_client,
)
from app.core.model_trainer import train_and_save_model

logger = logging.getLogger(__name__)

# ── Kafka config ──────────────────────────────────────────────────────────────
KAFKA_CONFIG = {
    "bootstrap.servers": "127.0.0.1:9092",
    "group.id": "analytic-anonymize-training-group",
    "auto.offset.reset": "earliest",
}

ANONYMIZED_TOPIC = "data-anonymized-topic"
ANONYMIZE_BUCKET = "anonymize-zone"
MODEL_BUCKET = "model-zone"

# ── Background thread management ──────────────────────────────────────────────
_worker_thread: Optional[Thread] = None
_stop_event = Event()


def start_anonymize_training_consumer() -> None:
    """Start background Kafka consumer thread (idempotent)."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return

    _stop_event.clear()
    _worker_thread = Thread(
        target=_consume_forever,
        name="kafka-anonymize-training-worker",
        daemon=True,
    )
    _worker_thread.start()
    logger.info(
        "Anonymize-training Kafka consumer started (topic=%s).", ANONYMIZED_TOPIC
    )


def stop_anonymize_training_consumer() -> None:
    """Signal background thread to stop and wait for it."""
    if not _worker_thread:
        return
    _stop_event.set()
    _worker_thread.join(timeout=10)
    logger.info("Anonymize-training Kafka consumer stopped.")


# ── Internal consumer loop ────────────────────────────────────────────────────

def _consume_forever() -> None:
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([ANONYMIZED_TOPIC])
    logger.info("Subscribed to topic: %s", ANONYMIZED_TOPIC)

    try:
        while not _stop_event.is_set():
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                code = msg.error().code()
                if code == KafkaError._PARTITION_EOF:
                    logger.debug("Reached end of partition on %s", ANONYMIZED_TOPIC)
                elif code == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    # Topic chưa tồn tại — chờ rồi thử lại
                    logger.warning(
                        "Topic '%s' not yet available, will retry… (%s)",
                        ANONYMIZED_TOPIC,
                        msg.error(),
                    )
                else:
                    logger.error("Kafka error on %s: %s", ANONYMIZED_TOPIC, msg.error())
                continue

            _handle_message(msg.value())
    except Exception:
        logger.exception("Anonymize-training Kafka consumer crashed.")
    finally:
        consumer.close()


def _handle_message(raw_value: Optional[bytes]) -> None:
    """Parse và dispatch một Kafka message."""
    if not raw_value:
        logger.warning("Empty Kafka payload on topic: %s", ANONYMIZED_TOPIC)
        return

    try:
        text = raw_value.decode("utf-8").strip()
    except UnicodeDecodeError:
        logger.warning("Non-UTF-8 Kafka payload, skipped.")
        return

    if not text:
        return

    try:
        payload: Dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON on %s: %s (preview: %r)", ANONYMIZED_TOPIC, exc.msg, text[:200])
        return

    # Chỉ xử lý event đúng loại và trạng thái success
    if payload.get("event_type") != "DATA_ANNONIMIZING_COMPLETED":
        logger.debug("Ignored event_type=%s", payload.get("event_type"))
        return
    if payload.get("status") != "success":
        logger.warning("Skipped non-success anonymize event: status=%s", payload.get("status"))
        return

    version_id: Optional[str] = payload.get("version_id")
    # Lưu ý: anonymize-svc dùng key "annonimize_file_paths" (typo intentional)
    anon_paths: List[str] = payload.get("annonimize_file_paths", [])

    if not version_id:
        logger.error("Missing version_id in anonymize event, skipped.")
        return
    if not anon_paths:
        logger.warning("No annonimize_file_paths for version_id=%s, skipped.", version_id)
        return

    logger.info(
        "Processing DATA_ANNONIMIZING_COMPLETED: version_id=%s, file_count=%d",
        version_id,
        len(anon_paths),
    )
    _process_anonymized_event(version_id, anon_paths)


def _process_anonymized_event(version_id: str, anon_paths: List[str]) -> None:
    """Download từng file anonymized, train model và upload lên model-zone."""
    client = get_minio_client()
    ensure_bucket(client, MODEL_BUCKET)

    for file_path in anon_paths:
        # file_path có dạng: "anonymize-zone/<version_id>/<stem>_anon_k5.parquet"
        if not file_path.startswith(f"{ANONYMIZE_BUCKET}/"):
            logger.warning("Skipping file with unexpected bucket prefix: %s", file_path)
            continue

        # Tách bucket và object key
        object_key = file_path[len(f"{ANONYMIZE_BUCKET}/"):]  # "<version_id>/<filename>"
        file_stem = Path(object_key).stem  # vd: "adult_anon_k5"

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            temp_path = tmp.name

        try:
            logger.info("Downloading %s/%s …", ANONYMIZE_BUCKET, object_key)
            download_parquet_from_minio(client, ANONYMIZE_BUCKET, object_key, temp_path)

            # Upload key trong model-zone: "<version_id>/anon/<stem>.joblib"
            model_key = f"{version_id}/anon/{file_stem}.joblib"
            logger.info("Training model for %s → model-zone/%s …", file_path, model_key)

            train_and_save_model(
                parquet_path=temp_path,
                version_id=version_id,
                model_key=model_key,
            )
            logger.info(
                "Model trained and uploaded: model-zone/%s (version_id=%s)",
                model_key,
                version_id,
            )

        except Exception:
            logger.exception(
                "Failed to train model for anonymized file: %s (version_id=%s)",
                file_path,
                version_id,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)
