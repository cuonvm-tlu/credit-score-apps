import logging
import signal
import sys
from threading import Thread

from app.core.kafka_consumer import consume_messages
from app.core.kafka_anonymize_consumer import (
    start_anonymize_training_consumer,
    stop_anonymize_training_consumer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for the Analytic Service.

    Chạy song song 2 Kafka consumers:
      1. ``data-cleaned-topic``    – train model từ dữ liệu đã clean (preprocess-svc)
      2. ``data-anonymized-topic`` – train model từ dữ liệu đã anonymize (anonymize-svc)
    """
    logger.info("Starting Analytic Service...")

    # ── Consumer 2: data-anonymized-topic (chạy nền) ──────────────────────────
    start_anonymize_training_consumer()

    # ── Consumer 1: data-cleaned-topic (chạy trên main thread, block ở đây) ──
    # Graceful shutdown khi nhận SIGINT/SIGTERM
    def _shutdown(signum, frame):
        logger.info("Shutdown signal received, stopping consumers...")
        stop_anonymize_training_consumer()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("Both Kafka consumers are running. Press Ctrl+C to stop.")
    consume_messages()  # blocks until KeyboardInterrupt


if __name__ == "__main__":
    main()