import logging

from fastapi import FastAPI

from app.api.routes import router
from app.core.kafka_anonymize_worker import (
    start_anonymization_consumer,
    stop_anonymization_consumer,
)
from app.core.minio_client import init_minio
from app.core.spark_session import stop_spark_session


def _configure_app_logging() -> None:
    """Make app.* INFO logs visible (uvicorn/root often leaves them below WARNING only)."""
    app_log = logging.getLogger("app")
    app_log.setLevel(logging.INFO)
    if app_log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"),
    )
    app_log.addHandler(handler)
    app_log.propagate = False


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Credit Scoring Anonymization Service",
        description="Consume cleaned-data Kafka events and run k-anonymity/l-diversity.",
    )

    @app.on_event("startup")
    def startup_event() -> None:
        _configure_app_logging()
        init_minio()
        start_anonymization_consumer()

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        stop_anonymization_consumer()
        stop_spark_session()

    app.include_router(router)
    return app


app = create_app()
