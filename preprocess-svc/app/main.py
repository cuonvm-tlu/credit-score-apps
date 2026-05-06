from fastapi import FastAPI

from app.api.routes import router
from app.core.kafka_anonymize_worker import (
    start_anonymization_consumer,
    stop_anonymization_consumer,
)
from app.core.minio_client import init_minio
from app.core.spark_session import stop_spark_session


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Credit Scoring Preprocessing Service",
        description="Preprocess Adult Census Income files and manage landing/clean zones in MinIO.",
    )

    @app.on_event("startup")
    def startup_event() -> None:
        init_minio()
        start_anonymization_consumer()

    @app.on_event("shutdown")
    def shutdown_event() -> None:
        stop_anonymization_consumer()
        stop_spark_session()

    app.include_router(router)
    return app


app = create_app()
