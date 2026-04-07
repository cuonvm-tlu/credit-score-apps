from fastapi import FastAPI

from app.api.routes import router
from app.core.minio_client import init_minio


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Credit Scoring Preprocessing Service",
        description="Preprocess Adult Census Income files and manage landing/clean zones in MinIO.",
    )

    @app.on_event("startup")
    def startup_event() -> None:
        init_minio()

    app.include_router(router)
    return app


app = create_app()
