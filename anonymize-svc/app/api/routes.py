from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health() -> dict:
    """Simple health endpoint for anonymize worker service."""
    return {"status": "ok", "service": "anonymize-svc"}
