import io
import os
from datetime import datetime
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.anonymize_k_anonymity import anonymize_cleaned_adult_k_anonymity_and_upload
from app.core.anonymize_l_diversity import anonymize_cleaned_adult_l_diversity_and_upload
from app.core.cleaner import clean_and_upload
from app.core.spark_cleaner import spark_clean_and_upload
from app.core.kafka_producer import send_cleaning_success_event
from app.core.minio_client import ensure_bucket, get_minio_client

router = APIRouter()


def build_version_folder() -> str:
    """Build a timestamp-based folder name used for versioning."""
    # Avoid unsupported characters (e.g. ":") in MinIO object keys.
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


async def read_upload_file(upload_file: UploadFile) -> bytes:
    """Read the content of an UploadFile into bytes."""
    content = await upload_file.read()
    if content is None:
        raise HTTPException(status_code=400, detail=f"Unable to read {upload_file.filename}")
    return content


@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)) -> dict:
    """Upload multiple files, save to landing zone, and conditionally clean data files."""
    version_folder = build_version_folder()
    client = get_minio_client()

    ensure_bucket(client, "landing-zone")
    ensure_bucket(client, "clean-zone")

    landing_zone_paths: List[str] = []
    clean_zone_paths: List[str] = []

    for upload_file in files:
        filename = os.path.basename(upload_file.filename or "")
        if not filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

        object_key = f"{version_folder}/{filename}"
        file_bytes = await read_upload_file(upload_file)

        client.put_object(
            Bucket="landing-zone",
            Key=object_key,
            Body=io.BytesIO(file_bytes),
            ContentLength=len(file_bytes),
            ContentType=upload_file.content_type or "application/octet-stream",
        )
        landing_zone_paths.append(f"landing-zone/{object_key}")

        if filename in {"adult.data", "adult.test"} or filename.lower().endswith(".csv"):
            #Clean with Pandas
            # cleaned_path = clean_and_upload(
            #     client=client,
            #     source_bucket="landing-zone",
            #     source_key=object_key,
            #     clean_bucket="clean-zone",
            #     version_folder=version_folder,
            #     original_filename=filename,
            # )

            #Clean + generalize with Spark
            cleaned_path = spark_clean_and_upload(
                client=client,
                source_bucket="landing-zone",
                source_key=object_key,
                clean_bucket="clean-zone",
                version_folder=version_folder,
                original_filename=filename,
            )

            clean_zone_paths.append(cleaned_path)
            # Separate anonymization flow: clean -> anonymize
            clean_bucket, clean_key = cleaned_path.split("/", 1)
            anonymized_path = anonymize_cleaned_adult_k_anonymity_and_upload(
                client=client,
                clean_bucket=clean_bucket,
                clean_object_key=clean_key,
                k=10,
            )
            clean_zone_paths.append(anonymized_path)
            ldiv_path = anonymize_cleaned_adult_l_diversity_and_upload(
                client=client,
                clean_bucket=clean_bucket,
                clean_object_key=clean_key,
                l_value=2,
            )
            clean_zone_paths.append(ldiv_path)
            # Apply Differential Privacy protection
            dp_protected_path = apply_dp_protection_and_upload(
                client=client,
                clean_bucket=clean_bucket,
                clean_object_key=clean_key,
                epsilon=0.3,
            )
            clean_zone_paths.append(dp_protected_path)

    # Send Kafka event after successful cleaning
    if clean_zone_paths:
        send_cleaning_success_event(version_folder, clean_zone_paths)

    return {
        "message": "Upload completed",
        "landing_zone_paths": landing_zone_paths,
        "clean_zone_paths": clean_zone_paths,
    }
