import os
import tempfile
from pathlib import Path
from threading import Lock

from pyspark.sql import SparkSession

_spark_session: SparkSession | None = None
_lock = Lock()


def get_spark_session(app_name: str, warehouse_suffix: str) -> SparkSession:
    global _spark_session
    if _spark_session is not None:
        return _spark_session
    with _lock:
        if _spark_session is None:
            warehouse_dir = Path(tempfile.gettempdir()) / warehouse_suffix
            warehouse_dir.mkdir(parents=True, exist_ok=True)
            warehouse_uri = str(warehouse_dir.resolve()).replace("\\", "/")
            master_url = os.getenv("SPARK_MASTER_URL", "local[*]")
            minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://127.0.0.1:9000")
            hadoop_aws_packages = os.getenv(
                "SPARK_JARS_PACKAGES",
                "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262",
            )

            builder = (
                SparkSession.builder.master(master_url)
                .appName(app_name)
                .config("spark.sql.warehouse.dir", warehouse_uri)
                .config("spark.hadoop.fs.s3a.endpoint", minio_endpoint)
                .config("spark.hadoop.fs.s3a.access.key", os.getenv("MINIO_ACCESS_KEY", "minioadmin"))
                .config("spark.hadoop.fs.s3a.secret.key", os.getenv("MINIO_SECRET_KEY", "minioadmin"))
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            )
            if hadoop_aws_packages:
                builder = builder.config("spark.jars.packages", hadoop_aws_packages)

            driver_host = os.getenv("SPARK_DRIVER_HOST")
            if driver_host:
                builder = (
                    builder.config("spark.driver.host", driver_host)
                    .config("spark.driver.bindAddress", "0.0.0.0")
                    .config("spark.driver.port", os.getenv("SPARK_DRIVER_PORT", "7078"))
                    .config("spark.blockManager.port", os.getenv("SPARK_BLOCKMANAGER_PORT", "7079"))
                )

            _spark_session = builder.getOrCreate()
            _spark_session.sparkContext.setLogLevel("WARN")
    return _spark_session


def stop_spark_session() -> None:
    global _spark_session
    with _lock:
        if _spark_session is not None:
            try:
                _spark_session.stop()
            except Exception:
                # Best-effort shutdown for dev reload mode.
                pass
            _spark_session = None
