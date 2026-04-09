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
            _spark_session = (
                SparkSession.builder.master("local[*]")
                .appName(app_name)
                .config("spark.sql.warehouse.dir", warehouse_uri)
                .getOrCreate()
            )
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

