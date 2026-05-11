from __future__ import annotations

import csv
import json
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, List, Optional


SERVICE_ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = SERVICE_ROOT / "logs" / "metrics"
DP_RUNS_PATH = METRICS_DIR / "dp_runs.jsonl"
DP_AGG_1M_PATH = METRICS_DIR / "dp_agg_1m.csv"

_WINDOW_SECONDS = 60
_LOCK = Lock()
_RECENT_RUNS: Deque[Dict[str, Any]] = deque()


def record_dp_run_metric(metric: Dict[str, Any]) -> None:
    """Persist run-level metric and update rolling 1-minute aggregate."""
    with _LOCK:
        _ensure_metrics_dir()
        _append_jsonl(DP_RUNS_PATH, metric)
        _update_recent_runs(metric)
        _append_agg_1m_row()


def _ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(file_path: Path, metric: Dict[str, Any]) -> None:
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(metric, ensure_ascii=False) + "\n")


def _update_recent_runs(metric: Dict[str, Any]) -> None:
    now = datetime.now(timezone.utc)
    _RECENT_RUNS.append(
        {
            "ts": now,
            "status": metric.get("status"),
            "row_count": metric.get("row_count"),
            "latency_total_ms": metric.get("latency_total_ms"),
        }
    )
    cutoff = now - timedelta(seconds=_WINDOW_SECONDS)
    while _RECENT_RUNS and _RECENT_RUNS[0]["ts"] < cutoff:
        _RECENT_RUNS.popleft()


def _append_agg_1m_row() -> None:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=_WINDOW_SECONDS)
    success_runs = [r for r in _RECENT_RUNS if r.get("status") == "success"]
    failed_runs = [r for r in _RECENT_RUNS if r.get("status") != "success"]
    rows_processed = sum(int(r.get("row_count") or 0) for r in success_runs)
    latencies = [float(r["latency_total_ms"]) for r in success_runs if r.get("latency_total_ms") is not None]

    row = {
        "window_start_utc": _fmt_utc(window_start),
        "window_end_utc": _fmt_utc(now),
        "files_success": len(success_runs),
        "files_failed": len(failed_runs),
        "rows_processed": rows_processed,
        "throughput_files_per_sec": round(len(success_runs) / _WINDOW_SECONDS, 6),
        "throughput_rows_per_sec": round(rows_processed / _WINDOW_SECONDS, 6),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
    }

    write_header = not DP_AGG_1M_PATH.exists()
    with DP_AGG_1M_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)

    rank = (p / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    value = sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    return round(value, 3)


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
