# DP Mechanism Metrics Specification (anonymize-svc)

## 1) Phạm vi tài liệu
Tài liệu này mô tả cách đo và lưu các chỉ số hiệu năng cho luồng DP mechanism trong `anonymize-svc`.
Mục tiêu:
- Đo được latency theo từng stage và end-to-end cho mỗi lần xử lý DP.
- Đo được throughput theo file và theo số dòng.
- Lưu dữ liệu theo format để vẽ dashboard sau mỗi lần service xử lý.

Phạm vi hiện tại: chỉ áp dụng cho DP mechanism (không bao gồm k-anonymity, l-diversity).

## 2) Định nghĩa metric
### 2.1 Latency
- `latency_total_ms`: thời gian tổng cho 1 DP run (1 profile trên 1 file).
- `latency_download_ms`: download parquet từ MinIO.
- `latency_read_parquet_ms`: đọc parquet vào pandas.
- `latency_dp_apply_ms`: áp noise + post-processing (clip).
- `latency_write_parquet_ms`: ghi parquet tạm local.
- `latency_upload_ms`: upload parquet đã DP lên MinIO.

Công thức tổng quát:

$$
latency_{stage} = t_{stage\_end} - t_{stage\_start}
$$

$$
latency_{total} = t_{run\_end} - t_{run\_start}
$$

Đơn vị: milliseconds (ms).

### 2.2 Throughput
- `throughput_files_per_sec`: số file DP xử lý thành công trong 1 cửa sổ thời gian.
- `throughput_rows_per_sec`: tổng số dòng đã xử lý DP trong 1 cửa sổ thời gian.

Công thức:

$$
throughput_{files/s} = \frac{N_{files}}{\Delta t}
$$

$$
throughput_{rows/s} = \frac{\sum rows}{\Delta t}
$$

Trong đó:
- `N_files`: số run thành công trong cửa sổ `\Delta t`.
- `sum rows`: tổng row_count của các run thành công trong cửa sổ `\Delta t`.

## 3) Đơn vị đo và nguyên tắc đo
- Dùng `time.perf_counter()` để tính duration (không dùng wall-clock để tính độ lệch).
- Dùng `datetime.utcnow()` chỉ để gán mốc thời gian sự kiện (timestamp).
- Mỗi DP profile trên mỗi file được xem là 1 `run` độc lập để tính latency.
- Nếu 1 message batch chạy 3 profile, tạo 3 bản ghi run riêng + 1 bản ghi tổng hợp batch.

## 4) Điểm đặt mốc trong luồng DP
Trong hàm xử lý DP (adapter):
1. `run_start`: trước khi download.
2. `download_start/end`: bao quanh `_download_object_to_temp`.
3. `read_start/end`: bao quanh `_read_parquet_with_pandas`.
4. `dp_apply_start/end`: bao quanh `_apply_dp_to_dataframe`.
5. `write_start/end`: bao quanh `to_parquet`.
6. `upload_start/end`: bao quanh `put_object`.
7. `run_end`: sau khi upload thành công hoặc khi fail.

## 5) Bộ trường metric cần lưu (run-level)
Mỗi run lưu 1 record với schema để xài được ngay cho dashboard:

```json
{
  "event_time_utc": "2026-05-11T01:23:45.123Z",
  "trace_id": "uuid-or-version-id",
  "service": "anonymize-svc",
  "component": "dp_mechanism",
  "profile_name": "baseline_adult",
  "epsilon": 0.3,
  "source_bucket": "clean-zone",
  "source_key": ".../adult_clean.parquet",
  "target_bucket": "anonymize-zone",
  "target_key": ".../adult_dp_baseline...parquet",
  "row_count": 30162,
  "column_count": 14,
  "status": "success",
  "error_type": null,
  "error_message": null,
  "latency_total_ms": 812.4,
  "latency_download_ms": 112.7,
  "latency_read_parquet_ms": 95.1,
  "latency_dp_apply_ms": 243.6,
  "latency_write_parquet_ms": 154.2,
  "latency_upload_ms": 198.8
}
```

Nếu fail:
- `status = "failed"`
- `error_type`, `error_message` phải có giá trị
- stage nào chưa chạy thì để `null`

## 6) Lưu trữ metric để trực quan hóa
### 6.1 Run-level storage (bắt buộc)
- File JSONL append-only:
  - Đề xuất: `anonymize-svc/logs/metrics/dp_runs.jsonl`
- Mỗi dòng là 1 JSON object theo schema run-level.
- Ưu điểm: dễ append, dễ parse bằng pandas, dễ ship sang ELK/Loki/ClickHouse.

### 6.2 Aggregate storage theo cửa sổ (khuyến nghị)
- Tạo job tổng hợp mỗi 1 phút (hoặc 5 phút) sang:
  - `anonymize-svc/logs/metrics/dp_agg_1m.csv`
- Trường gợi ý:
  - `window_start_utc`, `window_end_utc`
  - `files_success`, `files_failed`
  - `rows_processed`
  - `throughput_files_per_sec`, `throughput_rows_per_sec`
  - `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`

## 7) Cách tính throughput cho dashboard
Dashboard nên tính theo cửa sổ cuốn (rolling window), ví dụ 60s:

1. Lấy tập run thành công trong [now-60s, now].
2. Tính:

$$
TP_{files/s} = \frac{count(success\_runs)}{60}
$$

$$
TP_{rows/s} = \frac{sum(row\_count)}{60}
$$

3. Vẽ theo profile:
- baseline_adult
- privacy_strict
- utility_focused

4. Vẽ tổng tất cả profile để thấy sức tải tổng.

## 8) KPI để theo dõi sau mỗi lần xử lý
Tối thiểu cần hiển thị:
- Success rate = success_runs / total_runs
- Latency total p50/p95/p99
- Throughput files/s
- Throughput rows/s
- Top error_type (nếu có)

## 9) Quy ước naming
- Prefix metric key: `dp_`
- Timezone: UTC
- Đơn vị latency: ms
- Đơn vị throughput: per second

## 10) Kế hoạch triển khai (DP-first)
1. Thêm instrument timepoint trong adapter DP.
2. Ghi run-level metric vào JSONL sau mỗi run.
3. Tạo script tổng hợp 1m (cron hoặc scheduler nhẹ).
4. Dùng notebook/Grafana để trực quan p50/p95/p99 và throughput.

## 11) Ghi chú chất lượng đo
- Chạy warm-up trước benchmark để loại startup effect.
- Tách riêng benchmark profile epsilon (0.1 / 0.3 / 0.7).
- Nếu cần so sánh công bằng, giữ nguyên data size và số profile trong batch.
