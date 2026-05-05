# BƯỚC 5 — Gaussian Mechanism & Cập Nhật Event Schema

> **File tạo mới:** `preprocess-svc/app/core/dp_mechanisms/gaussian_mechanism.py`
> **Mục tiêu:** Thêm (ε,δ)-DP bằng Gaussian noise + cập nhật Kafka event schema đầy đủ hơn

---

## 5.1 Tại Sao Cần Gaussian Mechanism?

| | Laplace | Gaussian |
|---|---|---|
| Privacy type | (ε, 0)-DP | (ε, δ)-DP |
| Noise phân phối | Laplace(0, Δ/ε) | N(0, σ²) |
| Độ chính xác | Thấp hơn khi ε nhỏ | Cao hơn với cùng ε |
| Dùng khi | Count queries, aggregates | ML gradient, mean estimation |
| Sensitivity | L1 sensitivity | L2 sensitivity |

Gaussian phù hợp hơn khi:
- Có nhiều queries liên tiếp (composition tốt hơn với Rényi DP)
- Cần accuracy cao hơn với cùng mức privacy
- Làm việc với vector/gradient (federated learning)

---

## 5.2 Code `gaussian_mechanism.py`

```python
"""
gaussian_mechanism.py
Gaussian Mechanism cho (epsilon, delta)-Differential Privacy.

Privacy Guarantee: (epsilon, delta)-DP
Noise scale: sigma = sqrt(2 * ln(1.25/delta)) * L2_sensitivity / epsilon

Reference: Dwork & Roth (2014), Theorem A.1
"""

import logging
import math
from typing import Union, Optional

import numpy as np

logger = logging.getLogger(__name__)


class GaussianNoiseMechanism:
    """
    Gaussian Mechanism: thêm Gaussian noise cho (ε,δ)-DP.

    Attributes:
        epsilon: Privacy parameter (> 0)
        delta:   Privacy failure probability (0 < delta << 1)
        name:    Mechanism name for logging

    Usage:
        mechanism = GaussianNoiseMechanism(epsilon=1.0, delta=1e-5)
        noisy_mean = mechanism.apply(true_mean, l2_sensitivity=0.01)
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        name: str = "GaussianNoise",
    ):
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0,1), got {delta}")

        self.epsilon = epsilon
        self.delta   = delta
        self.name    = name
        self.sigma   = self._compute_sigma(epsilon, delta)

        logger.info(
            f"Initialized {name}: epsilon={epsilon:.4f}, delta={delta:.2e}, sigma={self.sigma:.4f}"
        )

    @staticmethod
    def _compute_sigma(epsilon: float, delta: float) -> float:
        """
        Compute Gaussian noise scale.
        sigma = sqrt(2 * ln(1.25/delta)) / epsilon
        (sensitivity=1 normalized; caller multiplies by actual sensitivity)
        """
        return math.sqrt(2 * math.log(1.25 / delta)) / epsilon

    def apply(
        self,
        query_result: Union[float, np.ndarray],
        l2_sensitivity: float,
        description: str = "",
    ) -> Union[float, np.ndarray]:
        """
        Thêm Gaussian noise vào query result.

        Args:
            query_result:    True query output (scalar hoặc array)
            l2_sensitivity:  L2 sensitivity của query
                             (max L2-norm change khi remove 1 record)
            description:     Logging description

        Returns:
            Noisy query result với cùng shape

        Raises:
            ValueError: Nếu l2_sensitivity <= 0
        """
        if l2_sensitivity <= 0:
            raise ValueError(f"l2_sensitivity must be > 0, got {l2_sensitivity}")

        sigma_actual = self.sigma * l2_sensitivity
        noise        = np.random.normal(0, sigma_actual, size=np.shape(query_result))
        noisy_result = query_result + noise

        acc = self._accuracy_bound(l2_sensitivity)
        logger.debug(
            f"{self.name} [{description}]: sigma={sigma_actual:.4f}, "
            f"accuracy (99%)=±{acc:.4f}"
        )
        return noisy_result

    def _accuracy_bound(self, l2_sensitivity: float, confidence: float = 0.99) -> float:
        """
        Upper bound on |true - noisy| với xác suất >= confidence.
        Bound = sigma * sqrt(2) * erfinv(confidence)
        """
        from scipy.special import erfinv
        sigma_actual = self.sigma * l2_sensitivity
        z = math.sqrt(2) * erfinv(confidence)
        return sigma_actual * z

    def apply_to_dataframe_columns(
        self,
        df: "pd.DataFrame",
        columns: list,
        l2_sensitivities: dict,
    ) -> "pd.DataFrame":
        """
        Apply Gaussian noise đến nhiều cột của DataFrame.

        Args:
            df:               pandas DataFrame
            columns:          List tên cột cần protect
            l2_sensitivities: Dict {col: l2_sensitivity}

        Returns:
            DataFrame mới với Gaussian noise ở các cột chỉ định
        """
        import pandas as pd
        df_noisy = df.copy()

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skip")
                continue

            sens = l2_sensitivities.get(col, 1.0)
            noisy_vals = self.apply(
                df_noisy[col].values.astype(float),
                l2_sensitivity=sens,
                description=col,
            )
            df_noisy[col] = noisy_vals
            logger.info(f"Applied Gaussian noise to column '{col}' (L2_sens={sens})")

        return df_noisy

    def get_privacy_guarantee(self) -> dict:
        return {
            "type":    "(epsilon, delta)-DP",
            "epsilon": self.epsilon,
            "delta":   self.delta,
            "sigma":   round(self.sigma, 6),
            "mechanism": "Gaussian",
        }

    def __repr__(self) -> str:
        return (
            f"GaussianNoiseMechanism("
            f"epsilon={self.epsilon:.4f}, delta={self.delta:.2e}, sigma={self.sigma:.4f})"
        )
```

---

## 5.3 Thêm Gaussian vào `dp_mechanisms/__init__.py`

```python
# dp_mechanisms/__init__.py — thêm export:
from .gaussian_mechanism import GaussianNoiseMechanism

__all__ = [
    "LaplaceNoiseMechanism",
    "AdaptiveLaplaceNoiseMechanism",
    "ExponentialMechanism",
    "AboveThresholdMechanism",
    "DPAnonymizationIntegration",
    "DPUtility",
    "GaussianNoiseMechanism",   # ← THÊM
]
```

---

## 5.4 Thêm API Endpoint `/anonymize` Với Tùy Chọn Mechanism

Thêm endpoint mới vào `routes.py` cho phép client chọn mechanism:

```python
# routes.py — thêm endpoint mới:
from pydantic import BaseModel

class AnonymizeRequest(BaseModel):
    version_id:  str
    clean_key:   str
    k:           int   = 10
    l:           int   = 2
    epsilon:     float = 0.3
    delta:       float = 1e-5
    mechanism:   str   = "laplace"   # "laplace" hoặc "gaussian"


@router.post("/anonymize")
async def anonymize_existing(req: AnonymizeRequest) -> dict:
    """
    Áp dụng k-anonymity, l-diversity và DP lên file đã clean.
    Cho phép tùy chọn mechanism (laplace/gaussian).
    """
    client = get_minio_client()
    results = {}

    # K-Anonymity
    k_path, k_ncp = anonymize_cleaned_adult_k_anonymity_and_upload(
        client=client, clean_bucket="clean-zone",
        clean_object_key=req.clean_key, k=req.k,
    )
    results["k_anonymity"] = {"path": k_path, "ncp": k_ncp}

    # L-Diversity
    l_path, l_ncp = anonymize_cleaned_adult_l_diversity_and_upload(
        client=client, clean_bucket="clean-zone",
        clean_object_key=req.clean_key, l_value=req.l,
    )
    results["l_diversity"] = {"path": l_path, "ncp": l_ncp}

    # DP — chọn mechanism
    if req.mechanism == "gaussian":
        from app.core.dp_mechanisms.gaussian_mechanism import GaussianNoiseMechanism
        # Load clean data
        clean_df = _load_parquet_from_minio(client, f"clean-zone/{req.clean_key}")
        gauss = GaussianNoiseMechanism(epsilon=req.epsilon, delta=req.delta)
        dp_df = gauss.apply_to_dataframe_columns(
            df=clean_df,
            columns=["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
            l2_sensitivities={
                "age": 1.0, "education-num": 1.0,
                "capital-gain": 100.0, "capital-loss": 10.0, "hours-per-week": 1.0,
            },
        )
        # Upload gaussian result
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        dp_df.to_parquet(tmp.name, index=False)
        eps_str = f"{req.epsilon:.2f}".replace(".", "_")
        dp_key  = req.clean_key.replace("_clean.parquet", f"_dp_gaussian_e{eps_str}.parquet")
        client.put_object(
            Bucket="clean-zone", Key=dp_key,
            Body=open(tmp.name, "rb"),
            ContentLength=Path(tmp.name).stat().st_size,
            ContentType="application/octet-stream",
        )
        results["differential_privacy"] = {
            "mechanism": "gaussian",
            "epsilon": req.epsilon,
            "delta": req.delta,
            "path": f"clean-zone/{dp_key}",
        }
    else:
        dp_path = apply_dp_protection_and_upload(
            client=client, clean_bucket="clean-zone",
            clean_object_key=req.clean_key, epsilon=req.epsilon,
        )
        results["differential_privacy"] = {
            "mechanism": "laplace",
            "epsilon": req.epsilon,
            "path": dp_path,
        }

    return {"status": "ok", "version_id": req.version_id, "results": results}
```

---

## 5.5 Cập Nhật Kafka Event Schema (kafka_producer.py)

```python
# kafka_producer.py — cập nhật hàm send_cleaning_success_event:

def send_cleaning_success_event(
    version_folder: str,
    clean_file_paths: List[str],
    privacy_metadata: Optional[dict] = None,   # ← THÊM
) -> None:
    """
    Send Kafka event sau khi cleaning + anonymization hoàn tất.

    Args:
        version_folder:   Timestamp folder
        clean_file_paths: List MinIO paths
        privacy_metadata: Dict chứa k, l, epsilon, ncp, ... (optional)
    """
    try:
        producer = Producer(KAFKA_CONFIG)

        message = {
            "event_type":       "DATA_CLEANING_COMPLETED",
            "status":           "success",
            "version_id":       version_folder,
            "clean_file_paths": clean_file_paths,
            "schema_version":   "1.1",
            # Thêm metadata nếu có
            "privacy_metadata": privacy_metadata or {
                "k_anonymity": {"k": 10},
                "l_diversity": {"l": 2},
                "differential_privacy": {"epsilon": 0.3, "mechanism": "Laplace"},
            },
        }

        producer.produce(
            TOPIC,
            value=json.dumps(message).encode("utf-8"),
            callback=delivery_report,
        )
        producer.flush()
        logger.info(f"Sent Kafka event for version {version_folder} | {len(clean_file_paths)} files")

    except Exception as e:
        logger.error(f"Failed to send Kafka message: {e}")
```

---

## 5.6 Thêm `scipy` vào requirements.txt

```
# preprocess-svc/requirements.txt — THÊM:
scipy>=1.12.0
```

---

## 5.7 Commit Bước 5

```bash
git add preprocess-svc/app/core/dp_mechanisms/gaussian_mechanism.py
git add preprocess-svc/app/core/dp_mechanisms/__init__.py
git add preprocess-svc/app/api/routes.py
git add preprocess-svc/app/core/kafka_producer.py
git add preprocess-svc/requirements.txt
git commit -m "feat: add GaussianNoiseMechanism, /anonymize endpoint, update Kafka event schema"
```

---

## 5.8 Checklist Bước 5

- [ ] Tạo `gaussian_mechanism.py`
- [ ] Export `GaussianNoiseMechanism` từ `__init__.py`
- [ ] Thêm endpoint `POST /anonymize` với tùy chọn mechanism
- [ ] Cập nhật `kafka_producer.py` với `privacy_metadata`
- [ ] Thêm `scipy` vào requirements
- [ ] Commit

**Tiếp theo:** [Bước 6 — Tests](./10_step06_tests.md)
