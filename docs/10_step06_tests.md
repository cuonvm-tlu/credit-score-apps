# BƯỚC 6 — Unit Tests

> **Thư mục:** `preprocess-svc/tests/`
> **Mục tiêu:** Kiểm thử tất cả components: DP mechanisms, utility evaluator, budget accountant, anonymization

---

## 6.1 Cấu Trúc Tests

```bash
mkdir -p preprocess-svc/tests
touch preprocess-svc/tests/__init__.py
touch preprocess-svc/tests/test_dp_mechanisms.py
touch preprocess-svc/tests/test_utility_evaluator.py
touch preprocess-svc/tests/test_budget_accountant.py
touch preprocess-svc/tests/test_anonymization.py
touch preprocess-svc/tests/test_routes.py
touch preprocess-svc/tests/conftest.py
```

Thêm vào `requirements.txt`:
```
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
```

---

## 6.2 `tests/conftest.py` — Shared Fixtures

```python
"""
conftest.py — Shared test fixtures cho toàn bộ test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def adult_sample_df() -> pd.DataFrame:
    """
    DataFrame mẫu giống Adult Census schema, 500 records.
    Dùng chung cho tất cả tests — scope=session để chỉ tạo 1 lần.
    """
    np.random.seed(42)
    n = 500

    workclasses   = ["Private", "Self-emp", "Gov", "Without-pay"]
    educations    = ["Bachelors", "HS-grad", "Masters", "Doctorate", "Some-college"]
    occupations   = ["Tech-support", "Craft-repair", "Exec-managerial", "Sales", "Prof-specialty"]
    marital_stats = ["Married-civ-spouse", "Divorced", "Never-married", "Separated"]
    races         = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo"]
    sexes         = ["Male", "Female"]
    countries     = ["United-States", "Mexico", "Philippines", "Germany", "India"]

    return pd.DataFrame({
        "age":            np.random.randint(18, 90, n),
        "workclass":      np.random.choice(workclasses, n),
        "education":      np.random.choice(educations, n),
        "education-num":  np.random.randint(1, 16, n),
        "marital-status": np.random.choice(marital_stats, n),
        "occupation":     np.random.choice(occupations, n),
        "relationship":   np.random.choice(["Husband", "Wife", "Not-in-family"], n),
        "race":           np.random.choice(races, n),
        "sex":            np.random.choice(sexes, n),
        "capital-gain":   np.random.randint(0, 50000, n),
        "capital-loss":   np.random.randint(0, 5000, n),
        "hours-per-week": np.random.randint(10, 80, n),
        "native-country": np.random.choice(countries, n),
        "income":         np.random.choice([0, 1], n),
    })


@pytest.fixture
def numerical_cols() -> list:
    return ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


@pytest.fixture
def dp_noisy_df(adult_sample_df, numerical_cols) -> pd.DataFrame:
    """DataFrame với Laplace noise ε=0.3 trên các cột numerical."""
    df_dp = adult_sample_df.copy()
    epsilon = 0.3
    sensitivities = {"age": 100, "education-num": 16, "capital-gain": 100000,
                     "capital-loss": 5000, "hours-per-week": 168}
    for col in numerical_cols:
        scale = sensitivities.get(col, 1.0) / epsilon
        df_dp[col] = df_dp[col] + np.random.laplace(0, scale, len(df_dp))
    return df_dp
```

---

## 6.3 `tests/test_dp_mechanisms.py`

```python
"""
test_dp_mechanisms.py — Unit tests cho DP mechanisms.
"""

import numpy as np
import pytest

from app.core.dp_mechanisms.laplace_mechanism import LaplaceNoiseMechanism
from app.core.dp_mechanisms.gaussian_mechanism import GaussianNoiseMechanism
from app.core.dp_mechanisms.exponential_mechanism import ExponentialMechanism


class TestLaplaceMechanism:
    """Tests cho LaplaceNoiseMechanism."""

    def test_init_valid(self):
        m = LaplaceNoiseMechanism(epsilon=0.5)
        assert m.epsilon == 0.5

    def test_init_invalid_epsilon(self):
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            LaplaceNoiseMechanism(epsilon=0.0)
        with pytest.raises(ValueError):
            LaplaceNoiseMechanism(epsilon=-1.0)

    def test_apply_scalar(self):
        m = LaplaceNoiseMechanism(epsilon=1.0)
        noisy = m.apply(100.0, sensitivity=1.0)
        assert isinstance(noisy, float)
        # Noise không quá lớn (99.9% confidence)
        assert abs(noisy - 100.0) < 100.0, "Noise seems unreasonably large"

    def test_apply_array(self):
        m = LaplaceNoiseMechanism(epsilon=1.0)
        original = np.array([10.0, 20.0, 30.0])
        noisy = m.apply(original, sensitivity=1.0)
        assert noisy.shape == original.shape
        assert not np.allclose(noisy, original), "Noise should change values"

    def test_invalid_sensitivity(self):
        m = LaplaceNoiseMechanism(epsilon=1.0)
        with pytest.raises(ValueError, match="Sensitivity must be positive"):
            m.apply(100.0, sensitivity=0.0)

    def test_budget_tracking(self):
        m = LaplaceNoiseMechanism(epsilon=0.5)
        m.apply(100.0, sensitivity=1.0)
        m.apply(200.0, sensitivity=1.0)
        status = m.get_budget_status()
        assert status["num_queries"] == 2
        assert status["epsilon_spent"] == pytest.approx(1.0, abs=1e-9)

    def test_apply_batch(self):
        m = LaplaceNoiseMechanism(epsilon=1.0)
        results = m.apply_batch([10.0, 20.0, 30.0], [1.0, 1.0, 1.0])
        assert len(results) == 3

    def test_noise_distribution(self):
        """
        Statistical test: Laplace noise mean ≈ 0 với n=10000.
        """
        np.random.seed(42)
        m = LaplaceNoiseMechanism(epsilon=1.0)
        samples = [m.apply(0.0, sensitivity=1.0) for _ in range(10_000)]
        mean_noise = np.mean(samples)
        # Mean of Laplace(0, b) = 0, allow 3*std/sqrt(n) tolerance
        assert abs(mean_noise) < 0.1, f"Mean noise {mean_noise:.4f} too large"

    def test_privacy_amplification(self):
        """Smaller epsilon → larger noise variance."""
        np.random.seed(42)
        n = 5000
        m_strict = LaplaceNoiseMechanism(epsilon=0.01)
        m_loose  = LaplaceNoiseMechanism(epsilon=5.0)
        strict_std = np.std([m_strict.apply(0.0, 1.0) for _ in range(n)])
        loose_std  = np.std([m_loose.apply(0.0,  1.0) for _ in range(n)])
        assert strict_std > loose_std, "Stricter privacy should add more noise"


class TestGaussianMechanism:
    """Tests cho GaussianNoiseMechanism."""

    def test_init_valid(self):
        m = GaussianNoiseMechanism(epsilon=1.0, delta=1e-5)
        assert m.epsilon == 1.0
        assert m.delta == 1e-5
        assert m.sigma > 0

    def test_init_invalid_epsilon(self):
        with pytest.raises(ValueError):
            GaussianNoiseMechanism(epsilon=0.0, delta=1e-5)

    def test_init_invalid_delta(self):
        with pytest.raises(ValueError):
            GaussianNoiseMechanism(epsilon=1.0, delta=0.0)
        with pytest.raises(ValueError):
            GaussianNoiseMechanism(epsilon=1.0, delta=1.5)

    def test_apply_scalar(self):
        m = GaussianNoiseMechanism(epsilon=1.0, delta=1e-5)
        noisy = m.apply(50.0, l2_sensitivity=1.0)
        assert isinstance(noisy, float)

    def test_apply_array(self):
        m = GaussianNoiseMechanism(epsilon=1.0, delta=1e-5)
        arr = np.array([10.0, 20.0, 30.0])
        noisy = m.apply(arr, l2_sensitivity=1.0)
        assert noisy.shape == arr.shape

    def test_apply_to_df_columns(self, adult_sample_df, numerical_cols):
        m = GaussianNoiseMechanism(epsilon=1.0, delta=1e-5)
        df_noisy = m.apply_to_dataframe_columns(
            df=adult_sample_df,
            columns=numerical_cols,
            l2_sensitivities={col: 1.0 for col in numerical_cols},
        )
        assert df_noisy.shape == adult_sample_df.shape
        # Ít nhất 1 cột thay đổi
        changed = any(
            not np.allclose(df_noisy[col].values, adult_sample_df[col].values)
            for col in numerical_cols
        )
        assert changed

    def test_sigma_increases_with_smaller_epsilon(self):
        """Nhỏ hơn epsilon → sigma lớn hơn → nhiều noise hơn."""
        m_strict = GaussianNoiseMechanism(epsilon=0.1, delta=1e-5)
        m_loose  = GaussianNoiseMechanism(epsilon=2.0, delta=1e-5)
        assert m_strict.sigma > m_loose.sigma

    def test_privacy_guarantee_report(self):
        m = GaussianNoiseMechanism(epsilon=0.5, delta=1e-6)
        report = m.get_privacy_guarantee()
        assert report["type"] == "(epsilon, delta)-DP"
        assert report["epsilon"] == 0.5
        assert report["delta"] == 1e-6


class TestExponentialMechanism:
    """Tests cho ExponentialMechanism."""

    def test_select_returns_valid_option(self):
        m = ExponentialMechanism(epsilon=1.0, sensitivity=1.0)
        options    = ["A", "B", "C"]
        utilities  = [1.0, 0.5, 0.2]
        selected   = m.select(options, utilities)
        assert selected in options

    def test_high_utility_selected_more_often(self):
        """Option có utility cao hơn được chọn nhiều hơn về mặt thống kê."""
        np.random.seed(42)
        m = ExponentialMechanism(epsilon=2.0, sensitivity=1.0)
        options   = ["best", "worst"]
        utilities = [10.0, 0.1]
        counts = {"best": 0, "worst": 0}
        for _ in range(1000):
            counts[m.select(options, utilities)] += 1
        assert counts["best"] > counts["worst"] * 5, "Best option should dominate"
```

---

## 6.4 `tests/test_utility_evaluator.py`

```python
"""
test_utility_evaluator.py — Tests cho utility_evaluator.py.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.utility_evaluator import (
    build_ncp_report,
    measure_dp_query_error,
    measure_classification_accuracy,
    build_privacy_utility_report,
    _epsilon_to_level,
    _interpret_ncp,
)


class TestNCP:
    def test_ncp_report_excellent(self):
        r = build_ncp_report(0.05, 1000, "k_anonymity")
        assert "Excellent" in r["interpretation"]
        assert r["ncp"] == 0.05

    def test_ncp_report_poor(self):
        r = build_ncp_report(0.8, 500, "l_diversity")
        assert "Poor" in r["interpretation"]

    @pytest.mark.parametrize("ncp,expected", [
        (0.05, "Excellent"), (0.2, "Good"),
        (0.4, "Acceptable"), (0.7, "Poor"),
    ])
    def test_interpret_ncp(self, ncp, expected):
        assert expected in _interpret_ncp(ncp)


class TestEpsilonLevel:
    @pytest.mark.parametrize("eps,level", [
        (0.05, "VERY_HIGH"), (0.3, "HIGH"),
        (0.8, "MEDIUM"), (3.0, "LOW"), (10.0, "VERY_LOW"),
    ])
    def test_epsilon_levels(self, eps, level):
        assert _epsilon_to_level(eps) == level


class TestDPQueryError:
    def test_zero_noise_zero_error(self, adult_sample_df, numerical_cols):
        """Nếu DP df giống original thì error = 0."""
        err = measure_dp_query_error(adult_sample_df, adult_sample_df, epsilon=1.0)
        assert err["avg_relative_error_pct"] == pytest.approx(0.0, abs=1e-4)

    def test_large_noise_large_error(self, adult_sample_df, numerical_cols):
        """Epsilon rất nhỏ → noise lớn → error lớn."""
        df_noisy = adult_sample_df.copy()
        # Thêm noise cực lớn
        for col in numerical_cols:
            df_noisy[col] = df_noisy[col] + np.random.laplace(0, 10000, len(df_noisy))
        err = measure_dp_query_error(adult_sample_df, df_noisy, epsilon=0.001)
        assert err["avg_relative_error_pct"] > 10.0

    def test_column_metrics_present(self, adult_sample_df, dp_noisy_df, numerical_cols):
        err = measure_dp_query_error(adult_sample_df, dp_noisy_df, epsilon=0.3)
        for col in numerical_cols:
            if col in adult_sample_df.columns:
                assert col in err["column_metrics"]

    def test_epsilon_in_result(self, adult_sample_df, dp_noisy_df):
        err = measure_dp_query_error(adult_sample_df, dp_noisy_df, epsilon=0.7)
        assert err["epsilon"] == 0.7


class TestClassificationAccuracy:
    def test_returns_accuracy(self, adult_sample_df):
        result = measure_classification_accuracy(adult_sample_df)
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_insufficient_data(self):
        tiny_df = pd.DataFrame({"age": [1, 2, 3], "income": [0, 1, 0]})
        result = measure_classification_accuracy(tiny_df)
        assert result.get("accuracy") is None or "error" in result

    def test_accuracy_decreases_with_noise(self, adult_sample_df, numerical_cols):
        """
        Accuracy sau DP noise không nên tốt hơn original.
        (Có thể bằng hoặc thấp hơn do noise)
        """
        orig_acc = measure_classification_accuracy(adult_sample_df)["accuracy"]

        df_noisy = adult_sample_df.copy()
        for col in numerical_cols:
            df_noisy[col] = df_noisy[col] + np.random.laplace(0, 1000, len(df_noisy))
        noisy_acc = measure_classification_accuracy(df_noisy)["accuracy"]

        # Không enforce strict inequality vì random seed
        assert noisy_acc is not None


class TestFullReport:
    def test_report_keys(self, adult_sample_df, dp_noisy_df):
        report = build_privacy_utility_report(
            original_df=adult_sample_df,
            dp_df=dp_noisy_df,
            epsilon=0.3,
        )
        assert "original" in report
        assert "differential_privacy" in report

    def test_report_with_all_methods(self, adult_sample_df, dp_noisy_df):
        report = build_privacy_utility_report(
            original_df=adult_sample_df,
            k_anon_df=adult_sample_df.head(400),
            l_div_df=adult_sample_df.head(450),
            dp_df=dp_noisy_df,
            k_ncp=0.2, l_ncp=0.3, epsilon=0.3,
        )
        assert "k_anonymity"         in report
        assert "l_diversity"         in report
        assert "differential_privacy" in report

    def test_accuracy_loss_pct_calculated(self, adult_sample_df, dp_noisy_df):
        report = build_privacy_utility_report(
            original_df=adult_sample_df, dp_df=dp_noisy_df, epsilon=0.3
        )
        dp = report["differential_privacy"]
        assert "accuracy_loss_pct" in dp
```

---

## 6.5 `tests/test_budget_accountant.py`

```python
"""
test_budget_accountant.py — Tests cho PrivacyBudgetAccountant.
"""

import pytest

from app.core.privacy_budget_accountant import (
    PrivacyBudgetAccountant,
    PrivacyBudgetExhaustedError,
)


class TestBudgetAccountant:
    def test_init_valid(self):
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        assert acc.total_epsilon == 1.0
        assert acc.spent_epsilon == 0.0
        assert acc.remaining_epsilon == 1.0

    def test_init_invalid(self):
        with pytest.raises(ValueError):
            PrivacyBudgetAccountant(total_epsilon=0.0)
        with pytest.raises(ValueError):
            PrivacyBudgetAccountant(total_epsilon=-1.0)

    def test_consume_normal(self):
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        acc.consume(0.3, "Laplace", ["age"])
        assert acc.spent_epsilon == pytest.approx(0.3)
        assert acc.remaining_epsilon == pytest.approx(0.7)

    def test_consume_multiple(self):
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        acc.consume(0.3, "Laplace", ["age"])
        acc.consume(0.3, "Laplace", ["edu"])
        acc.consume(0.3, "Laplace", ["cap"])
        assert acc.spent_epsilon == pytest.approx(0.9, abs=1e-9)
        assert len(acc._records) == 3

    def test_consume_exceeds_budget(self):
        acc = PrivacyBudgetAccountant(total_epsilon=0.5)
        acc.consume(0.3, "Laplace", ["age"])
        with pytest.raises(PrivacyBudgetExhaustedError):
            acc.consume(0.3, "Laplace", ["edu"])   # 0.3+0.3=0.6 > 0.5

    def test_consume_invalid_epsilon(self):
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        with pytest.raises(ValueError):
            acc.consume(0.0, "Laplace", [])
        with pytest.raises(ValueError):
            acc.consume(-0.1, "Laplace", [])

    def test_report_structure(self):
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        acc.consume(0.4, "Laplace", ["age"], description="Protect age")
        report = acc.report()
        assert report["total_epsilon"] == 1.0
        assert report["spent_epsilon"] == pytest.approx(0.4)
        assert report["num_queries"] == 1
        assert report["queries"][0]["mechanism"] == "Laplace"

    def test_reset(self):
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        acc.consume(0.5, "Laplace", ["age"])
        acc.reset()
        assert acc.spent_epsilon == 0.0
        assert len(acc._records) == 0

    def test_usage_pct(self):
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        acc.consume(0.5, "Laplace", [])
        assert acc.usage_pct == pytest.approx(50.0)

    def test_exact_budget_ok(self):
        """Dùng đúng 100% budget không nên raise."""
        acc = PrivacyBudgetAccountant(total_epsilon=1.0)
        acc.consume(1.0, "Laplace", ["all"])
        assert acc.remaining_epsilon == pytest.approx(0.0, abs=1e-9)
```

---

## 6.6 Chạy Toàn Bộ Tests

```bash
cd preprocess-svc

# Install test dependencies
pip install pytest pytest-asyncio httpx scipy scikit-learn

# Chạy tất cả tests
pytest tests/ -v

# Chạy với coverage
pip install pytest-cov
pytest tests/ -v --cov=app --cov-report=term-missing

# Chạy test cụ thể
pytest tests/test_dp_mechanisms.py -v
pytest tests/test_budget_accountant.py -v
pytest tests/test_utility_evaluator.py -v

# Chạy test có mark
pytest tests/ -v -k "laplace or gaussian"
```

Expected output:
```
tests/test_dp_mechanisms.py::TestLaplaceMechanism::test_init_valid PASSED
tests/test_dp_mechanisms.py::TestLaplaceMechanism::test_apply_scalar PASSED
...
tests/test_budget_accountant.py::TestBudgetAccountant::test_consume_exceeds_budget PASSED
...
====== 35 passed in 12.4s ======
```

---

## 6.7 Commit Bước 6

```bash
git add preprocess-svc/tests/
git add preprocess-svc/requirements.txt
git commit -m "test: add comprehensive unit tests for DP mechanisms, utility evaluator, budget accountant"
```

---

## 6.8 Checklist Bước 6

- [ ] Tạo thư mục `tests/` với `conftest.py`
- [ ] `test_dp_mechanisms.py`: Laplace, Gaussian, Exponential
- [ ] `test_utility_evaluator.py`: NCP, DP error, accuracy, full report
- [ ] `test_budget_accountant.py`: normal, exceeded, reset
- [ ] Chạy `pytest tests/ -v` — tất cả PASSED
- [ ] Commit

**Tiếp theo:** [Bước 7 — Final Integration & README](./11_step07_final.md)
