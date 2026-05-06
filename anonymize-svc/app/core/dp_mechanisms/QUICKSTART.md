"""
Quick Start Guide for Differential Privacy Integration

This file provides practical instructions to get started with DP in the project.
"""

# ============================================================================
# QUICK START: DIFFERENTIAL PRIVACY IN CREDIT-SCORE-APPS
# ============================================================================

## 1. BASIC USAGE - ADD NOISE TO NUMERICAL DATA

```python
from app.core.dp_mechanisms import LaplaceNoiseMechanism

# Create mechanism with privacy budget epsilon=0.1
mechanism = LaplaceNoiseMechanism(epsilon=0.1)

# Add noise to a query result (e.g., average salary)
true_salary = 50000
noisy_salary = mechanism.apply(
    query_result=true_salary,
    sensitivity=100,  # Max salary change per record
    description="Average employee salary"
)

print(f"True: ${true_salary}, Noisy: ${noisy_salary:.2f}")
```

---

## 2. SELECT BEST ATTRIBUTE WITH PRIVACY

```python
from app.core.dp_mechanisms import ExponentialMechanism

# Create mechanism
mechanism = ExponentialMechanism(epsilon=0.1, sensitivity=1.0)

# Select which attribute to generalize with privacy guarantee
attributes = ["age", "occupation", "education"]
quality_scores = [0.95, 0.85, 0.75]  # How good each generalization is

best_attr = mechanism.select(attributes, quality_scores)
print(f"Selected to generalize: {best_attr}")  # Privacy-protected choice
```

---

## 3. PROTECT MULTIPLE COUNTING QUERIES

```python
from app.core.dp_mechanisms import CountingQueriesMechanism

mechanism = CountingQueriesMechanism(epsilon=0.3)

# Query 1: How many people earn > 50k?
noisy_count_1 = mechanism.answer_count(
    true_count=250,
    total_records=1000,
    description="High-income count"
)
print(f"High-income records (noisy): {noisy_count_1}")

# Query 2: What proportion are high-income?
noisy_prop = mechanism.answer_proportion(
    true_count=250,
    total_records=1000,
    description="High-income proportion"
)
print(f"High-income proportion (noisy): {noisy_prop:.3f}")
```

---

## 4. COMBINE WITH ANONYMIZATION (MONDRIAN + DP)

```python
from app.core.dp_mechanisms import DPAnonymizationIntegration, DPPresets
import pandas as pd

# After Mondrian anonymization, add DP protection
anonymized_df = pd.DataFrame({
    'age_range': ['20-30', '30-40', '40-50'],
    'occupation': ['Engineer', 'Manager', 'Technician'],
    'income': [45000, 55000, 65000],
})

# Apply DP
integration = DPAnonymizationIntegration(epsilon=0.3)
df_final = integration.apply_dp_to_numerical_columns(
    df=anonymized_df,
    numerical_columns=['income'],
    sensitivities={'income': 10000}
)

print("DP-protected data:")
print(df_final)
```

---

## 5. USE PRIVACY PRESETS

```python
from app.core.dp_mechanisms import DPPresets, init_dp_config_from_preset

# Initialize from preset
config = init_dp_config_from_preset("high_privacy")
# Options: "high_privacy", "balanced", "utility_focused", "credit_scoring", "research_dataset"

print(f"Privacy level: {config.privacy_level.name}")
print(f"Total epsilon: {config.total_epsilon}")
```

---

## 6. ADAPTIVE EPSILON ALLOCATION

```python
from app.core.dp_mechanisms import AdaptiveLaplaceNoiseMechanism

# Total budget for 5 queries
adaptive = AdaptiveLaplaceNoiseMechanism(total_epsilon=0.5, num_queries=5)

# Each query automatically gets 0.1 epsilon
noisy_1 = adaptive.apply(100, sensitivity=1, description="Query 1")
noisy_2 = adaptive.apply(200, sensitivity=1, description="Query 2")
noisy_3 = adaptive.apply(150, sensitivity=1, description="Query 3")
```

---

## 7. PRIVACY-UTILITY TRADEOFF ANALYSIS

```python
from app.core.dp_mechanisms import DPUtility

# Understand accuracy guarantees for different epsilon values
for epsilon in [0.1, 0.5, 1.0, 5.0]:
    accuracy = DPUtility.compute_accuracy_guarantee(
        sensitivity=1.0,
        epsilon=epsilon,
        dimension=1,
        beta=1e-5  # 99.999% confidence
    )
    print(f"Epsilon={epsilon}: Max error ±{accuracy:.2f} (99.999% confidence)")

# Output:
# Epsilon=0.1: Max error ±69.08 (99.999% confidence)
# Epsilon=0.5: Max error ±13.82 (99.999% confidence)
# Epsilon=1.0: Max error ±6.91 (99.999% confidence)
# Epsilon=5.0: Max error ±1.38 (99.999% confidence)
```

---

## 8. THRESHOLD-BASED QUERIES

```python
from app.core.dp_mechanisms import AboveThresholdMechanism

# Query: Is income above 50k?
mechanism = AboveThresholdMechanism(
    epsilon=0.1,
    threshold=50000,
    delta=0.0  # Pure DP
)

# Test with different values
incomes = [75000, 45000, 60000, 30000]
for income in incomes:
    result = mechanism.query(income)
    status = "Above threshold (noisy)" if result else "Below threshold"
    print(f"Income ${income}: {status}")
```

---

## KEY CONCEPTS TO REMEMBER

### Epsilon (ε)
- **Privacy budget**: How much privacy you "spend" on queries
- **Smaller ε = Stronger privacy = More noise**
- **Typical range**: 0.1 (high privacy) to 1.0 (weak privacy)

### Sensitivity
- **Maximum change in query output** when one record is removed
- Example: For count query, sensitivity = 1 (removing one person changes count by ≤1)
- Example: For average age, sensitivity = max_age - min_age

### Differential Privacy
- **Individual-level privacy**: Protects against any statistical adversary
- **Composition**: Multiple queries increase epsilon spending
- **Guarantee**: (ε, 0) for pure DP, (ε, δ) for approximate DP

### Defense in Depth
- **Anonymization (Mondrian)**: Structural privacy via grouping
- **Differential Privacy**: Statistical privacy via noise
- **Combined**: Both work together for stronger protection

---

## COMMON USE CASES IN THIS PROJECT

### 1. Credit Scoring with Privacy
```python
# Anonymize credit data with k-anonymity
# Add DP noise to income/credit metrics
# Result: Both structural and statistical privacy
```

### 2. Data Publishing
```python
# Use high epsilon (0.1 or less) for research
# Ensures strong privacy guarantees
# Accept more utility loss for stronger privacy
```

### 3. Internal Analytics
```python
# Use balanced epsilon (0.5) for business decisions
# Good balance between privacy and utility
# Minimal noise, reasonable privacy
```

---

## TESTING YOUR IMPLEMENTATION

```bash
# Run all DP tests
cd app/core/dp_mechanisms
python tests.py

# Run examples
python examples.py
```

---

## INTEGRATION WITH EXISTING CODE

### In routes.py:
```python
from app.core.dp_mechanisms import DPAnonymizationIntegration

@router.post("/anonymize-with-dp")
async def anonymize_with_dp(files: List[UploadFile]):
    # ... load and clean data ...
    
    # Apply anonymization
    anonymized_data = spark_clean_and_upload(...)
    
    # Apply DP
    integration = DPAnonymizationIntegration(epsilon=0.3)
    final_data = integration.anonymize_with_dp(
        df=anonymized_data,
        qi_columns=["age", "occupation"],
        sensitive_col="income"
    )
    
    return {"success": True, "privacy_epsilon": 0.3}
```

---

## TROUBLESHOOTING

### "Epsilon budget exceeded"
- Solution: Reduce number of queries or allocate more epsilon
- Or use adaptive allocation to prioritize important queries

### "Error: Sensitivity must be positive"
- Solution: Ensure sensitivity value is > 0
- Example: For age data, sensitivity = max_age - min_age

### "Results too noisy"
- Solution: Increase epsilon (weaker privacy) or reduce number of queries
- Trade-off: More privacy = More noise

---

## NEXT STEPS

1. **Read Documentation**
   - `README.md`: Comprehensive DP overview
   - `IMPLEMENTATION_SUMMARY.md`: Technical details

2. **Run Examples**
   - `examples.py`: 8 practical examples
   - Shows each mechanism in action

3. **Review Tests**
   - `tests.py`: Unit tests for all mechanisms
   - Validates functionality

4. **Integrate**
   - Modify `routes.py` to use DP mechanisms
   - Add DP parameters to API endpoints
   - Test end-to-end privacy protection

5. **Monitor**
   - Track epsilon spending
   - Log all DP operations
   - Document privacy assumptions

---

## ADDITIONAL RESOURCES

- GitHub Reference: https://github.com/mbrg/differential-privacy
- Wikipedia: https://en.wikipedia.org/wiki/Differential_privacy
- Paper: Dwork & Roth (2014) - "The Algorithmic Foundations of Differential Privacy"

---

## QUICK REFERENCE

```python
# Import all mechanisms
from app.core.dp_mechanisms import (
    LaplaceNoiseMechanism,
    ExponentialMechanism,
    AboveThresholdMechanism,
    CountingQueriesMechanism,
    DPAnonymizationIntegration,
    DPUtility,
    DPPresets,
)

# Initialize with preset
from app.core.dp_mechanisms import init_dp_config_from_preset
config = init_dp_config_from_preset("balanced")  # or "high_privacy", "credit_scoring"

# Create mechanism
mechanism = LaplaceNoiseMechanism(epsilon=0.5)

# Use mechanism
noisy_result = mechanism.apply(query_result, sensitivity=1.0)
```

---

**Questions?** Check the comprehensive README.md in the dp_mechanisms folder!
