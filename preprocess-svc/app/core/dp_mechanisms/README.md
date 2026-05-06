"""
README for Differential Privacy (DP) Mechanisms

This module provides comprehensive Differential Privacy implementations for the credit-score-apps project.
Based on the implementation from: https://github.com/mbrg/differential-privacy

## Overview

Differential Privacy is a mathematical framework for quantifying and protecting privacy in data analysis.
Instead of removing identifying information (anonymization), DP adds carefully calibrated noise to query results
to guarantee that individual privacy is protected regardless of what adversary knows about the data.

## Key Components

### 1. Laplace Mechanism (`laplace_mechanism.py`)

**Purpose**: Add noise to numerical query results

**Privacy Guarantee**: (epsilon, 0)-Differential Privacy (pure DP)

**Usage**:
```python
from app.core.dp_mechanisms import LaplaceNoiseMechanism

mechanism = LaplaceNoiseMechanism(epsilon=0.1)

# Add noise to a single query result
noisy_result = mechanism.apply(
    query_result=42.5,
    sensitivity=1.0,  # Max change for 1-record change
    description="Average age"
)

# Add noise to multiple queries
results = mechanism.apply_batch(
    query_results=[100, 200, 150],
    sensitivities=[1.0, 1.0, 1.0],
    descriptions=["Count above 50k", "Count below 25k", "Average salary"]
)
```

**Key Parameters**:
- `epsilon`: Privacy budget (smaller = more private, more noise)
- `sensitivity`: How much query output changes if one record is removed
- Scale of Laplace noise = sensitivity / epsilon

**Accuracy Guarantee**:
With probability 1 - beta: |true_result - noisy_result| <= (sensitivity/epsilon) * log(k/beta)
where k is output dimension

---

### 2. Exponential Mechanism (`exponential_mechanism.py`)

**Purpose**: Select best option from candidates with privacy guarantee

**Privacy Guarantee**: (epsilon, 0)-Differential Privacy

**Usage**:
```python
from app.core.dp_mechanisms import ExponentialMechanism

mechanism = ExponentialMechanism(
    epsilon=0.1,
    sensitivity=1.0  # Max utility difference between adjacent records
)

# Select best attribute to generalize
attributes = ["age_range", "occupation", "education_level"]
utilities = [0.95, 0.80, 0.75]  # Quality scores

selected = mechanism.select(attributes, utilities)
# Output: "age_range" (with high probability, but DP-protected)

# Get top-3 selections with probabilities
top_3 = mechanism.get_top_k_probabilities(attributes, utilities, k=3)
```

**Use Cases**:
- Select which QI (quasi-identifier) attribute to generalize first in Mondrian
- Choose optimal generalization hierarchy branch
- Binary classification decisions

---

### 3. Above Threshold Mechanism (`above_threshold.py`)

**Purpose**: Answer multiple threshold queries

**Privacy Guarantee**: (epsilon, delta)-Differential Privacy

**Usage**:
```python
from app.core.dp_mechanisms import AboveThresholdMechanism

mechanism = AboveThresholdMechanism(
    epsilon=0.1,
    threshold=50000,  # Threshold for high income
    delta=1e-6
)

# Query whether value is above threshold
result = mechanism.query(75000)  # Returns noisy approximate value
result = mechanism.query(30000)  # Returns None (below threshold)

# Batch queries
values = [60000, 40000, 55000, 35000]
results = mechanism.query_batch(values)
```

---

### 4. DP-Anonymization Integration (`dp_anonymization_integration.py`)

**Purpose**: Combine Differential Privacy with k-anonymity and l-diversity

**Strategy**: Defense-in-depth
1. Apply anonymization (Mondrian for k-anonymity/l-diversity)
2. Apply DP post-processing for statistical privacy guarantee

**Usage**:
```python
from app.core.dp_mechanisms import DPAnonymizationIntegration

integration = DPAnonymizationIntegration(epsilon=0.5)

# Apply DP to numerical QI columns after anonymization
df_protected = integration.apply_dp_to_numerical_columns(
    df=anonymized_df,
    numerical_columns=["age_range", "salary"],
    sensitivities={"age_range": 10, "salary": 5000}
)

# Apply targeted DP to sensitive attribute
df_final = integration.apply_dp_to_sensitive_attribute(
    df=df_protected,
    sensitive_col="income",
    epsilon=0.1
)
```

---

## Integration with Anonymization

### Workflow: Mondrian + DP

```python
from app.core.dp_mechanisms import DPAnonymizationIntegration
from app.core.anonymize_k_anonymity import anonymize_cleaned_adult_k_anonymity

# Step 1: Apply k-anonymity (Mondrian)
anonymized_data = anonymize_cleaned_adult_k_anonymity(...)

# Step 2: Apply DP protection
dp_integration = DPAnonymizationIntegration(epsilon=0.3)
final_data = dp_integration.anonymize_with_dp(
    df=anonymized_data,
    qi_columns=["age", "occupation", "education"],
    sensitive_col="income",
    apply_dp=True,
    epsilon_for_dp=0.3
)
```

---

## Privacy Budget Allocation

### Epsilon (ε) Budget Management

**Total Budget**: Usually between 0.1 and 10
- ε < 0.1: Very strong privacy, more noise
- ε ≈ 1: Reasonable privacy-utility tradeoff
- ε > 5: Weaker privacy, less noise

**Allocation Strategy**:
```python
# Total epsilon = 1.0 (moderate privacy)
total_epsilon = 1.0

# Allocate across phases
anonymization_epsilon = 0.6  # k-anonymity phase
dp_epsilon = 0.4              # DP noise phase

# Further allocate DP epsilon
laplace_epsilon = 0.3
sensitive_attr_epsilon = 0.1
```

---

## Technical Details

### Sensitivity Calculation

Sensitivity is the maximum change in query output when one record is removed:

```python
# Example: Count query
# Sensitivity = 1 (removing a record changes count by ≤ 1)

# Example: Average age
# Sensitivity = max_age - min_age (age range)
# If data is clipped to [0, 100], sensitivity = 100

# Example: Proportion
# Sensitivity = 1/n where n = total records
```

### Composition Rules

Multiple queries on same data consumes more epsilon:

```python
# Sequential composition (worst case):
# Total epsilon = eps1 + eps2 + eps3 + ...

# Better: Adaptive composition (in practice)
# Total epsilon ≈ sqrt(k * eps1²) where k = number of queries
```

---

## Best Practices

1. **Plan epsilon budget upfront**: Allocate total epsilon before queries
2. **Minimize queries**: Each query consumes epsilon budget
3. **Use adaptive allocation**: Allocate epsilon dynamically based on query importance
4. **Combine mechanisms**: Use Mondrian + DP for defense-in-depth
5. **Monitor budget**: Track epsilon spending during processing
6. **Document assumptions**: Record sensitivity calculations and epsilon allocation

---

## Evaluation Metrics

### Privacy-Utility Tradeoff

```python
from app.core.dp_mechanisms import DPUtility

# Accuracy guarantee with 99.999% confidence
accuracy = DPUtility.compute_accuracy_guarantee(
    sensitivity=1.0,
    epsilon=0.1,
    dimension=1,
    beta=1e-5  # 99.999%
)
# Result: larger epsilon = smaller range = more accurate
```

---

## Comparison: Anonymization vs DP

| Aspect | k-anonymity + l-diversity | Differential Privacy |
|--------|---------------------------|----------------------|
| **Guarantee** | Structural (group-level) | Statistical (individual-level) |
| **Attack** | Record linkage, attribute inference | Any statistical analysis |
| **Utility Loss** | Generalization (coarse groups) | Noise (small random errors) |
| **Composability** | Limited | Strong (composable guarantees) |
| **Best For** | Traditional privacy | Strong privacy with queries |

**Recommendation**: Combine both!
- Use Mondrian for k-anonymity (structural privacy)
- Add DP post-processing for statistical privacy guarantee

---

## References

1. Dwork, C., & Roth, A. (2014). "The algorithmic foundations of differential privacy."
2. https://github.com/mbrg/differential-privacy - Reference implementation
3. https://en.wikipedia.org/wiki/Differential_privacy

---

## File Structure

```
dp_mechanisms/
├── __init__.py                          # Package exports
├── dp_utils.py                          # Utility functions and helpers
├── laplace_mechanism.py                 # Laplace noise mechanism
├── exponential_mechanism.py             # Exponential selection mechanism
├── above_threshold.py                   # Above threshold queries + counting
├── dp_anonymization_integration.py      # Integration with anonymization
└── README.md                            # This file
```

---

## Example Usage in Routes

```python
from fastapi import APIRouter
from app.core.dp_mechanisms import DPAnonymizationIntegration, LaplaceNoiseMechanism

# In your API route
@app.post("/anonymize-with-dp")
async def anonymize_with_dp(files: List[UploadFile]):
    # ... load and clean data ...
    
    # Apply k-anonymity
    anonymized_data = spark_clean_and_upload(...)
    
    # Apply DP
    dp_integration = DPAnonymizationIntegration(epsilon=0.3)
    final_data = dp_integration.anonymize_with_dp(
        df=anonymized_data,
        qi_columns=["age", "occupation"],
        sensitive_col="income",
        apply_dp=True
    )
    
    return {"status": "success", "privacy_epsilon": 0.3}
```

---

## Future Enhancements

1. **Composition accounting**: Track epsilon spending across functions
2. **Adaptive allocation**: Dynamically allocate epsilon based on data distribution
3. **Delta accounting**: Support (epsilon, delta) DP better
4. **Gaussian mechanism**: Add for better accuracy with delta
5. **Query logs**: Track all DP queries and epsilon usage
