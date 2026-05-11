# Current DP Mechanism Parameters

This file summarizes the Differential Privacy parameters currently used by the anonymization flow.

## Runtime Parameters In Use

These are the values currently used by `apply_dp_protection_and_upload()` and `DPAnonymizationIntegration`.

- epsilon: `0.3`
- mechanism: `Laplace`
- delta: `0.0`
- max_epsilon_per_attribute: `0.1`
- anonymize target bucket: `anonymize-zone`

## Budget Allocation

Defined in `config.py`:

- total_epsilon: `0.5`
- epsilon_for_laplace: `0.3`
- epsilon_for_exponential: `0.1`
- epsilon_for_above_threshold: `0.1`

## Privacy Level Presets

Defined in `PrivacyLevel`:

- VERY_HIGH: `0.01`
- HIGH: `0.1`
- MEDIUM: `0.5`
- LOW: `1.0`
- VERY_LOW: `5.0`

Note: the current adapter call is using `epsilon=0.3` directly, not the `MEDIUM` preset value `0.5`.

## Mechanism Flags In Config

Defined in `DPConfig`:

- enabled: `True`
- use_laplace: `True`
- use_exponential: `True`
- use_above_threshold: `False`
- use_adaptive_allocation: `True`
- strict_composition: `True`
- max_queries: `None`
- track_budget: `True`
- raise_on_budget_exceed: `True`
- use_composition_accounting: `False`
- composition_type: `sequential`

## Sensitivity Settings

Defined in `DPConfig.numerical_sensitivity`:

```python
{
    "age": 80,
    "salary": 100000,
    "hours": 100,
}
```

## Clipping Ranges

Defined in `DPConfig.clipping_ranges`:

```python
{
    "age": (0, 100),
    "salary": (0, 500000),
    "hours": (0, 168),
}
```

## Anonymization + DP Settings

Defined in `AnonymizationDPConfig`:

- k_value: `5`
- l_value: `2`
- apply_dp_after_anonymization: `True`
- dp_epsilon: `0.3`
- generalize_numerical_qi: `True`
- add_noise_to_sensitive: `True`
- preserve_utility: `True`
- acceptable_info_loss: `0.3`

## Attribute Groups

Quasi-identifiers:

```python
["age", "occupation", "education", "marital_status"]
```

Sensitive attributes:

```python
["income", "capital_gain", "capital_loss"]
```

## Source Files

- `app/core/dp_anonymization_adapter.py`
- `app/core/dp_mechanisms/config.py`
- `app/core/dp_mechanisms/dp_anonymization_integration.py`
