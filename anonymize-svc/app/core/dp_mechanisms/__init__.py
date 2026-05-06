"""
Differential Privacy (DP) mechanisms for data anonymization.

This module implements various DP techniques for protecting sensitive data:
- Laplace Mechanism: Adds Laplace noise to numeric queries
- Exponential Mechanism: Selects from options based on utility scores
- Report Noisy Max: Returns argmax with noise for optimal attribute selection
- Above Threshold: Answers multiple binary threshold queries

Reference: mbrg/differential-privacy (https://github.com/mbrg/differential-privacy)
Based on: Dwork, Cynthia, and Aaron Roth. "The algorithmic foundations of differential privacy."
"""

from .laplace_mechanism import LaplaceNoiseMechanism, AdaptiveLaplaceNoiseMechanism
from .exponential_mechanism import ExponentialMechanism, UtilityAwareExponentialMechanism
from .above_threshold import AboveThresholdMechanism, CountingQueriesMechanism
from .dp_utils import DPUtility, SensitivityInfo
from .dp_anonymization_integration import DPAnonymizationIntegration, combine_anonymization_and_dp
from .config import (
    DPConfig,
    PrivacyLevel,
    AnonymizationDPConfig,
    DPPresets,
    get_dp_config,
    set_dp_config,
    init_dp_config_from_preset,
)

__all__ = [
    # Mechanisms
    "LaplaceNoiseMechanism",
    "AdaptiveLaplaceNoiseMechanism",
    "ExponentialMechanism",
    "UtilityAwareExponentialMechanism",
    "AboveThresholdMechanism",
    "CountingQueriesMechanism",
    # Utilities
    "DPUtility",
    "SensitivityInfo",
    # Integration
    "DPAnonymizationIntegration",
    "combine_anonymization_and_dp",
    # Configuration
    "DPConfig",
    "PrivacyLevel",
    "AnonymizationDPConfig",
    "DPPresets",
    "get_dp_config",
    "set_dp_config",
    "init_dp_config_from_preset",
]
