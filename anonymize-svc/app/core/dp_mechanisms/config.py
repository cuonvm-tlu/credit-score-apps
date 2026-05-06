"""
Configuration for Differential Privacy mechanisms in the project.

Provides default settings, presets, and configuration management
for DP mechanisms used in the preprocessing service.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Predefined privacy levels."""
    VERY_HIGH = 0.01      # epsilon = 0.01 (very strong privacy, large noise)
    HIGH = 0.1            # epsilon = 0.1 (strong privacy, significant noise)
    MEDIUM = 0.5          # epsilon = 0.5 (balance between privacy and utility)
    LOW = 1.0             # epsilon = 1.0 (weak privacy, minimal noise)
    VERY_LOW = 5.0        # epsilon = 5.0 (minimal privacy, very small noise)


@dataclass
class DPConfig:
    """Differential Privacy configuration."""
    
    # Basic DP settings
    enabled: bool = True
    privacy_level: PrivacyLevel = PrivacyLevel.MEDIUM
    
    # Epsilon budgets
    total_epsilon: float = 0.5
    epsilon_for_laplace: float = 0.3
    epsilon_for_exponential: float = 0.1
    epsilon_for_above_threshold: float = 0.1
    
    # Mechanism settings
    use_laplace: bool = True
    use_exponential: bool = True
    use_above_threshold: bool = False
    use_adaptive_allocation: bool = True
    
    # Composition
    strict_composition: bool = True  # Enforce epsilon budget limits
    max_queries: Optional[int] = None
    
    # Data settings
    numerical_sensitivity: Dict[str, float] = field(default_factory=lambda: {
        "age": 80,          # Max age difference
        "salary": 100000,   # Max salary difference
        "hours": 100,       # Max hours worked
    })
    
    clipping_ranges: Dict[str, tuple] = field(default_factory=lambda: {
        "age": (0, 100),
        "salary": (0, 500000),
        "hours": (0, 168),
    })
    
    # Logging and monitoring
    log_level: str = "INFO"
    track_budget: bool = True
    raise_on_budget_exceed: bool = True
    
    # Advanced settings
    delta: float = 0.0  # 0 for pure DP, > 0 for (eps, delta)-DP
    use_composition_accounting: bool = False
    composition_type: str = "sequential"  # "sequential" or "adaptive"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.total_epsilon <= 0:
            raise ValueError("total_epsilon must be positive")
        
        if not (0 <= self.delta < 1):
            raise ValueError("delta must be in [0, 1)")
        
        # Set epsilon from privacy level
        if isinstance(self.privacy_level, PrivacyLevel):
            self.total_epsilon = self.privacy_level.value
    
    def set_privacy_level(self, level: PrivacyLevel):
        """Set privacy level and update epsilon."""
        self.privacy_level = level
        self.total_epsilon = level.value
        logger.info(f"Privacy level set to {level.name} (epsilon={self.total_epsilon})")
    
    def get_mechanism_epsilon(self, mechanism: str) -> float:
        """Get epsilon budget for specific mechanism."""
        mapping = {
            "laplace": self.epsilon_for_laplace,
            "exponential": self.epsilon_for_exponential,
            "above_threshold": self.epsilon_for_above_threshold,
        }
        return mapping.get(mechanism, 0)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "privacy_level": self.privacy_level.name,
            "total_epsilon": self.total_epsilon,
            "epsilon_for_laplace": self.epsilon_for_laplace,
            "epsilon_for_exponential": self.epsilon_for_exponential,
            "epsilon_for_above_threshold": self.epsilon_for_above_threshold,
            "use_laplace": self.use_laplace,
            "use_exponential": self.use_exponential,
            "use_above_threshold": self.use_above_threshold,
            "use_adaptive_allocation": self.use_adaptive_allocation,
            "delta": self.delta,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class AnonymizationDPConfig:
    """Configuration for combining anonymization with DP."""
    
    # Anonymization settings
    k_value: int = 5
    l_value: int = 2
    
    # DP settings after anonymization
    apply_dp_after_anonymization: bool = True
    dp_epsilon: float = 0.3
    
    # Attribute settings
    quasi_identifiers: List[str] = field(default_factory=lambda: [
        "age", "occupation", "education", "marital_status"
    ])
    sensitive_attributes: List[str] = field(default_factory=lambda: [
        "income", "capital_gain", "capital_loss"
    ])
    
    # Post-processing
    generalize_numerical_qi: bool = True
    add_noise_to_sensitive: bool = True
    
    # Quality metrics
    preserve_utility: bool = True
    acceptable_info_loss: float = 0.3  # 30% information loss acceptable
    
    def __post_init__(self):
        """Validate configuration."""
        if self.k_value < 1:
            raise ValueError("k_value must be >= 1")
        if self.l_value < 1:
            raise ValueError("l_value must be >= 1")
        if self.dp_epsilon <= 0:
            raise ValueError("dp_epsilon must be positive")


# Preset configurations
class DPPresets:
    """Predefined DP configurations for common scenarios."""
    
    @staticmethod
    def high_privacy() -> DPConfig:
        """Maximum privacy setting."""
        config = DPConfig()
        config.set_privacy_level(PrivacyLevel.VERY_HIGH)
        config.use_adaptive_allocation = True
        config.strict_composition = True
        return config
    
    @staticmethod
    def balanced() -> DPConfig:
        """Balanced privacy-utility tradeoff."""
        config = DPConfig()
        config.set_privacy_level(PrivacyLevel.MEDIUM)
        config.epsilon_for_laplace = 0.3
        config.epsilon_for_exponential = 0.1
        config.epsilon_for_above_threshold = 0.1
        return config
    
    @staticmethod
    def utility_focused() -> DPConfig:
        """Prioritize data utility (weaker privacy)."""
        config = DPConfig()
        config.set_privacy_level(PrivacyLevel.LOW)
        config.use_adaptive_allocation = True
        return config
    
    @staticmethod
    def credit_scoring() -> DPConfig:
        """Configuration for credit scoring application."""
        config = DPConfig()
        config.set_privacy_level(PrivacyLevel.MEDIUM)
        config.numerical_sensitivity = {
            "age": 80,
            "annual_income": 100000,
            "credit_limit": 50000,
            "credit_utilization": 100,
        }
        config.use_adaptive_allocation = True
        return config
    
    @staticmethod
    def research_dataset() -> DPConfig:
        """Configuration for research dataset publication."""
        config = DPConfig()
        config.set_privacy_level(PrivacyLevel.HIGH)
        config.delta = 1e-6  # (eps, delta)-DP with small delta
        config.strict_composition = True
        config.track_budget = True
        return config


# Global configuration instance
_global_dp_config: Optional[DPConfig] = None


def get_dp_config() -> DPConfig:
    """Get global DP configuration."""
    global _global_dp_config
    if _global_dp_config is None:
        _global_dp_config = DPConfig()
    return _global_dp_config


def set_dp_config(config: DPConfig):
    """Set global DP configuration."""
    global _global_dp_config
    _global_dp_config = config
    logger.info(f"Global DP config updated: {config.to_dict()}")


def init_dp_config_from_preset(preset_name: str) -> DPConfig:
    """Initialize DP configuration from preset."""
    presets = {
        "high_privacy": DPPresets.high_privacy,
        "balanced": DPPresets.balanced,
        "utility_focused": DPPresets.utility_focused,
        "credit_scoring": DPPresets.credit_scoring,
        "research_dataset": DPPresets.research_dataset,
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    config = presets[preset_name]()
    set_dp_config(config)
    logger.info(f"Initialized DP config from preset: {preset_name}")
    return config


# Example usage and default initialization
if __name__ == "__main__":
    # High privacy preset
    config_high = DPPresets.high_privacy()
    print("High Privacy Config:")
    print(config_high.to_json())
    
    print("\n" + "="*70 + "\n")
    
    # Balanced preset
    config_balanced = DPPresets.balanced()
    print("Balanced Config:")
    print(config_balanced.to_json())
    
    print("\n" + "="*70 + "\n")
    
    # Credit scoring preset
    config_credit = DPPresets.credit_scoring()
    print("Credit Scoring Config:")
    print(config_credit.to_json())
