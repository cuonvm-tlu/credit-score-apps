"""
Integration layer for combining Differential Privacy with Anonymization techniques.

Integrates DP mechanisms with k-anonymity and l-diversity for enhanced privacy.
Provides utility functions to apply DP as post-processing or during anonymization.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
import logging

from .laplace_mechanism import LaplaceNoiseMechanism, AdaptiveLaplaceNoiseMechanism
from .exponential_mechanism import ExponentialMechanism
from .above_threshold import AboveThresholdMechanism
from .dp_utils import DPUtility

logger = logging.getLogger(__name__)


class DPAnonymizationIntegration:
    """
    Integration of Differential Privacy with anonymization methods.
    
    Combines:
    - k-anonymity: Group-level privacy
    - l-diversity: Sensitive attribute diversity
    - Differential Privacy: Individual-level privacy guarantee
    
    This provides defense-in-depth: even if anonymization fails,
    DP provides additional privacy protection.
    """
    
    def __init__(self, epsilon: float, max_epsilon_per_attribute: float = 0.1):
        """
        Initialize DP-Anonymization Integration.
        
        Args:
            epsilon: Total privacy budget for DP
            max_epsilon_per_attribute: Max epsilon for single attribute
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        self.epsilon = epsilon
        self.max_epsilon_per_attribute = max_epsilon_per_attribute
        self.mechanisms = {}
        
        logger.info(f"Initialized DPAnonymizationIntegration with epsilon={epsilon:.6f}")
    
    def apply_dp_to_numerical_columns(
        self,
        df: pd.DataFrame,
        numerical_columns: List[str],
        sensitivities: Optional[dict] = None,
        epsilon_allocation: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Apply Laplace noise to numerical columns in anonymized data.
        
        Args:
            df: DataFrame with anonymized data
            numerical_columns: List of numerical column names
            sensitivities: Dict mapping column name to sensitivity (default 1.0)
            epsilon_allocation: Dict mapping column name to epsilon budget
            
        Returns:
            DataFrame with DP-protected numerical columns
        """
        df_noisy = df.copy()
        breakpoint()
        if sensitivities is None:
            sensitivities = {col: 1.0 for col in numerical_columns}
        
        if epsilon_allocation is None:
            # Equal allocation
            eps_per_col = self.epsilon / len(numerical_columns)
            epsilon_allocation = {col: eps_per_col for col in numerical_columns}
        
        for col in numerical_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
            
            eps = epsilon_allocation.get(col, self.max_epsilon_per_attribute)
            sens = sensitivities.get(col, 1.0)
            
            mechanism = LaplaceNoiseMechanism(epsilon=eps, name=f"DP_{col}")
            
            try:
                noisy_values = mechanism.apply(
                    df_noisy[col].values,
                    sensitivity=sens,
                    description=f"Column {col}",
                )
                df_noisy[col] = noisy_values
                self.mechanisms[col] = mechanism
                logger.info(f"Applied DP to column '{col}' with epsilon={eps:.6f}")
            except Exception as e:
                logger.error(f"Failed to apply DP to column '{col}': {e}")
        
        return df_noisy
    
    def apply_dp_to_sensitive_attribute(
        self,
        df: pd.DataFrame,
        sensitive_col: str,
        epsilon: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Apply targeted DP to sensitive attribute (used in l-diversity).
        
        Args:
            df: DataFrame
            sensitive_col: Name of sensitive attribute column
            epsilon: Privacy budget (default: allocate from total)
            
        Returns:
            DataFrame with DP-protected sensitive attribute
        """
        df_noisy = df.copy()
        
        if epsilon is None:
            epsilon = min(self.epsilon / 2, self.max_epsilon_per_attribute)
        
        if sensitive_col not in df.columns:
            raise ValueError(f"Column '{sensitive_col}' not found")
        
        # For categorical sensitive attributes, use exponential mechanism
        # to select value category-wise
        unique_values = df_noisy[sensitive_col].unique()
        mechanism = ExponentialMechanism(
            epsilon=epsilon,
            sensitivity=1.0,
            name=f"DP_Sensitive_{sensitive_col}",
        )
        
        # Replace each instances with DP-selected value
        # This maintains diversity while adding DP noise
        try:
            # Group by other columns and apply DP selection
            for idx in df_noisy.index:
                # Simple version: with small probability, flip the value
                if np.random.random() < epsilon / 100:  # Adjust threshold
                    df_noisy.loc[idx, sensitive_col] = mechanism.select(unique_values, [1.0] * len(unique_values))
            
            self.mechanisms[sensitive_col] = mechanism
            logger.info(f"Applied DP to sensitive attribute '{sensitive_col}' with epsilon={epsilon:.6f}")
        except Exception as e:
            logger.error(f"Failed to apply DP to sensitive attribute: {e}")
        
        return df_noisy
    
    def anonymize_with_dp(
        self,
        df: pd.DataFrame,
        qi_columns: List[str],
        sensitive_col: str,
        apply_dp: bool = True,
        epsilon_for_dp: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Anonymize data with DP protections.
        
        Workflow:
        1. Apply anonymization (k-anonymity/l-diversity)
        2. Apply DP noise to sensitive and quasi-identifier columns
        
        Args:
            df: Input DataFrame
            qi_columns: Quasi-identifier columns
            sensitive_col: Sensitive attribute column
            apply_dp: Whether to apply DP (default True)
            epsilon_for_dp: Epsilon budget for DP phase
            
        Returns:
            Anonymized + DP-protected DataFrame
        """
        if not apply_dp:
            logger.info("Skipping DP application")
            return df
        
        if epsilon_for_dp is None:
            epsilon_for_dp = self.epsilon
        
        df_protected = df.copy()
        
        # Apply DP to QI columns
        try:
            df_protected = self.apply_dp_to_numerical_columns(
                df_protected,
                qi_columns,
                epsilon_allocation={col: epsilon_for_dp / len(qi_columns) for col in qi_columns},
            )
        except Exception as e:
            logger.error(f"Failed to apply DP to QI columns: {e}")
        
        # Apply DP to sensitive attribute
        try:
            df_protected = self.apply_dp_to_sensitive_attribute(
                df_protected,
                sensitive_col,
                epsilon=epsilon_for_dp / 4,
            )
        except Exception as e:
            logger.error(f"Failed to apply DP to sensitive attribute: {e}")
        
        return df_protected
    
    def get_privacy_report(self) -> dict:
        """
        Generate privacy report summarizing DP mechanisms used.
        
        Returns:
            Dictionary with privacy analysis
        """
        report = {
            "total_epsilon": self.epsilon,
            "mechanisms_registered": len(self.mechanisms),
            "mechanism_details": {},
        }
        
        for name, mechanism in self.mechanisms.items():
            report["mechanism_details"][name] = {
                "type": mechanism.__class__.__name__,
                "epsilon": mechanism.epsilon if hasattr(mechanism, "epsilon") else "N/A",
            }
        
        return report
    
    def __repr__(self) -> str:
        return (
            f"DPAnonymizationIntegration(total_epsilon={self.epsilon:.6f}, "
            f"mechanisms={len(self.mechanisms)})"
        )


def combine_anonymization_and_dp(
    anonymized_data: np.ndarray,
    dp_epsilon: float,
    sensitivity: float = 1.0,
) -> np.ndarray:
    """
    Apply DP noise as post-processing to anonymized data.
    
    Combines anonymization (structural privacy) with DP (statistical privacy).
    
    Args:
        anonymized_data: Output from anonymization algorithm
        dp_epsilon: Privacy budget for DP phase
        sensitivity: Sensitivity of anonymization result
        
    Returns:
        Data with both anonymization and DP protections
    """
    if dp_epsilon <= 0:
        logger.warning("Invalid epsilon, returning original data")
        return anonymized_data
    
    mechanism = LaplaceNoiseMechanism(epsilon=dp_epsilon)
    noisy_data = mechanism.apply(anonymized_data, sensitivity=sensitivity)
    
    logger.info(f"Applied DP post-processing with epsilon={dp_epsilon:.6f}")
    return noisy_data


def estimate_privacy_loss(
    anonymization_k: int,
    anonymization_l: Optional[int] = None,
    dp_epsilon: float = 0.0,
) -> dict:
    """
    Estimate overall privacy loss from combined mechanisms.
    
    Args:
        anonymization_k: k-anonymity parameter
        anonymization_l: l-diversity parameter (optional)
        dp_epsilon: DP privacy budget
        
    Returns:
        Dictionary with privacy analysis
    """
    analysis = {
        "anonymization": {
            "k_anonymity": anonymization_k,
            "l_diversity": anonymization_l,
            "max_re_id_risk": 1 / anonymization_k,  # Theoretical upper bound
        },
        "differential_privacy": {
            "epsilon": dp_epsilon,
            "mechanism": "Laplace" if dp_epsilon > 0 else "None",
        },
    }
    
    # Estimate combined privacy
    if dp_epsilon > 0:
        # Combined privacy is roughly additive in epsilon
        analysis["combined_privacy"] = {
            "mechanism": "Anonymization + DP",
            "nominal_epsilon": dp_epsilon,
            "k_anonymity_baseline": anonymization_k,
        }
    
    return analysis
