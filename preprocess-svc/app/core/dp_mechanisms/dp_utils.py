"""
Utility functions for Differential Privacy mechanisms.

Provides helper functions for sensitivity calculation, noise addition, and data validation.
"""

import numpy as np
from typing import Any, List, Tuple, Union
from dataclasses import dataclass


@dataclass
class SensitivityInfo:
    """Information about query sensitivity for DP mechanisms."""
    
    value: float
    dimension: int = 1
    description: str = ""
    
    def __repr__(self) -> str:
        return f"Sensitivity(value={self.value:.4f}, dim={self.dimension})"


class DPUtility:
    """Utility class for Differential Privacy operations."""
    
    @staticmethod
    def laplace_noise(scale: float, size: int = 1) -> Union[float, np.ndarray]:
        """
        Generate Laplace noise.
        
        Args:
            scale: Scale parameter (b) for Laplace distribution
            size: Number of noise samples to generate
            
        Returns:
            Laplace-distributed noise value(s) with loc=0, scale=b
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        return np.random.laplace(loc=0, scale=scale, size=size)
    
    @staticmethod
    def gaussian_noise(scale: float, size: int = 1) -> Union[float, np.ndarray]:
        """
        Generate Gaussian noise for (epsilon, delta) DP.
        
        Args:
            scale: Standard deviation for Gaussian distribution
            size: Number of noise samples to generate
            
        Returns:
            Gaussian-distributed noise value(s)
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        return np.random.normal(loc=0, scale=scale, size=size)
    
    @staticmethod
    def add_laplace_noise(
        data: Union[float, np.ndarray],
        sensitivity: float,
        epsilon: float,
    ) -> Union[float, np.ndarray]:
        """
        Add Laplace noise to query result for epsilon-DP.
        
        The scale of Laplace noise is set to sensitivity/epsilon,
        ensuring (epsilon, 0)-differential privacy.
        
        Args:
            data: Query result (scalar or array)
            sensitivity: Sensitivity of the query function
            epsilon: Privacy budget (smaller = more private)
            
        Returns:
            Noisy query result
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        
        scale = sensitivity / epsilon
        noise = DPUtility.laplace_noise(scale, size=np.asarray(data).shape)
        
        return data + noise
    
    @staticmethod
    def add_gaussian_noise(
        data: Union[float, np.ndarray],
        sensitivity: float,
        epsilon: float,
        delta: float,
    ) -> Union[float, np.ndarray]:
        """
        Add Gaussian noise for (epsilon, delta) DP.
        
        Uses the composition of Laplace + Gaussian for better accuracy.
        
        Args:
            data: Query result (scalar or array)
            sensitivity: Sensitivity of the query function
            epsilon: Privacy budget
            delta: Probabilistic privacy bound parameter
            
        Returns:
            Noisy query result
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if delta <= 0 or delta >= 1:
            raise ValueError(f"Delta must be in (0,1), got {delta}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        
        # Recommended scale for Gaussian DP
        scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = DPUtility.gaussian_noise(scale, size=np.asarray(data).shape)
        
        return data + noise
    
    @staticmethod
    def exponential_weights(
        utilities: Union[List[float], np.ndarray],
        epsilon: float,
        sensitivity: float,
    ) -> np.ndarray:
        """
        Compute exponential mechanism weights.
        
        For Exponential Mechanism, compute weights as:
            w_i ∝ exp(epsilon * utility_i / (2 * sensitivity))
        
        Args:
            utilities: Utility scores for each option
            epsilon: Privacy budget
            sensitivity: Sensitivity of the utility function
            
        Returns:
            Normalized probability distribution over options
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        
        utilities = np.asarray(utilities, dtype=float)
        
        # Compute exponential weights
        exponent = (epsilon / (2 * sensitivity)) * utilities
        
        # Prevent overflow
        exponent = exponent - np.max(exponent)
        weights = np.exp(exponent)
        
        # Normalize to probability distribution
        return weights / np.sum(weights)
    
    @staticmethod
    def compute_laplace_scale(
        sensitivity: float,
        epsilon: float,
    ) -> float:
        """
        Compute Laplace noise scale for epsilon-DP.
        
        Args:
            sensitivity: Query sensitivity
            epsilon: Privacy budget
            
        Returns:
            Scale parameter for Laplace distribution
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        
        return sensitivity / epsilon
    
    @staticmethod
    def compute_accuracy_guarantee(
        sensitivity: float,
        epsilon: float,
        dimension: int = 1,
        beta: float = 1e-5,
    ) -> float:
        """
        Compute accuracy guarantee for Laplace mechanism.
        
        For Laplace mechanism, with probability 1 - beta:
            |f(x) - result| <= (sensitivity/epsilon) * log(dim/beta)
        
        Args:
            sensitivity: Query sensitivity
            epsilon: Privacy budget
            dimension: Output dimension
            beta: Failure probability
            
        Returns:
            Maximum expected error with probability 1 - beta
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        
        return (sensitivity / epsilon) * np.log(dimension / beta)
    
    @staticmethod
    def verify_epsilon_budget(
        spent: float,
        budget: float,
        name: str = "Query",
    ) -> bool:
        """
        Check if privacy budget allows spending more.
        
        Args:
            spent: Epsilon already spent
            budget: Total epsilon budget
            name: Name for logging
            
        Returns:
            True if epsilon spent < budget
            
        Raises:
            ValueError if budget exceeded
        """
        if spent >= budget:
            raise ValueError(
                f"{name}: Privacy budget exceeded. Spent: {spent:.6f}, Budget: {budget:.6f}"
            )
        return True
    
    @staticmethod
    def clip_data(
        data: np.ndarray,
        lower_bound: float,
        upper_bound: float,
    ) -> np.ndarray:
        """
        Clip data to a bounded range (for bounded differential privacy).
        
        Args:
            data: Input data
            lower_bound: Lower clipping value
            upper_bound: Upper clipping value
            
        Returns:
            Clipped data
        """
        return np.clip(data, lower_bound, upper_bound)
    
    @staticmethod
    def normalize_sensitivity(
        sensitivity: float,
        data_range: float,
    ) -> float:
        """
        Normalize sensitivity based on data range.
        
        Args:
            sensitivity: Original sensitivity
            data_range: Range of data values (max - min)
            
        Returns:
            Normalized sensitivity as fraction of data range
        """
        if data_range <= 0:
            raise ValueError(f"Data range must be positive, got {data_range}")
        
        return sensitivity / data_range
