"""
Laplace Mechanism for Differential Privacy.

Mechanism: Adds Laplace-distributed noise to numerical query results.
Privacy Guarantee: (epsilon, 0)-Differential Privacy
Reference: Dwork & Roth, The Algorithmic Foundations of Differential Privacy
"""

import numpy as np
from typing import Union, Optional, Callable
import logging

from .dp_utils import DPUtility, SensitivityInfo

logger = logging.getLogger(__name__)


class LaplaceNoiseMechanism:
    """
    Laplace Mechanism for differential privacy.
    
    Adds Laplace-distributed noise to query results, guaranteeing
    (epsilon, 0)-differential privacy.
    
    Attributes:
        epsilon: Privacy budget (smaller = stricter privacy)
        name: Mechanism name for logging
    """
    
    def __init__(self, epsilon: float, name: str = "LaplaceNoise"):
        """
        Initialize Laplace Mechanism.
        
        Args:
            epsilon: Privacy budget (must be > 0)
            name: Name for logging
            
        Raises:
            ValueError: If epsilon <= 0
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        self.epsilon = epsilon
        self.name = name
        self.epsilon_spent = 0.0
        logger.info(f"Initialized {name} with epsilon={epsilon:.6f}")
    
    def apply(
        self,
        query_result: Union[float, np.ndarray],
        sensitivity: float,
        description: str = "",
    ) -> Union[float, np.ndarray]:
        """
        Apply Laplace noise to query result.
        
        Args:
            query_result: True query result (scalar or array)
            sensitivity: Sensitivity of the query (max change for 1-record change)
            description: Query description for logging
            
        Returns:
            Noisy query result with same shape as input
            
        Raises:
            ValueError: If sensitivity <= 0
        """
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        
        # Update budget
        self.epsilon_spent += self.epsilon
        
        # Add Laplace noise
        noisy_result = DPUtility.add_laplace_noise(
            query_result,
            sensitivity=sensitivity,
            epsilon=self.epsilon,
        )
        
        # Compute accuracy guarantee
        dimension = 1 if isinstance(query_result, (int, float)) else len(query_result)
        accuracy = DPUtility.compute_accuracy_guarantee(
            sensitivity=sensitivity,
            epsilon=self.epsilon,
            dimension=dimension,
        )
        
        info = f"{description} | Sensitivity: {sensitivity:.4f}, Accuracy (99.999%): ±{accuracy:.4f}"
        logger.debug(f"{self.name}: Applied noise to {info}")
        
        return noisy_result
    
    def apply_batch(
        self,
        query_results: list,
        sensitivities: list,
        descriptions: Optional[list] = None,
    ) -> list:
        """
        Apply Laplace noise to multiple query results.
        
        Args:
            query_results: List of query results
            sensitivities: List of sensitivities (one per query)
            descriptions: Optional descriptions for logging
            
        Returns:
            List of noisy query results
            
        Raises:
            ValueError: If list lengths don't match
        """
        if len(query_results) != len(sensitivities):
            raise ValueError(
                f"Mismatch: {len(query_results)} results, {len(sensitivities)} sensitivities"
            )
        
        if descriptions is None:
            descriptions = [f"Query {i}" for i in range(len(query_results))]
        elif len(descriptions) != len(query_results):
            raise ValueError(
                f"Mismatch: {len(query_results)} results, {len(descriptions)} descriptions"
            )
        
        results = []
        for result, sens, desc in zip(query_results, sensitivities, descriptions):
            noisy = self.apply(result, sens, desc)
            results.append(noisy)
        
        logger.info(f"{self.name}: Applied noise to {len(results)} queries")
        return results
    
    def get_budget_status(self) -> dict:
        """
        Get privacy budget status.
        
        Returns:
            Dictionary with budget information
        """
        return {
            "epsilon_allocated": self.epsilon,
            "epsilon_spent": self.epsilon_spent,
            "epsilon_remaining": max(0, self.epsilon - self.epsilon_spent),
            "num_queries": int(self.epsilon_spent / self.epsilon),
        }
    
    def reset(self):
        """Reset the epsilon budget counter."""
        self.epsilon_spent = 0.0
        logger.info(f"{self.name}: Budget counter reset")
    
    def __repr__(self) -> str:
        return f"LaplaceNoiseMechanism(epsilon={self.epsilon:.6f}, spent={self.epsilon_spent:.6f})"


class AdaptiveLaplaceNoiseMechanism:
    """
    Adaptive Laplace Mechanism with progressive privacy budget allocation.
    
    Allows allocating privacy budget across multiple queries with automatic scaling.
    """
    
    def __init__(self, total_epsilon: float, num_queries: Optional[int] = None):
        """
        Initialize Adaptive Laplace Mechanism.
        
        Args:
            total_epsilon: Total privacy budget to allocate across queries
            num_queries: Expected number of queries (for equal allocation)
        """
        if total_epsilon <= 0:
            raise ValueError(f"Total epsilon must be positive, got {total_epsilon}")
        
        self.total_epsilon = total_epsilon
        self.num_queries = num_queries
        self.queries_executed = 0
        
        if num_queries is not None and num_queries > 0:
            self.epsilon_per_query = total_epsilon / num_queries
        else:
            self.epsilon_per_query = None
        
        logger.info(
            f"Initialized AdaptiveLaplaceNoiseMechanism with total_epsilon={total_epsilon:.6f}"
        )
    
    def apply(
        self,
        query_result: Union[float, np.ndarray],
        sensitivity: float,
        description: str = "",
    ) -> Union[float, np.ndarray]:
        """
        Apply adaptive Laplace noise.
        
        Args:
            query_result: Query result
            sensitivity: Query sensitivity
            description: Query description
            
        Returns:
            Noisy query result
        """
        self.queries_executed += 1
        
        if self.epsilon_per_query is None:
            # Dynamic allocation: remaining budget / remaining queries
            remaining_epsilon = self.total_epsilon
            epsilon_for_this = remaining_epsilon / (self.num_queries - self.queries_executed + 1) if self.num_queries else remaining_epsilon
        else:
            epsilon_for_this = self.epsilon_per_query
        
        mechanism = LaplaceNoiseMechanism(epsilon_for_this)
        return mechanism.apply(query_result, sensitivity, description)
    
    def __repr__(self) -> str:
        return (
            f"AdaptiveLaplaceNoiseMechanism(total_epsilon={self.total_epsilon:.6f}, "
            f"queries={self.queries_executed}/{self.num_queries})"
        )
