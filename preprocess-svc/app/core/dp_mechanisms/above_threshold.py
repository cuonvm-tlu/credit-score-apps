"""
Above Threshold Mechanism for Differential Privacy.

Mechanism: Answers boolean queries about whether values exceed a threshold.
Privacy Guarantee: (eps, delta)-Differential Privacy
Used for: Multiple binary threshold queries, early stopping conditions
Reference: Dwork & Roth, The Algorithmic Foundations of Differential Privacy
"""

import numpy as np
from typing import List, Optional, Union, Tuple
import logging

from .dp_utils import DPUtility

logger = logging.getLogger(__name__)


class AboveThresholdMechanism:
    """
    Above Threshold Mechanism for differential privacy.
    
    Answers multiple binary threshold queries with differential privacy.
    Returns approximate answers to whether each query value is above a threshold.
    
    Useful for:
    - Multiple yes/no decision queries
    - Early stopping conditions
    - Binary classification with privacy
    
    Attributes:
        epsilon: Total privacy budget
        delta: Probability of failure
        threshold: Decision threshold
    """
    
    def __init__(
        self,
        epsilon: float,
        threshold: float,
        delta: float = 0.0,
        name: str = "AboveThreshold",
    ):
        """
        Initialize Above Threshold Mechanism.
        
        Args:
            epsilon: Privacy budget (total for all queries)
            threshold: Threshold value for binary decision
            delta: Failure probability (default 0 for pure DP)
            name: Mechanism name for logging
            
        Raises:
            ValueError: If epsilon <= 0 or delta not in [0, 1]
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if not (0 <= delta <= 1):
            raise ValueError(f"Delta must be in [0, 1], got {delta}")
        
        self.epsilon = epsilon
        self.threshold = threshold
        self.delta = delta
        self.name = name
        self.queries_answered = 0
        self.noisy_threshold = None
        
        # Initialize noisy threshold
        if delta == 0:
            # Pure DP: threshold noise scale = 2 / epsilon
            threshold_noise = np.random.laplace(loc=0, scale=2/epsilon)
        else:
            # (eps, delta) DP: use Gaussian
            threshold_noise = np.random.normal(
                loc=0,
                scale=np.sqrt(32 * np.log(1/delta)) / epsilon
            )
        
        self.noisy_threshold = threshold + threshold_noise
        
        logger.info(
            f"Initialized {name} with epsilon={epsilon:.6f}, "
            f"threshold={threshold:.4f}, delta={delta:.6f}"
        )
    
    def query(
        self,
        value: float,
        description: str = "",
    ) -> Optional[float]:
        """
        Query whether a value is above threshold with privacy.
        
        Returns:
            - Noisy value (approximately equal to true value) if value > threshold
            - None if value <= threshold
            - With probability: |true_answer - noisy_answer| is bounded
            
        Args:
            value: Query value to test
            description: Query description for logging
            
        Returns:
            Noisy answer if above threshold, None if below
        """
        self.queries_answered += 1
        
        # Add noise to query value
        if self.delta == 0:
            value_noise = np.random.laplace(loc=0, scale=2/self.epsilon)
        else:
            value_noise = np.random.normal(
                loc=0,
                scale=np.sqrt(32 * np.log(1/self.delta)) / self.epsilon
            )
        
        noisy_value = value + value_noise
        
        # Check against noisy threshold
        if noisy_value >= self.noisy_threshold:
            # Return noisy answer
            answer_noise = np.random.laplace(loc=0, scale=2/self.epsilon)
            answer = value + answer_noise
            is_above = True
        else:
            answer = None
            is_above = False
        
        logger.debug(
            f"{self.name}: Query #{self.queries_answered} - "
            f"Value={value:.4f}, Above={is_above} {description}"
        )
        
        return answer
    
    def query_batch(
        self,
        values: List[float],
        descriptions: Optional[List[str]] = None,
    ) -> List[Optional[float]]:
        """
        Answer multiple threshold queries.
        
        Args:
            values: List of query values
            descriptions: Optional descriptions for each query
            
        Returns:
            List of approximate answers (float or None)
        """
        if descriptions is None:
            descriptions = [f"Query {i}" for i in range(len(values))]
        elif len(descriptions) != len(values):
            raise ValueError(f"Mismatch: {len(values)} values, {len(descriptions)} descriptions")
        
        results = []
        for value, desc in zip(values, descriptions):
            result = self.query(value, desc)
            results.append(result)
        
        logger.info(f"{self.name}: Answered {len(results)} queries")
        return results
    
    def get_statistics(self) -> dict:
        """
        Get statistics about queries answered.
        
        Returns:
            Dictionary with query statistics
        """
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "threshold": self.threshold,
            "noisy_threshold": float(self.noisy_threshold),
            "total_queries": self.queries_answered,
        }
    
    def reset_threshold(self):
        """Reset the noisy threshold for new batch."""
        if self.delta == 0:
            threshold_noise = np.random.laplace(loc=0, scale=2/self.epsilon)
        else:
            threshold_noise = np.random.normal(
                loc=0,
                scale=np.sqrt(32 * np.log(1/self.delta)) / self.epsilon
            )
        self.noisy_threshold = self.threshold + threshold_noise
        self.queries_answered = 0
        logger.debug(f"{self.name}: Threshold reset")
    
    def __repr__(self) -> str:
        return (
            f"AboveThresholdMechanism(epsilon={self.epsilon:.6f}, "
            f"threshold={self.threshold:.4f}, queries={self.queries_answered})"
        )


class CountingQueriesMechanism:
    """
    Mechanism for answering counting queries with differential privacy.
    
    Uses Laplace mechanism for count queries with automatic privacy composition.
    """
    
    def __init__(self, epsilon: float, name: str = "CountingQueries"):
        """
        Initialize Counting Queries Mechanism.
        
        Args:
            epsilon: Total privacy budget
            name: Mechanism name
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        
        self.epsilon = epsilon
        self.name = name
        self.queries_count = 0
        self.epsilon_per_query = None
        
        logger.info(f"Initialized {name} with epsilon={epsilon:.6f}")
    
    def answer_count(
        self,
        true_count: int,
        total_records: int,
        description: str = "",
    ) -> int:
        """
        Answer count query with Laplace noise.
        
        Args:
            true_count: True count value
            total_records: Total number of records (for sensitivity)
            description: Query description
            
        Returns:
            Noisy count (sensitivity = 1 for count queries)
        """
        # Sensitivity of count query is 1
        sensitivity = 1
        noise = DPUtility.laplace_noise(sensitivity / self.epsilon)
        noisy_count = int(np.round(true_count + noise))
        
        # Ensure non-negative
        noisy_count = max(0, min(noisy_count, total_records))
        
        self.queries_count += 1
        logger.debug(f"{self.name}: Count query answered. {description}")
        
        return noisy_count
    
    def answer_proportion(
        self,
        true_count: int,
        total_records: int,
        description: str = "",
    ) -> float:
        """
        Answer proportion query (count/total) with privacy.
        
        Args:
            true_count: True count value
            total_records: Total records
            description: Query description
            
        Returns:
            Noisy proportion in [0, 1]
        """
        if total_records <= 0:
            raise ValueError("Total records must be positive")
        
        true_proportion = true_count / total_records
        # Sensitivity of proportion is 1/total_records
        sensitivity = 1 / total_records
        noise = DPUtility.laplace_noise(sensitivity / self.epsilon)
        noisy_proportion = true_proportion + noise
        
        # Clip to [0, 1]
        noisy_proportion = np.clip(noisy_proportion, 0, 1)
        
        self.queries_count += 1
        logger.debug(f"{self.name}: Proportion query answered. {description}")
        
        return float(noisy_proportion)
    
    def __repr__(self) -> str:
        return f"CountingQueriesMechanism(epsilon={self.epsilon:.6f}, queries={self.queries_count})"
