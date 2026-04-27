"""
Exponential Mechanism for Differential Privacy.

Mechanism: Selects from options based on utility scores with exponential weighting.
Privacy Guarantee: (epsilon, 0)-Differential Privacy
Used for: Selecting best categorical choices, optimal attribute for generalization
Reference: Dwork & Roth, The Algorithmic Foundations of Differential Privacy
"""

import numpy as np
from typing import List, Union, Any, Optional, Tuple
import logging

from .dp_utils import DPUtility

logger = logging.getLogger(__name__)


class ExponentialMechanism:
    """
    Exponential Mechanism for differential privacy.
    
    Selects from a set of options based on utility scores, adding noise
    to ensure (epsilon, 0)-differential privacy.
    
    Useful for:
    - Selecting optimal attribute for generalization
    - Choosing best categorical value
    - Decision-making under privacy constraints
    
    Attributes:
        epsilon: Privacy budget
        sensitivity: Sensitivity of utility function
    """
    
    def __init__(
        self,
        epsilon: float,
        sensitivity: float,
        name: str = "ExponentialMechanism",
    ):
        """
        Initialize Exponential Mechanism.
        
        Args:
            epsilon: Privacy budget (must be > 0)
            sensitivity: Sensitivity of the utility function (max difference in utility)
            name: Mechanism name for logging
            
        Raises:
            ValueError: If epsilon <= 0 or sensitivity <= 0
        """
        if epsilon <= 0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if sensitivity <= 0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.name = name
        self.selections_count = 0
        
        logger.info(
            f"Initialized {name} with epsilon={epsilon:.6f}, sensitivity={sensitivity:.4f}"
        )
    
    def select(
        self,
        options: List[Any],
        utilities: Union[List[float], np.ndarray],
        return_probabilities: bool = False,
        description: str = "",
    ) -> Union[Any, Tuple[Any, float]]:
        """
        Select option from candidates based on utility with privacy guarantee.
        
        Args:
            options: List of options to choose from
            utilities: Utility score for each option (higher = better)
            return_probabilities: If True, return (selection, probability)
            description: Description for logging
            
        Returns:
            Selected option (or tuple of (option, probability) if return_probabilities=True)
            
        Raises:
            ValueError: If options and utilities have different lengths
        """
        options = list(options)
        utilities = np.asarray(utilities, dtype=float)
        
        if len(options) != len(utilities):
            raise ValueError(
                f"Mismatch: {len(options)} options, {len(utilities)} utilities"
            )
        
        if len(options) == 0:
            raise ValueError("Cannot select from empty options list")
        
        # Compute exponential weights
        weights = DPUtility.exponential_weights(
            utilities,
            epsilon=self.epsilon,
            sensitivity=self.sensitivity,
        )
        
        # Sample from distribution
        index = np.random.choice(len(options), p=weights)
        selected = options[index]
        probability = weights[index]
        
        self.selections_count += 1
        
        # Log
        top_3_indices = np.argsort(utilities)[-3:][::-1]
        logger.debug(
            f"{self.name}: Selected '{selected}' (utility={utilities[index]:.4f}) "
            f"from {len(options)} options. {description}"
        )
        
        if return_probabilities:
            return selected, float(probability)
        return selected
    
    def select_batch(
        self,
        batch_options: List[List[Any]],
        batch_utilities: List[Union[List[float], np.ndarray]],
        descriptions: Optional[List[str]] = None,
    ) -> List[Any]:
        """
        Select from multiple sets of options.
        
        Args:
            batch_options: List of option lists
            batch_utilities: List of utility lists
            descriptions: Optional descriptions for each selection
            
        Returns:
            List of selected options
        """
        if len(batch_options) != len(batch_utilities):
            raise ValueError(
                f"Mismatch: {len(batch_options)} option lists, {len(batch_utilities)} utility lists"
            )
        
        if descriptions is None:
            descriptions = [f"Selection {i}" for i in range(len(batch_options))]
        
        results = []
        for opts, utils, desc in zip(batch_options, batch_utilities, descriptions):
            selected = self.select(opts, utils, description=desc)
            results.append(selected)
        
        logger.info(f"{self.name}: Completed {len(results)} selections")
        return results
    
    def get_top_k_probabilities(
        self,
        options: List[Any],
        utilities: Union[List[float], np.ndarray],
        k: int = 3,
    ) -> List[Tuple[Any, float]]:
        """
        Get top-k options with their selection probabilities.
        
        Args:
            options: List of options
            utilities: Utility scores
            k: Number of top options to return
            
        Returns:
            List of (option, probability) tuples sorted by probability
        """
        options = list(options)
        utilities = np.asarray(utilities, dtype=float)
        
        if k > len(options):
            k = len(options)
        
        weights = DPUtility.exponential_weights(
            utilities,
            epsilon=self.epsilon,
            sensitivity=self.sensitivity,
        )
        
        top_indices = np.argsort(weights)[-k:][::-1]
        return [(options[i], float(weights[i])) for i in top_indices]
    
    def get_selection_statistics(self) -> dict:
        """
        Get statistics about selections made.
        
        Returns:
            Dictionary with selection statistics
        """
        return {
            "epsilon": self.epsilon,
            "sensitivity": self.sensitivity,
            "total_selections": self.selections_count,
        }
    
    def __repr__(self) -> str:
        return (
            f"ExponentialMechanism(epsilon={self.epsilon:.6f}, "
            f"sensitivity={self.sensitivity:.4f}, selections={self.selections_count})"
        )


class UtilityAwareExponentialMechanism(ExponentialMechanism):
    """
    Enhanced Exponential Mechanism with utility function specification.
    
    Allows defining custom utility functions for evaluating options.
    """
    
    def __init__(
        self,
        epsilon: float,
        sensitivity: float,
        utility_func: callable = None,
        name: str = "UtilityAwareExponentialMechanism",
    ):
        """
        Initialize Utility-Aware Exponential Mechanism.
        
        Args:
            epsilon: Privacy budget
            sensitivity: Sensitivity of utility function
            utility_func: Function to compute utility from options
            name: Mechanism name
        """
        super().__init__(epsilon, sensitivity, name)
        self.utility_func = utility_func
    
    def select_with_function(
        self,
        options: List[Any],
        description: str = "",
    ) -> Any:
        """
        Select option by computing utilities using specified function.
        
        Args:
            options: Candidate options
            description: Description for logging
            
        Returns:
            Selected option with privacy guarantee
            
        Raises:
            ValueError: If utility_func not set
        """
        if self.utility_func is None:
            raise ValueError("Utility function not set. Use select() instead.")
        
        utilities = [self.utility_func(opt) for opt in options]
        return self.select(options, utilities, description=description)
    
    def set_utility_function(self, utility_func: callable):
        """Set the utility function."""
        self.utility_func = utility_func
        logger.debug(f"{self.name}: Utility function updated")
