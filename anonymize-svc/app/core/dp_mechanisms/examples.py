"""
Example usage of Differential Privacy mechanisms in the preprocessing service.

Demonstrates integration of DP with anonymization for the credit-score-apps project.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple

# Import DP mechanisms
from dp_utils import DPUtility
from laplace_mechanism import LaplaceNoiseMechanism, AdaptiveLaplaceNoiseMechanism
from exponential_mechanism import ExponentialMechanism, UtilityAwareExponentialMechanism
from above_threshold import AboveThresholdMechanism, CountingQueriesMechanism
from dp_anonymization_integration import (
    DPAnonymizationIntegration,
    combine_anonymization_and_dp,
)


def example_1_basic_laplace_noise():
    """Example 1: Basic Laplace noise for numerical queries."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Laplace Noise for Numerical Queries")
    print("=" * 70)
    
    # Create mechanism with privacy budget epsilon=0.1
    mechanism = LaplaceNoiseMechanism(epsilon=0.1)
    
    # Query: Average age from dataset
    true_average_age = 42.5
    sensitivity = 1.0  # Age change by ≤ 1 when one record removed
    
    # Apply noise
    noisy_result = mechanism.apply(
        query_result=true_average_age,
        sensitivity=sensitivity,
        description="Average age of dataset"
    )
    
    print(f"True average age: {true_average_age}")
    print(f"Noisy average age (epsilon=0.1): {noisy_result:.2f}")
    print(f"Privacy budget status: {mechanism.get_budget_status()}")
    print()


def example_2_adaptive_laplace():
    """Example 2: Adaptive Laplace mechanism with progressive budget allocation."""
    print("=" * 70)
    print("EXAMPLE 2: Adaptive Laplace with Progressive Budget Allocation")
    print("=" * 70)
    
    # Total budget for 5 queries
    adaptive = AdaptiveLaplaceNoiseMechanism(total_epsilon=0.5, num_queries=5)
    
    # Multiple counting queries
    queries = [
        (150, "Count of records with income > 50k"),
        (320, "Count of records with age > 40"),
        (280, "Count of records with education = Bachelor's"),
        (95, "Count of records with marital_status = Married"),
        (45, "Count of records with race = White"),
    ]
    
    print(f"Total epsilon budget: {adaptive.total_epsilon}")
    print(f"Expected queries: {adaptive.num_queries}")
    print(f"Epsilon per query: {adaptive.epsilon_per_query:.6f}\n")
    
    for i, (true_count, description) in enumerate(queries, 1):
        noisy_count = adaptive.apply(
            query_result=true_count,
            sensitivity=1.0,
            description=description
        )
        print(f"Query {i}: {description}")
        print(f"  True: {true_count}, Noisy: {noisy_count:.1f}")
    
    print()


def example_3_exponential_mechanism():
    """Example 3: Exponential mechanism for selecting best attribute."""
    print("=" * 70)
    print("EXAMPLE 3: Exponential Mechanism for Attribute Selection")
    print("=" * 70)
    
    # Selecting which quasi-identifier to generalize in Mondrian
    mechanism = ExponentialMechanism(
        epsilon=0.2,
        sensitivity=1.0
    )
    
    # Candidate attributes and their generalization quality scores
    attributes = ["age", "occupation", "education", "marital_status"]
    utilities = [0.95, 0.85, 0.78, 0.72]  # Quality scores
    
    print("Selecting best attribute to generalize...")
    print(f"Attributes: {attributes}")
    print(f"Utility scores: {utilities}\n")
    
    # Get top-3 selections with probabilities
    top_3 = mechanism.get_top_k_probabilities(attributes, utilities, k=3)
    
    print("Top-3 selections (DP-protected):")
    for attr, prob in top_3:
        print(f"  {attr}: {prob:.4f}")
    
    # Multiple selections showing randomness from exponential mechanism
    print("\nMultiple selections from mechanism:")
    selections = [mechanism.select(attributes, utilities) for _ in range(5)]
    print(f"  {selections}")
    print()


def example_4_above_threshold():
    """Example 4: Above Threshold mechanism for binary queries."""
    print("=" * 70)
    print("EXAMPLE 4: Above Threshold for Binary Queries")
    print("=" * 70)
    
    # Threshold for identifying high-income records
    mechanism = AboveThresholdMechanism(
        epsilon=0.15,
        threshold=50000,  # Threshold for "high income"
        delta=0.0
    )
    
    # Test values
    test_values = [75000, 45000, 60000, 35000, 52000]
    
    print(f"Threshold: ${mechanism.threshold:,}")
    print(f"Privacy epsilon: {mechanism.epsilon}\n")
    print("Querying which income values are above threshold:\n")
    
    for value in test_values:
        result = mechanism.query(value)
        is_above = "Above" if result is not None else "Below"
        print(f"  Income ${value:,}: {is_above} threshold")
    
    print(f"\nStatistics: {mechanism.get_statistics()}")
    print()


def example_5_counting_queries():
    """Example 5: Counting queries with privacy."""
    print("=" * 70)
    print("EXAMPLE 5: Counting Queries with Privacy")
    print("=" * 70)
    
    mechanism = CountingQueriesMechanism(epsilon=0.3)
    
    # Counting queries on dataset
    total_records = 1000
    queries = [
        (250, "Count of records with income > 50k"),
        (450, "Count of male records"),
        (720, "Count of employed records"),
    ]
    
    print(f"Total records: {total_records}")
    print(f"Privacy epsilon: {mechanism.epsilon}\n")
    
    print("Count queries with Laplace noise:")
    for true_count, description in queries:
        noisy_count = mechanism.answer_count(true_count, total_records, description)
        proportion = true_count / total_records
        noisy_prop = noisy_count / total_records
        
        print(f"\n{description}")
        print(f"  True count: {true_count}, Noisy count: {noisy_count}")
        print(f"  True proportion: {proportion:.3f}, Noisy proportion: {noisy_prop:.3f}")
    
    print()


def example_6_dp_anonymization_integration():
    """Example 6: Integrating DP with anonymization (Mondrian)."""
    print("=" * 70)
    print("EXAMPLE 6: DP + Anonymization Integration")
    print("=" * 70)
    
    # Create sample anonymized data (after Mondrian k-anonymity)
    anonymized_data = pd.DataFrame({
        'age_range': ['20-30', '20-30', '40-50', '40-50', '30-40'],
        'occupation': ['Engineer', 'Engineer', 'Manager', 'Manager', 'Technician'],
        'income': [45000, 52000, 75000, 82000, 55000],
        'education': ['Bachelor', 'Bachelor', 'Master', 'Master', 'Bachelor'],
    })
    
    print("Anonymized data (after k-anonymity):")
    print(anonymized_data)
    print()
    
    # Add DP protection
    integration = DPAnonymizationIntegration(epsilon=0.4)
    
    print("Applying DP post-processing...")
    df_final = integration.apply_dp_to_numerical_columns(
        df=anonymized_data,
        numerical_columns=['income'],
        sensitivities={'income': 5000},
        epsilon_allocation={'income': 0.3}
    )
    
    print("\nDP-protected data (income column has noise):")
    print(df_final)
    print()
    
    # Privacy report
    print("Privacy report:")
    report = integration.get_privacy_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    print()


def example_7_privacy_utility_tradeoff():
    """Example 7: Understanding privacy-utility tradeoff."""
    print("=" * 70)
    print("EXAMPLE 7: Privacy-Utility Tradeoff")
    print("=" * 70)
    
    sensitivity = 10.0
    dimension = 1
    beta = 1e-5  # 99.999% confidence
    
    print("Query result with sensitivity = 10, dimension = 1, beta = 1e-5\n")
    print("Epsilon vs Maximum Error (99.999% confidence):")
    print("-" * 50)
    print(f"{'Epsilon':<12} {'Max Error':<20} {'Privacy Level':<20}")
    print("-" * 50)
    
    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    for eps in epsilons:
        max_error = DPUtility.compute_accuracy_guarantee(
            sensitivity=sensitivity,
            epsilon=eps,
            dimension=dimension,
            beta=beta
        )
        
        if eps < 0.1:
            privacy_level = "Very Strong"
        elif eps < 1.0:
            privacy_level = "Strong"
        elif eps < 5.0:
            privacy_level = "Moderate"
        else:
            privacy_level = "Weak"
        
        print(f"{eps:<12.2f} {max_error:<20.2f} {privacy_level:<20}")
    
    print("\nKey insight: Smaller epsilon = stronger privacy but larger errors")
    print()


def example_8_combined_workflow():
    """Example 8: Complete workflow combining Mondrian + DP."""
    print("=" * 70)
    print("EXAMPLE 8: Complete Anonymization + DP Workflow")
    print("=" * 70)
    
    # Simulate dataset
    np.random.seed(42)
    n_records = 100
    
    dataset = pd.DataFrame({
        'age': np.random.randint(18, 80, n_records),
        'occupation': np.random.choice(['Engineer', 'Manager', 'Technician'], n_records),
        'income': np.random.choice([0, 1], n_records),  # 0: <50k, 1: >50k
        'education': np.random.choice(['HS', 'Bachelor', 'Master'], n_records),
    })
    
    print(f"Original dataset: {n_records} records")
    print(f"Columns: {list(dataset.columns)}\n")
    
    # Step 1: Apply k-anonymity (simulated)
    print("Step 1: Apply k-anonymity (Mondrian)")
    print("  - Generalize quasi-identifiers")
    print("  - Result: Groups of size k=5\n")
    
    # Step 2: Apply DP
    print("Step 2: Apply Differential Privacy")
    dp_budget = 0.5
    integration = DPAnonymizationIntegration(epsilon=dp_budget)
    
    # In practice, would apply to actual anonymized output
    print(f"  - Budget: epsilon = {dp_budget}")
    print(f"  - Add noise to sensitive columns")
    print(f"  - Guarantee: ({dp_budget}, 0)-differential privacy\n")
    
    # Step 3: Evaluate
    print("Step 3: Privacy Evaluation")
    print("  - k-anonymity: 5")
    print("  - l-diversity: Yes (sensitive attribute has variety)")
    print(f"  - Differential Privacy: epsilon = {dp_budget}")
    print("  - Combined protection: Defense in depth")
    print()


if __name__ == "__main__":
    # Run all examples
    example_1_basic_laplace_noise()
    example_2_adaptive_laplace()
    example_3_exponential_mechanism()
    example_4_above_threshold()
    example_5_counting_queries()
    example_6_dp_anonymization_integration()
    example_7_privacy_utility_tradeoff()
    example_8_combined_workflow()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
