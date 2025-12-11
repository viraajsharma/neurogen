"""
Synthetic Dataset Generator for Neurogen v1.1

This module generates synthetic datasets for testing and development.
Includes binary patterns, continuous vectors, and mixed data types.
"""

import numpy as np
from typing import Tuple, Optional, Literal


def generate_synthetic_dataset(
    n_samples: int = 100,
    input_dim: int = 10,
    output_dim: int = 2,
    pattern_type: Literal["binary", "continuous", "mixed"] = "binary",
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset for training and testing.
    
    Args:
        n_samples: Number of samples to generate (default: 100)
        input_dim: Input dimensionality (default: 10)
        output_dim: Output dimensionality (default: 2)
        pattern_type: Type of patterns to generate (default: "binary")
            - "binary": Binary patterns (0s and 1s)
            - "continuous": Continuous values from normal distribution
            - "mixed": Mix of binary and continuous features
        noise_level: Amount of noise to add (0.0 to 1.0, default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (X, y) where:
            X: Input data (shape: [n_samples, input_dim])
            y: Target data (shape: [n_samples, output_dim])
    
    Example:
        >>> X, y = generate_synthetic_dataset(n_samples=50, input_dim=10, output_dim=2)
        >>> print(X.shape, y.shape)
        (50, 10) (50, 2)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if pattern_type == "binary":
        X, y = _generate_binary_patterns(n_samples, input_dim, output_dim, noise_level)
    elif pattern_type == "continuous":
        X, y = _generate_continuous_patterns(n_samples, input_dim, output_dim, noise_level)
    elif pattern_type == "mixed":
        X, y = _generate_mixed_patterns(n_samples, input_dim, output_dim, noise_level)
    else:
        raise ValueError(f"Unknown pattern_type: {pattern_type}")
    
    return X, y


def _generate_binary_patterns(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise_level: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binary patterns (0s and 1s).
    
    Creates random binary vectors with some correlation to outputs.
    """
    # Generate random binary inputs
    X = np.random.binomial(1, 0.5, size=(n_samples, input_dim)).astype(np.float32)
    
    # Generate outputs based on simple rules
    # Rule: output depends on sum of first half vs second half of inputs
    mid = input_dim // 2
    sum_first_half = X[:, :mid].sum(axis=1)
    sum_second_half = X[:, mid:].sum(axis=1)
    
    y = np.zeros((n_samples, output_dim), dtype=np.float32)
    
    if output_dim == 2:
        # Binary classification
        y[:, 0] = (sum_first_half > sum_second_half).astype(np.float32)
        y[:, 1] = 1.0 - y[:, 0]
    else:
        # Multi-class or regression
        for i in range(output_dim):
            threshold = (i + 1) * (input_dim / (output_dim + 1))
            y[:, i] = (sum_first_half > threshold).astype(np.float32)
    
    # Add noise
    if noise_level > 0:
        noise = np.random.binomial(1, noise_level, size=X.shape)
        X = np.abs(X - noise)  # Flip bits with probability noise_level
    
    return X, y


def _generate_continuous_patterns(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise_level: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate continuous patterns from normal distribution.
    
    Creates inputs from N(0, 1) with linear relationship to outputs.
    """
    # Generate random continuous inputs
    X = np.random.randn(n_samples, input_dim).astype(np.float32)
    
    # Generate random weight matrix for linear relationship
    W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.5
    
    # Compute outputs as linear combination
    y = X @ W
    
    # Apply non-linearity (tanh)
    y = np.tanh(y)
    
    # Add noise
    if noise_level > 0:
        X += np.random.randn(*X.shape) * noise_level
        y += np.random.randn(*y.shape) * noise_level * 0.1
    
    return X, y


def _generate_mixed_patterns(
    n_samples: int,
    input_dim: int,
    output_dim: int,
    noise_level: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mixed binary and continuous patterns.
    
    First half of features are binary, second half are continuous.
    """
    mid = input_dim // 2
    
    # Binary features
    X_binary = np.random.binomial(1, 0.5, size=(n_samples, mid)).astype(np.float32)
    
    # Continuous features
    X_continuous = np.random.randn(n_samples, input_dim - mid).astype(np.float32)
    
    # Combine
    X = np.concatenate([X_binary, X_continuous], axis=1)
    
    # Generate outputs based on both types
    y = np.zeros((n_samples, output_dim), dtype=np.float32)
    
    for i in range(output_dim):
        # Combine binary sum and continuous mean
        binary_sum = X_binary.sum(axis=1)
        continuous_mean = X_continuous.mean(axis=1)
        y[:, i] = np.tanh(binary_sum * 0.1 + continuous_mean)
    
    # Add noise
    if noise_level > 0:
        X += np.random.randn(*X.shape) * noise_level
    
    return X, y


def get_dataset_info(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Get information about a dataset.
    
    Args:
        X: Input data
        y: Target data
    
    Returns:
        Dictionary with dataset statistics
    """
    return {
        'n_samples': X.shape[0],
        'input_dim': X.shape[1],
        'output_dim': y.shape[1] if y.ndim > 1 else 1,
        'X_mean': X.mean(),
        'X_std': X.std(),
        'X_min': X.min(),
        'X_max': X.max(),
        'y_mean': y.mean(),
        'y_std': y.std(),
        'y_min': y.min(),
        'y_max': y.max()
    }


def print_dataset_info(X: np.ndarray, y: np.ndarray):
    """
    Print dataset information.
    
    Args:
        X: Input data
        y: Target data
    """
    info = get_dataset_info(X, y)
    
    print(f"\n{'='*60}")
    print("Dataset Information")
    print(f"{'='*60}")
    print(f"Samples: {info['n_samples']}")
    print(f"Input dimension: {info['input_dim']}")
    print(f"Output dimension: {info['output_dim']}")
    print(f"\nInput statistics:")
    print(f"  Mean: {info['X_mean']:.4f}")
    print(f"  Std:  {info['X_std']:.4f}")
    print(f"  Min:  {info['X_min']:.4f}")
    print(f"  Max:  {info['X_max']:.4f}")
    print(f"\nOutput statistics:")
    print(f"  Mean: {info['y_mean']:.4f}")
    print(f"  Std:  {info['y_std']:.4f}")
    print(f"  Min:  {info['y_min']:.4f}")
    print(f"  Max:  {info['y_max']:.4f}")
    print(f"{'='*60}\n")
