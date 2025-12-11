"""
Deterministic Seed Setup for Neurogen v1.1

This module provides utilities to set random seeds for reproducibility
across Python, NumPy, and other libraries.
"""

import random
import numpy as np
import os
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    This function sets seeds for:
    - Python's built-in random module
    - NumPy
    - OS environment variables for hash randomization
    
    Args:
        seed: Random seed value (default: 42)
    
    Example:
        >>> set_seed(42)
        >>> # All random operations will now be deterministic
        >>> np.random.randn(5)  # Will always produce same values
    """
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set Python hash seed for reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed} for reproducibility")


def get_random_state() -> dict:
    """
    Get current random state from all libraries.
    
    Returns:
        Dictionary containing random states
    """
    return {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'python_hash_seed': os.environ.get('PYTHONHASHSEED', 'not set')
    }


def set_random_state(state: dict) -> None:
    """
    Restore random state from saved state dictionary.
    
    Args:
        state: Dictionary containing random states (from get_random_state)
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])
    
    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])
    
    if 'python_hash_seed' in state:
        os.environ['PYTHONHASHSEED'] = state['python_hash_seed']


def create_seeded_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create a seeded NumPy random number generator.
    
    This is the modern way to handle random number generation in NumPy.
    
    Args:
        seed: Random seed (if None, uses random seed)
    
    Returns:
        NumPy random number generator
    
    Example:
        >>> rng = create_seeded_rng(42)
        >>> samples = rng.normal(0, 1, size=100)
    """
    return np.random.default_rng(seed)


def verify_determinism(seed: int = 42, n_trials: int = 3) -> bool:
    """
    Verify that random operations are deterministic with the given seed.
    
    Args:
        seed: Seed to test
        n_trials: Number of trials to run
    
    Returns:
        True if all trials produce identical results, False otherwise
    """
    results = []
    
    for _ in range(n_trials):
        set_seed(seed)
        
        # Generate some random values
        python_rand = random.random()
        numpy_rand = np.random.randn(10)
        
        results.append({
            'python': python_rand,
            'numpy': numpy_rand.tolist()
        })
    
    # Check if all results are identical
    first_result = results[0]
    for result in results[1:]:
        if result['python'] != first_result['python']:
            return False
        if result['numpy'] != first_result['numpy']:
            return False
    
    return True
