import random
import numpy as np

def set_seed(seed):
    """
    Sets the random seed for Python's random module and NumPy.
    Ensures reproducibility across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    print(f"Random seed set to: {seed}")
