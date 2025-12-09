"""
Local learning rules for NEUROGEN.

Implements Hebbian and other local learning rules that can be applied
during network evaluation (not backpropagation).
"""

import torch
from typing import Dict, Callable


class HebbianLearning:
    """
    Implements Hebbian learning rule: "Neurons that fire together, wire together"
    
    Weight update: Δw = η * pre_activation * post_activation
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Args:
            learning_rate: Learning rate (η) for weight updates
        """
        self.learning_rate = learning_rate
    
    def update(self, pre_activation: torch.Tensor, post_activation: torch.Tensor) -> torch.Tensor:
        """
        Compute Hebbian weight update.
        
        Args:
            pre_activation: Pre-synaptic activation (input)
            post_activation: Post-synaptic activation (output)
        
        Returns:
            Weight change (Δw)
        """
        # Simple Hebbian rule
        delta_w = self.learning_rate * pre_activation * post_activation
        return delta_w
    
    def __repr__(self):
        return f"HebbianLearning(lr={self.learning_rate})"


class OjaLearning:
    """
    Implements Oja's rule: Normalized Hebbian learning
    
    Weight update: Δw = η * (pre * post - post² * w)
    
    This prevents weights from growing unbounded.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Args:
            learning_rate: Learning rate (η) for weight updates
        """
        self.learning_rate = learning_rate
    
    def update(self, pre_activation: torch.Tensor, post_activation: torch.Tensor, 
               current_weight: torch.Tensor) -> torch.Tensor:
        """
        Compute Oja's rule weight update.
        
        Args:
            pre_activation: Pre-synaptic activation
            post_activation: Post-synaptic activation
            current_weight: Current weight value
        
        Returns:
            Weight change (Δw)
        """
        delta_w = self.learning_rate * (
            pre_activation * post_activation - 
            post_activation ** 2 * current_weight
        )
        return delta_w
    
    def __repr__(self):
        return f"OjaLearning(lr={self.learning_rate})"


class BCMLearning:
    """
    Implements BCM (Bienenstock-Cooper-Munro) learning rule.
    
    Includes a sliding threshold that adapts based on post-synaptic activity.
    Weight update: Δw = η * pre * post * (post - θ)
    where θ is a sliding threshold.
    """
    
    def __init__(self, learning_rate: float = 0.01, threshold_decay: float = 0.99):
        """
        Args:
            learning_rate: Learning rate (η) for weight updates
            threshold_decay: Decay rate for sliding threshold
        """
        self.learning_rate = learning_rate
        self.threshold_decay = threshold_decay
        self.threshold = 0.0
    
    def update(self, pre_activation: torch.Tensor, post_activation: torch.Tensor) -> torch.Tensor:
        """
        Compute BCM weight update.
        
        Args:
            pre_activation: Pre-synaptic activation
            post_activation: Post-synaptic activation
        
        Returns:
            Weight change (Δw)
        """
        # Update sliding threshold
        self.threshold = (self.threshold_decay * self.threshold + 
                         (1 - self.threshold_decay) * post_activation ** 2)
        
        # BCM rule
        delta_w = self.learning_rate * pre_activation * post_activation * (
            post_activation - self.threshold
        )
        return delta_w
    
    def __repr__(self):
        return f"BCMLearning(lr={self.learning_rate}, threshold={self.threshold:.3f})"


def get_learning_rule(rule_name: str = 'hebbian', **kwargs) -> Callable:
    """
    Factory function to get a learning rule by name.
    
    Args:
        rule_name: Name of the learning rule ('hebbian', 'oja', 'bcm')
        **kwargs: Additional arguments for the learning rule
    
    Returns:
        Learning rule instance
    """
    rules = {
        'hebbian': HebbianLearning,
        'oja': OjaLearning,
        'bcm': BCMLearning
    }
    
    if rule_name not in rules:
        raise ValueError(f"Unknown learning rule: {rule_name}. "
                        f"Available: {list(rules.keys())}")
    
    return rules[rule_name](**kwargs)
