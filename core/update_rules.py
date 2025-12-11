import numpy as np

def adjust_weights(network, learning_rate=0.01):
    """
    Applies local plasticity rules to update weights.
    Standard Hebbian Learning: delta_w = rate * activation_pre * activation_post
    """
    activations = network.get_activations()
    total_change = 0.0

    for neuron in network.neurons:
        for idx, (target_id, weight) in enumerate(neuron.connections):
            # Calculate weight change
            pre = neuron.activation
            post = activations[target_id]
            
            # Simple Hebbian term
            delta = learning_rate * pre * post
            
            # Update weight
            new_weight = weight + delta
            
            # Basic weight bounding to prevent explosion
            new_weight = np.clip(new_weight, -1.0, 1.0)
            
            # Apply update
            neuron.connections[idx][1] = new_weight
            total_change += abs(delta)

    return total_change
