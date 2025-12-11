import numpy as np

def compute_energy(network):
    """
    Computes the global energy of the network.
    E = -0.5 * sum(w_ij * a_i * a_j)
    Lower energy implies a more stable configuration (in Hopfield-like dynamics).
    """
    energy = 0.0
    activations = network.get_activations()
    
    for neuron in network.neurons:
        for target_id, weight in neuron.connections:
            # E contribution: - weight * source_activation * target_activation
            # This aligns with Hebbian principles: consistent firing lowers energy.
            energy -= 0.5 * weight * neuron.activation * activations[target_id]
            
    return energy
