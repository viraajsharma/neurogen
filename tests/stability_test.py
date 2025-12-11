import sys
import os
import numpy as np

# Adjust path to import core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.network import Network
from core.energy import compute_energy
from core.update_rules import adjust_weights
from data.synthetic_dataset import generate_dataset
from utils.seeds import set_seed

def test_stability():
    print("Running Stability Test...")
    set_seed(42)
    
    # 1. Load single sample
    dataset = generate_dataset(num_samples=1, input_size=64)
    sample = dataset[0]
    
    # 2. Init Network
    network = Network(num_neurons=64, connection_density=0.3)
    
    # 3. Run Iterations
    energies = []
    weight_changes = []
    
    print("\nIteration | Energy | Weight Change")
    print("-" * 35)
    
    for i in range(50):
        # Update
        network.forward(sample, steps=5)
        
        # Energy
        E = compute_energy(network)
        energies.append(E)
        
        # Plasticity
        dW = adjust_weights(network, learning_rate=0.01)
        weight_changes.append(dW)
        
        if i % 10 == 0:
            print(f"{i:9d} | {E:6.2f} | {dW:13.4f}")

    # 4. Checks
    print("-" * 35)
    
    # Check 1: Energy should generally decrease or stabilize (not explode)
    energy_decreasing = energies[-1] <= energies[0]
    print(f"Energy Decreased/Stabilized: {energy_decreasing} (Start: {energies[0]:.2f}, End: {energies[-1]:.2f})")
    
    # Check 2: Weights shouldn't explode (clipping handles this, but good to verify)
    weights = [w for n in network.neurons for _, w in n.connections]
    max_weight = max(np.abs(weights)) if weights else 0
    weights_bounded = max_weight <= 1.0
    print(f"Weights Bounded (<=1.0): {weights_bounded} (Max: {max_weight:.2f})")
    
    # Check 3: Activations bounded
    activations = network.get_activations()
    activations_bounded = np.all(np.abs(activations) <= 1.0) # Tanh range
    print(f"Activations Bounded (<=1.0): {activations_bounded}")

    if energy_decreasing and weights_bounded and activations_bounded:
        print("\n[PASS] Stability Test Passed.")
    else:
        print("\n[FAIL] Stability Test Failed.")

if __name__ == "__main__":
    test_stability()
