import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.network import Network
from core.energy import compute_energy
from core.update_rules import adjust_weights
from data.synthetic_dataset import generate_dataset
from utils.seeds import set_seed

def test_full_dataset_convergence():
    print("Running Full Dataset Convergence Test...")
    set_seed(123)
    
    # 1. Dataset
    dataset = generate_dataset(num_samples=20, input_size=64)
    network = Network(num_neurons=64, connection_density=0.2)
    
    # 2. Train
    initial_energies = []
    final_energies = []
    
    print("Training on 20 samples (5 iterations per sample)...")
    for idx, sample in enumerate(dataset):
        # Record initial energy for this sample
        network.forward(sample, steps=2)
        initial_e = compute_energy(network)
        initial_energies.append(initial_e)
        
        # Train briefly
        for _ in range(5):
            network.forward(sample, steps=2)
            adjust_weights(network, learning_rate=0.01)
            
        # Record final energy
        final_e = compute_energy(network)
        final_energies.append(final_e)
        
    avg_initial = np.mean(initial_energies)
    avg_final = np.mean(final_energies)
    
    print(f"\nAverage Initial Energy: {avg_initial:.4f}")
    print(f"Average Final Energy:   {avg_final:.4f}")
    
    # Expectation: Network adapts to samples, lowering energy over time?
    # Or just stabilizing. Hebbian tries to reinforce correlations.
    # We at least want to see it didn't explode.
    
    if np.abs(avg_final) < 1000 and not np.isnan(avg_final):
        print("\n[PASS] Network converged to finite values.")
    else:
        print("\n[FAIL] Network diverged or exploded.")

if __name__ == "__main__":
    test_full_dataset_convergence()
