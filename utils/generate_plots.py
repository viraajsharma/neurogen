import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.network import Network
from core.energy import compute_energy
from core.update_rules import adjust_weights
from data.synthetic_dataset import generate_dataset
from utils.seeds import set_seed

def generate_energy_plot():
    print("Generating Energy Plot...")
    set_seed(42)
    
    # Setup
    dataset = generate_dataset(num_samples=1, input_size=64)
    sample = dataset[0]
    network = Network(num_neurons=64, connection_density=0.3)
    
    energies = []
    iterations = range(50)
    
    # Run simulation
    for i in iterations:
        network.forward(sample, steps=5)
        E = compute_energy(network)
        energies.append(E)
        adjust_weights(network, learning_rate=0.01)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, energies, label='Global Energy', color='blue', linewidth=2)
    plt.title('System Energy vs. Iterations (Hebbian Stabilization)', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Global Energy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save
    output_path = os.path.join("assets", "v1.1_energy_graph.png")
    if not os.path.exists("assets"):
        os.makedirs("assets")
        
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    generate_energy_plot()
