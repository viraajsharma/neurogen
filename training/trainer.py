import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
# import yaml
from core.network import Network
from core.energy import compute_energy
from core.update_rules import adjust_weights
from data.synthetic_dataset import generate_dataset
from training.logger import CSVLogger
from utils.seeds import set_seed

class Trainer:
    def __init__(self, config_path="configs/default.yaml", seed=42):
        self.config_path = config_path
        self.seed = seed
        set_seed(self.seed)
        
        # Load Config (Mocking for now if file doesn't exist, or implementing basic dict)
        # For v1.1 scope, we can hardcode default params if config missing
        self.config = {
            "num_neurons": 64,
            "connection_density": 0.2,
            "iterations": 100,
            "learning_rate": 0.01
        }
        
    def train(self):
        print("Initializing Training...")
        
        # 1. Load Data
        dataset = generate_dataset(num_samples=50, input_size=self.config["num_neurons"])
        print(f"Loaded dataset: {dataset.shape}")
        
        # 2. Initialize Network
        network = Network(num_neurons=self.config["num_neurons"], 
                          connection_density=self.config["connection_density"])
        
        # 3. Setup Logger
        logger = CSVLogger()
        print(f"Logging to {logger.filepath}")
        
        # 4. Training Loop
        print("Starting Iterations...")
        for iteration in range(self.config["iterations"]):
            total_energy = 0
            
            # For each sample (or a subset/batch)
            # In this simple v1.1, let's present one random sample per iteration 
            # or iterate through whole dataset. Let's do random sample for online learning feel.
            sample_idx = np.random.randint(0, len(dataset))
            sample = dataset[sample_idx]
            
            # A. Forward Pass (Clamp input)
            network.forward(sample, steps=5) 
            
            # B. Compute Energy
            energy = compute_energy(network)
            
            # C. Update Rules (Plasticity)
            weight_change = adjust_weights(network, learning_rate=self.config["learning_rate"])
            
            # Metrics
            avg_activation = np.mean(network.get_activations())
            
            # D. Log
            logger.log(iteration, energy, avg_activation, weight_change)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration}: E={energy:.4f}, dW={weight_change:.4f}")
                
        print("Training Complete.")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
