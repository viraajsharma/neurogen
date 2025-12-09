"""
XOR Task for NEUROGEN V1.

Evolve a neural network to solve XOR using:
- Evolutionary search (no backprop)
- Hebbian learning during evaluation
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.genome import Genome
from core.network import DynamicNetwork
from core.evolution import EvolutionEngine
from utils.logger import Logger, print_header, print_section
from utils.visualizer import visualize_genome, plot_fitness_history


# XOR truth table
XOR_DATA = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
], dtype=torch.float32)

XOR_LABELS = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
], dtype=torch.float32)


def evaluate_xor_fitness(network: DynamicNetwork, 
                         num_episodes: int = 5,
                         hebbian_lr: float = 0.01) -> float:
    """
    Evaluate network fitness on XOR task.
    
    Uses multiple episodes with Hebbian learning to allow the network
    to adapt during evaluation.
    
    Args:
        network: Neural network to evaluate
        num_episodes: Number of evaluation episodes with Hebbian updates
        hebbian_lr: Learning rate for Hebbian updates
    
    Returns:
        Fitness score (accuracy-based)
    """
    network.eval()
    
    best_accuracy = 0.0
    
    # Run multiple episodes with local learning
    for episode in range(num_episodes):
        # Forward pass
        with torch.no_grad():
            outputs = network(XOR_DATA)
        
        # Apply Hebbian learning
        if episode < num_episodes - 1:  # Don't update on last episode
            network.apply_local_learning(learning_rate=hebbian_lr)
        
        # Compute accuracy
        predictions = (outputs > 0.5).float()
        correct = (predictions == XOR_LABELS).float().sum().item()
        accuracy = correct / len(XOR_LABELS)
        
        best_accuracy = max(best_accuracy, accuracy)
    
    # Fitness is based on best accuracy achieved
    # Also add small bonus for getting close to correct values
    with torch.no_grad():
        final_outputs = network(XOR_DATA)
        mse = ((final_outputs - XOR_LABELS) ** 2).mean().item()
        proximity_bonus = max(0, 1.0 - mse)  # Bonus for low MSE
    
    fitness = best_accuracy + 0.1 * proximity_bonus
    
    return fitness


def run_xor_evolution(population_size: int = 150,
                     num_generations: int = 200,
                     elite_size: int = 15,
                     num_episodes: int = 5,
                     hebbian_lr: float = 0.01,
                     save_best: bool = True,
                     visualize: bool = True):
    """
    Run evolutionary algorithm to solve XOR.
    
    Args:
        population_size: Size of the population
        num_generations: Number of generations to evolve
        elite_size: Number of elite individuals to preserve
        num_episodes: Episodes per fitness evaluation
        hebbian_lr: Hebbian learning rate
        save_best: Whether to save the best genome
        visualize: Whether to create visualizations
    """
    print_header("NEUROGEN V1 - XOR Evolution Demo")
    
    logger = Logger(log_file='xor_evolution.log', verbose=True)
    
    logger.info(f"Configuration:")
    logger.info(f"  Population size: {population_size}")
    logger.info(f"  Generations: {num_generations}")
    logger.info(f"  Elite size: {elite_size}")
    logger.info(f"  Hebbian episodes: {num_episodes}")
    logger.info(f"  Hebbian learning rate: {hebbian_lr}")
    
    print_section("Initializing Evolution Engine")
    
    # Create evolution engine
    engine = EvolutionEngine(
        num_inputs=2,
        num_outputs=1,
        population_size=population_size,
        elite_size=elite_size
    )
    
    # Define fitness function
    def fitness_fn(network):
        return evaluate_xor_fitness(network, num_episodes, hebbian_lr)
    
    print_section("Starting Evolution")
    
    # Run evolution
    best_individual = engine.evolve(
        fitness_fn=fitness_fn,
        num_generations=num_generations,
        target_fitness=1.0,  # Perfect accuracy
        verbose=True
    )
    
    print_section("Evolution Results")
    
    logger.success(f"Best fitness achieved: {best_individual.fitness:.4f}")
    logger.info(f"Best genome: {best_individual.genome}")
    logger.info(f"  Nodes: {len(best_individual.genome.nodes)}")
    logger.info(f"  Connections: {len([c for c in best_individual.genome.connections if c.enabled])}")
    
    # Test the best network
    print_section("Testing Best Network")
    
    best_network = DynamicNetwork(best_individual.genome)
    
    logger.info("XOR Truth Table Predictions:")
    with torch.no_grad():
        outputs = best_network(XOR_DATA)
        for i, (inputs, label, output) in enumerate(zip(XOR_DATA, XOR_LABELS, outputs)):
            prediction = "1" if output.item() > 0.5 else "0"
            correct = "[PASS]" if prediction == str(int(label.item())) else "[FAIL]"
            logger.info(f"  {inputs[0]:.0f} XOR {inputs[1]:.0f} = {label[0]:.0f} | "
                       f"Predicted: {output.item():.3f} ({prediction}) {correct}")
    
    # Save best genome
    if save_best:
        print_section("Saving Best Genome")
        engine.save_best('best_xor_genome.json')
    
    # Create visualizations
    if visualize:
        print_section("Creating Visualizations")
        
        # Visualize best genome
        visualize_genome(best_individual.genome, 
                        save_path='best_xor_network.png',
                        show=False)
        
        # Plot fitness history
        stats = engine.get_statistics()
        plot_fitness_history(stats['best_fitness_history'],
                           stats['avg_fitness_history'],
                           save_path='xor_fitness_history.png',
                           show=False)
        
        logger.success("Visualizations saved!")
    
    print_section("Demo Complete")
    logger.info("Check the generated files:")
    logger.info("  - xor_evolution.log (training log)")
    logger.info("  - best_xor_genome.json (best genome)")
    logger.info("  - best_xor_network.png (network visualization)")
    logger.info("  - xor_fitness_history.png (fitness plot)")
    
    logger.close()
    
    return best_individual


if __name__ == '__main__':
    # Run the XOR evolution demo
    best = run_xor_evolution(
        population_size=150,
        num_generations=200,
        elite_size=15,
        num_episodes=5,
        hebbian_lr=0.01,
        save_best=True,
        visualize=True
    )
