"""
Evolutionary algorithm for NEUROGEN.

Implements a simple evolutionary loop:
1. Initialize population of genomes
2. Evaluate fitness
3. Select top performers
4. Mutate to create new generation
5. Repeat
"""

import random
import json
from typing import List, Callable, Tuple, Dict, Optional
from core.genome import Genome
from core.network import DynamicNetwork


class Individual:
    """Represents an individual in the population."""
    
    def __init__(self, genome: Genome):
        self.genome = genome
        self.fitness = 0.0
        self.network = None
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f}, {self.genome})"


class EvolutionEngine:
    """
    Manages the evolutionary process for neural network genomes.
    """
    
    def __init__(self, 
                 num_inputs: int,
                 num_outputs: int,
                 population_size: int = 100,
                 elite_size: int = 10,
                 mutation_rates: Optional[Dict[str, float]] = None):
        """
        Initialize the evolution engine.
        
        Args:
            num_inputs: Number of input nodes
            num_outputs: Number of output nodes
            population_size: Size of the population
            elite_size: Number of top individuals to keep unchanged
            mutation_rates: Dictionary of mutation probabilities
        """
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rates = mutation_rates or {
            'add_node': 0.03,
            'remove_node': 0.02,
            'add_connection': 0.05,
            'remove_connection': 0.03,
            'perturb_weight': 0.8
        }
        
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
    
    def initialize_population(self):
        """Create initial population of random genomes."""
        self.population = []
        for _ in range(self.population_size):
            genome = Genome(self.num_inputs, self.num_outputs)
            # Apply some initial mutations for diversity
            for _ in range(random.randint(0, 3)):
                genome.mutate(self.mutation_rates)
            self.population.append(Individual(genome))
        
        print(f"Initialized population of {self.population_size} individuals")
    
    def evaluate_population(self, fitness_fn: Callable[[DynamicNetwork], float]):
        """
        Evaluate fitness for all individuals in the population.
        
        Args:
            fitness_fn: Function that takes a network and returns fitness score
        """
        for individual in self.population:
            # Build network from genome
            individual.network = DynamicNetwork(individual.genome)
            
            # Evaluate fitness
            individual.fitness = fitness_fn(individual.network)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best individual
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = Individual(self.population[0].genome.copy())
            self.best_individual.fitness = self.population[0].fitness
        
        # Record statistics
        best_fitness = self.population[0].fitness
        avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
    
    def select_and_reproduce(self):
        """
        Select top individuals and create next generation through mutation.
        """
        # Keep elite individuals unchanged
        new_population = []
        for i in range(self.elite_size):
            elite = Individual(self.population[i].genome.copy())
            elite.fitness = self.population[i].fitness
            new_population.append(elite)
        
        # Create rest of population through mutation
        while len(new_population) < self.population_size:
            # Tournament selection: pick best of 3 random individuals
            tournament = random.sample(self.population[:self.population_size // 2], 
                                      min(3, len(self.population) // 2))
            parent = max(tournament, key=lambda x: x.fitness)
            
            # Create offspring through mutation
            offspring_genome = parent.genome.copy()
            offspring_genome.mutate(self.mutation_rates)
            new_population.append(Individual(offspring_genome))
        
        self.population = new_population
        self.generation += 1
    
    def evolve(self, 
               fitness_fn: Callable[[DynamicNetwork], float],
               num_generations: int = 100,
               target_fitness: Optional[float] = None,
               verbose: bool = True) -> Individual:
        """
        Run the evolutionary loop.
        
        Args:
            fitness_fn: Function to evaluate network fitness
            num_generations: Number of generations to evolve
            target_fitness: Stop early if this fitness is reached
            verbose: Print progress information
        
        Returns:
            Best individual found
        """
        if not self.population:
            self.initialize_population()
        
        for gen in range(num_generations):
            # Evaluate fitness
            self.evaluate_population(fitness_fn)
            
            best_fitness = self.best_fitness_history[-1]
            avg_fitness = self.avg_fitness_history[-1]
            
            if verbose and gen % 10 == 0:
                print(f"Generation {self.generation}: "
                      f"Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, "
                      f"Nodes={len(self.population[0].genome.nodes)}, "
                      f"Connections={len([c for c in self.population[0].genome.connections if c.enabled])}")
            
            # Check for early stopping
            if target_fitness is not None and best_fitness >= target_fitness:
                print(f"\nTarget fitness {target_fitness} reached at generation {self.generation}!")
                break
            
            # Create next generation
            if gen < num_generations - 1:  # Don't reproduce after last generation
                self.select_and_reproduce()
        
        # Final evaluation
        if self.generation == num_generations:
            self.evaluate_population(fitness_fn)
        
        if verbose:
            print(f"\nEvolution complete!")
            print(f"Best fitness: {self.best_individual.fitness:.4f}")
            print(f"Best genome: {self.best_individual.genome}")
        
        return self.best_individual
    
    def save_best(self, filepath: str):
        """
        Save the best individual's genome to a file.
        
        Args:
            filepath: Path to save the genome
        """
        if self.best_individual is None:
            print("No best individual to save")
            return
        
        genome_data = {
            'num_inputs': self.best_individual.genome.num_inputs,
            'num_outputs': self.best_individual.genome.num_outputs,
            'nodes': [
                {
                    'id': node.id,
                    'type': node.type,
                    'activation_fn': node.activation_fn
                }
                for node in self.best_individual.genome.nodes
            ],
            'connections': [
                {
                    'in_id': conn.in_id,
                    'out_id': conn.out_id,
                    'weight': conn.weight,
                    'enabled': conn.enabled
                }
                for conn in self.best_individual.genome.connections
            ],
            'fitness': self.best_individual.fitness,
            'generation': self.generation
        }
        
        with open(filepath, 'w') as f:
            json.dump(genome_data, f, indent=2)
        
        print(f"Saved best genome to {filepath}")
    
    def get_statistics(self) -> Dict:
        """Get evolution statistics."""
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'best_fitness': self.best_fitness_history[-1] if self.best_fitness_history else 0,
            'avg_fitness': self.avg_fitness_history[-1] if self.avg_fitness_history else 0,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }
