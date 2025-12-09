# NEUROGEN V1

**A framework for training neural networks using evolutionary search and local learning rules (no backpropagation)**

NEUROGEN evolves neural network architectures through mutation and selection while using Hebbian learning rules to adapt weights during evaluation. This approach is inspired by biological evolution and local synaptic plasticity.

## Features

- 🧬 **Evolutionary Architecture Search**: Networks evolve through mutations (add/remove nodes and connections)
- 🧠 **Local Learning Rules**: Hebbian, Oja's, and BCM learning (no gradient descent)
- 🔄 **Dynamic Networks**: PyTorch networks built dynamically from genome representation
- 📊 **Built-in Visualization**: Network graphs and fitness evolution plots
- ✅ **XOR Demo**: Working example solving XOR without backpropagation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neurogen.git
cd neurogen

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- NetworkX
- Matplotlib

## Quick Start

Run the XOR evolution demo:

```bash
python tasks/xor.py
```

This will:
1. Initialize a population of random neural networks
2. Evolve them over 200 generations using mutation and selection
3. Apply Hebbian learning during fitness evaluation
4. Save the best genome and create visualizations

### Expected Output

```
Generation 0: Best=0.2500, Avg=0.2500, Nodes=3, Connections=2
Generation 10: Best=0.5000, Avg=0.3125, Nodes=4, Connections=5
...
Generation 150: Best=1.0000, Avg=0.7500, Nodes=6, Connections=8

Target fitness 1.0 reached at generation 156!

XOR Truth Table Predictions:
  0 XOR 0 = 0 | Predicted: 0.023 (0) ✓
  0 XOR 1 = 1 | Predicted: 0.987 (1) ✓
  1 XOR 0 = 1 | Predicted: 0.991 (1) ✓
  1 XOR 1 = 0 | Predicted: 0.031 (0) ✓
```

### Generated Files

- `best_xor_genome.json` - Best evolved genome
- `best_xor_network.png` - Network visualization
- `xor_fitness_history.png` - Fitness evolution plot
- `xor_evolution.log` - Training log

## Architecture

```
neurogen/
│
├── core/
│   ├── genome.py          # Genome representation and mutations
│   ├── network.py         # Dynamic PyTorch network construction
│   ├── local_learning.py  # Hebbian, Oja, BCM learning rules
│   └── evolution.py       # Evolutionary algorithm
│
├── tasks/
│   └── xor.py            # XOR task demo
│
├── utils/
│   ├── logger.py         # Logging utilities
│   └── visualizer.py     # Network and fitness visualization
│
└── notebooks/            # Jupyter notebooks (coming soon)
```

## How It Works

### 1. Genome Representation

Each genome encodes a neural network with:
- **Nodes**: Input, hidden, and output nodes with activation functions
- **Connections**: Weighted edges between nodes
- **Mutations**: Add/remove nodes, add/remove connections, perturb weights

```python
from core.genome import Genome

# Create a genome with 2 inputs and 1 output
genome = Genome(num_inputs=2, num_outputs=1)

# Apply mutations
genome.mutate()
```

### 2. Dynamic Network Construction

Networks are built dynamically from genomes using PyTorch:

```python
from core.network import DynamicNetwork

# Build network from genome
network = DynamicNetwork(genome)

# Forward pass
output = network(input_tensor)

# Apply Hebbian learning
network.apply_local_learning(learning_rate=0.01)
```

### 3. Local Learning Rules

Weights adapt during evaluation using local rules:

- **Hebbian**: Δw = η × pre × post
- **Oja's Rule**: Δw = η × (pre × post - post² × w)
- **BCM**: Δw = η × pre × post × (post - θ)

```python
from core.local_learning import HebbianLearning

rule = HebbianLearning(learning_rate=0.01)
delta_w = rule.update(pre_activation, post_activation)
```

### 4. Evolution

Population evolves through:
1. **Evaluation**: Fitness measured on task
2. **Selection**: Tournament selection + elitism
3. **Reproduction**: Mutation of parent genomes

```python
from core.evolution import EvolutionEngine

engine = EvolutionEngine(
    num_inputs=2,
    num_outputs=1,
    population_size=150,
    elite_size=15
)

best = engine.evolve(
    fitness_fn=my_fitness_function,
    num_generations=200
)
```

## Customization

### Create Your Own Task

```python
import torch
from core.network import DynamicNetwork
from core.evolution import EvolutionEngine

# Define your fitness function
def my_fitness_fn(network: DynamicNetwork) -> float:
    # Your evaluation logic here
    inputs = torch.tensor([[...]])
    outputs = network(inputs)
    
    # Apply local learning
    network.apply_local_learning(learning_rate=0.01)
    
    # Return fitness score
    return accuracy

# Run evolution
engine = EvolutionEngine(num_inputs=4, num_outputs=2)
best = engine.evolve(fitness_fn=my_fitness_fn, num_generations=100)
```

### Adjust Mutation Rates

```python
mutation_rates = {
    'add_node': 0.05,        # Higher = more nodes added
    'remove_node': 0.02,     # Higher = more nodes removed
    'add_connection': 0.08,  # Higher = more connections added
    'remove_connection': 0.03,
    'perturb_weight': 0.9    # Higher = more weight changes
}

genome.mutate(mutation_rates)
```

## Roadmap

### V2 (Planned)
- [ ] Recurrent connections (remove acyclic constraint)
- [ ] More complex tasks (MNIST, CartPole)
- [ ] Speciation (NEAT-style)
- [ ] Multi-objective optimization
- [ ] GPU acceleration

### V3 (Future)
- [ ] Modular networks
- [ ] Meta-learning capabilities
- [ ] Neuromodulation
- [ ] Developmental encoding

## Contributing

Contributions are welcome! This is an open-source research project.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Citation

If you use NEUROGEN in your research, please cite:

```bibtex
@software{neurogen2024,
  title={NEUROGEN: Evolutionary Neural Networks with Local Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/neurogen}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Inspired by:
- NEAT (NeuroEvolution of Augmenting Topologies)
- Hebbian learning theory
- Evolutionary computation research

## Contact

For questions or collaboration: your.email@example.com

---

**Built with ❤️ for evolutionary AI research**
