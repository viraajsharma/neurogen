# Neurogen v1.0 Architecture

## 🏗️ Overview

Neurogen v1.0 implements **evolutionary neural networks with local learning rules** as an alternative to traditional backpropagation-based training. The system combines:

1. **Evolutionary Search**: Genetic algorithms evolve network topology and initial weights
2. **Local Learning**: Hebbian-style rules adapt weights during network evaluation
3. **Dynamic Networks**: Networks are constructed dynamically from genome representations

---

## 📦 Core Components

### 1. Genome Representation (`core/genome.py`)

The genome encodes a neural network's structure and parameters.

#### **Node**
Represents a single neuron in the network.

```python
class Node:
    - id: int              # Unique identifier
    - type: str            # 'input', 'hidden', or 'output'
    - activation_fn: str   # 'relu', 'tanh', 'sigmoid', 'linear'
```

#### **Connection**
Represents a weighted edge between two nodes.

```python
class Connection:
    - in_id: int      # Source node ID
    - out_id: int     # Destination node ID
    - weight: float   # Connection weight
    - enabled: bool   # Whether connection is active
```

#### **Genome**
Container for nodes and connections with mutation operators.

**Initialization:**
- Creates input and output nodes
- Establishes fully-connected initial topology
- Assigns random weights in [-1, 1]

**Mutation Operations:**
- `add_node()`: Splits a connection, inserting a new hidden node
- `remove_node()`: Deletes a hidden node and its connections
- `add_connection()`: Adds a new edge (ensures acyclicity)
- `remove_connection()`: Disables a random connection
- `perturb_weight()`: Adjusts weight (90% small perturbation, 10% reset)

**Mutation Rates (default):**
```python
{
    'add_node': 0.03,
    'remove_node': 0.02,
    'add_connection': 0.05,
    'remove_connection': 0.03,
    'perturb_weight': 0.8
}
```

---

### 2. Dynamic Network Construction (`core/network.py`)

Converts a genome into an executable PyTorch network.

#### **DynamicNetwork**

**Build Process:**
1. Maps node IDs to indices
2. Creates learnable parameters for each enabled connection
3. Computes topological evaluation order (Kahn's algorithm)
4. Stores activation functions per node

**Forward Pass:**
1. Initialize node activation values
2. Set input node values from input tensor
3. Evaluate nodes in topological order:
   - Compute weighted sum of incoming connections
   - Apply activation function
   - Store pre- and post-activation values (for local learning)
4. Collect output node values

**Local Learning (Hebbian Rule):**
```python
Δw = η * pre_activation * post_activation
```

- Applied after each forward pass
- Updates connection weights based on correlated activity
- Weights clamped to [-5, 5] to prevent explosion
- Averaged over batch dimension

**Key Methods:**
- `forward(x)`: Feedforward evaluation
- `apply_local_learning(lr)`: Hebbian weight updates
- `get_weights()` / `set_weights()`: Weight access

---

### 3. Evolutionary Algorithm (`core/evolution.py`)

Manages population-based search for optimal network structures.

#### **Individual**
Wrapper for a genome with fitness tracking.

```python
class Individual:
    - genome: Genome
    - fitness: float
    - network: DynamicNetwork
```

#### **EvolutionEngine**

**Initialization:**
- Creates population of random genomes
- Applies initial mutations for diversity

**Evolution Loop:**
1. **Evaluate**: Build networks from genomes, compute fitness
2. **Select**: Sort by fitness, preserve elite individuals
3. **Reproduce**: Tournament selection + mutation to create offspring
4. **Repeat**: Until target fitness or max generations reached

**Selection Strategy:**
- **Elitism**: Top `elite_size` individuals preserved unchanged
- **Tournament Selection**: Pick best of 3 random candidates from top 50%
- **Mutation-Only Reproduction**: No crossover (asexual reproduction)

**Fitness Tracking:**
- `best_fitness_history`: Best fitness per generation
- `avg_fitness_history`: Population average per generation
- `best_individual`: Global best genome found

---

### 4. Local Learning Rules (`core/local_learning.py`)

Implements biologically-inspired weight update rules.

#### **HebbianLearning**
Classic Hebbian rule: "Neurons that fire together, wire together"

```python
Δw = η * pre * post
```

#### **OjaLearning**
Normalized Hebbian learning (prevents unbounded growth)

```python
Δw = η * (pre * post - post² * w)
```

#### **BCMLearning**
Bienenstock-Cooper-Munro rule with sliding threshold

```python
Δw = η * pre * post * (post - θ)
θ = decay * θ + (1 - decay) * post²
```

**Note:** v1.0 primarily uses basic Hebbian learning in `DynamicNetwork`.

---

### 5. Training Loop (Example: `tasks/xor.py`)

Demonstrates the complete training pipeline.

#### **Fitness Evaluation**
```python
def evaluate_xor_fitness(network, num_episodes=5, hebbian_lr=0.01):
    1. Run multiple episodes
    2. For each episode:
       - Forward pass on training data
       - Apply Hebbian learning (except last episode)
       - Compute accuracy
    3. Return best accuracy + proximity bonus
```

**Fitness Formula:**
```python
fitness = best_accuracy + 0.1 * (1 - MSE)
```

#### **Evolution Process**
1. Initialize population (150 individuals)
2. For each generation:
   - Evaluate fitness (with Hebbian learning)
   - Select top performers
   - Mutate to create next generation
3. Track best individual and fitness history
4. Save best genome and visualizations

---

## 🔄 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY LOOP                        │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │  Initialize  │ ───▶ │   Evaluate   │ ───▶ │  Select  │ │
│  │  Population  │      │   Fitness    │      │   Elite  │ │
│  └──────────────┘      └──────────────┘      └──────────┘ │
│         │                     │                     │       │
│         │                     ▼                     │       │
│         │              ┌──────────────┐             │       │
│         │              │ Build Network│             │       │
│         │              │ from Genome  │             │       │
│         │              └──────────────┘             │       │
│         │                     │                     │       │
│         │                     ▼                     │       │
│         │              ┌──────────────┐             │       │
│         │              │ Multi-Episode│             │       │
│         │              │  Evaluation  │             │       │
│         │              │  + Hebbian   │             │       │
│         │              └──────────────┘             │       │
│         │                                           │       │
│         │              ┌──────────────┐             │       │
│         └────────────▶ │    Mutate    │ ◀───────────┘       │
│                        │   Offspring  │                     │
│                        └──────────────┘                     │
│                               │                             │
│                               ▼                             │
│                        Next Generation                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 Network Evaluation Flow

```
Input Data (batch)
      │
      ▼
┌─────────────────┐
│  Input Nodes    │  (Linear activation)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Hidden Nodes   │  (ReLU/Tanh/Sigmoid)
│  (Topological   │
│   Order)        │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│  Output Nodes   │  (Tanh activation)
└─────────────────┘
      │
      ▼
   Predictions
      │
      ▼
┌─────────────────┐
│ Hebbian Update  │  Δw = η * pre * post
│  (if training)  │
└─────────────────┘
```

---

## 📊 Weight Update Mechanism

### Evolution (Slow, Global)
- **Frequency**: Once per generation
- **Scope**: Entire population
- **Mechanism**: Mutation operators
- **Purpose**: Explore topology and weight space

### Local Learning (Fast, Local)
- **Frequency**: Every forward pass
- **Scope**: Individual network
- **Mechanism**: Hebbian rule
- **Purpose**: Fine-tune weights during evaluation

**Synergy:**
- Evolution finds good network structures
- Local learning adapts weights to specific patterns
- Combined approach explores both structure and parameter space

---

## 🎯 Input/Output Handling

### Input Processing
- Input tensor shape: `(batch_size, num_inputs)`
- Each input dimension maps to one input node
- Input nodes use linear activation (identity function)

### Output Generation
- Output nodes use `tanh` activation (range: [-1, 1])
- For binary classification: threshold at 0.5
- Output tensor shape: `(batch_size, num_outputs)`

### Training Data Format (XOR Example)
```python
XOR_DATA = torch.tensor([
    [0.0, 0.0],  # Input
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

XOR_LABELS = torch.tensor([
    [0.0],  # Expected output
    [1.0],
    [1.0],
    [0.0]
])
```

---

## 🔧 Current Configuration (XOR Task)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Population Size | 150 | Number of genomes per generation |
| Generations | 200 | Maximum evolution iterations |
| Elite Size | 15 | Top individuals preserved |
| Hebbian Episodes | 5 | Forward passes per fitness eval |
| Hebbian LR | 0.01 | Local learning rate |
| Mutation Rates | See genome.py | Probability of each mutation |

---

## 🗂️ Current File Structure

```
neurogen/
├── core/
│   ├── __init__.py
│   ├── genome.py          # Genome representation & mutations
│   ├── network.py         # Dynamic network construction
│   ├── evolution.py       # Evolutionary algorithm
│   └── local_learning.py  # Hebbian/Oja/BCM rules
├── utils/
│   ├── __init__.py
│   ├── logger.py          # Simple logging utilities
│   └── visualizer.py      # Network & fitness plotting
├── tasks/
│   ├── __init__.py
│   └── xor.py             # XOR evolution demo
├── docs/
│   └── ...
├── requirements.txt       # Dependencies
└── README.md
```

---

## 🧪 Typical Training Run

1. **Initialization** (Gen 0)
   - Create 150 random genomes
   - Each starts with 2 inputs → 1 output (fully connected)
   - Random weights in [-1, 1]

2. **Early Generations** (Gen 1-50)
   - Networks grow hidden nodes
   - Topology becomes more complex
   - Fitness slowly improves

3. **Mid Training** (Gen 51-150)
   - Successful structures emerge
   - Weight perturbations fine-tune
   - Fitness plateaus or improves

4. **Late Training** (Gen 151-200)
   - Elite individuals dominate
   - Small mutations refine solutions
   - Target fitness may be reached

5. **Result**
   - Best genome saved to JSON
   - Network visualization generated
   - Fitness history plotted

---

## 🎓 Key Design Principles

1. **No Backpropagation**: All learning is local or evolutionary
2. **Dynamic Topology**: Network structure evolves, not just weights
3. **Feedforward Only**: Acyclic graphs enforced
4. **Minimal Dependencies**: PyTorch for tensors, NetworkX for visualization
5. **Modular Design**: Clear separation of genome, network, and evolution

---

## 🔬 Strengths of v1.0

✅ Proof-of-concept for evolution + local learning  
✅ Solves XOR reliably  
✅ Clean genome representation  
✅ Modular code structure  
✅ Good visualization tools  

---

## ⚠️ Known Limitations

See `TODO_v1.1.md` for detailed structural problems to address in v1.1.

---

**This architecture document reflects Neurogen v1.0 as frozen at the v1.0.0 tag.**
