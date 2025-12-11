# Neurogen v1.1 Recommended Folder Structure

```
neurogen/
│
├── .git/                          # Git repository
├── .gitignore                     # Git ignore rules
├── .github/                       # GitHub-specific files (optional)
│   └── workflows/                 # CI/CD workflows
│       └── tests.yml              # Automated testing
│
├── README.md                      # Project overview
├── LICENSE                        # License file
├── CHANGELOG.md                   # Version history
├── requirements.txt               # Python dependencies (pinned versions)
├── pyproject.toml                 # Optional: Poetry/modern Python packaging
├── setup.py                       # Optional: Package installation script
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md            # System architecture (v1.0 baseline)
│   ├── DEV_BRANCH_SETUP.md        # Git workflow guide
│   ├── REPRODUCIBILITY.md         # Reproducibility guide
│   ├── API.md                     # API reference
│   ├── TUTORIALS.md               # Usage tutorials
│   └── V2.md                      # Future vision (existing)
│
├── configs/                       # Configuration files (NEW)
│   ├── base.yaml                  # Base configuration
│   ├── evolution.yaml             # Evolution parameters
│   ├── network.yaml               # Network architecture defaults
│   ├── training.yaml              # Training hyperparameters
│   └── tasks/                     # Task-specific configs
│       ├── xor.yaml               # XOR task config
│       ├── synthetic_patterns.yaml
│       └── mnist.yaml             # Future: MNIST config
│
├── core/                          # Core framework (REFACTORED)
│   ├── __init__.py
│   ├── genome.py                  # Genome representation (pure data)
│   ├── mutations.py               # Mutation operators (NEW)
│   ├── network.py                 # Dynamic network construction
│   ├── learning_rules.py          # Local learning rules (Hebbian, Oja, BCM)
│   ├── config.py                  # Config manager (NEW)
│   │
│   └── evolution/                 # Evolution module (NEW)
│       ├── __init__.py
│       ├── engine.py              # Main evolution engine
│       ├── strategies.py          # Evolution strategies
│       ├── selection.py           # Selection methods (elite, tournament, etc.)
│       └── population.py          # Population management
│
├── training/                      # Training infrastructure (NEW)
│   ├── __init__.py
│   ├── trainer.py                 # Main training loop
│   ├── fitness_functions.py       # Fitness evaluation functions
│   ├── evaluator.py               # Network evaluation logic
│   └── callbacks.py               # Training callbacks (logging, checkpointing)
│
├── data/                          # Data handling (NEW)
│   ├── __init__.py
│   ├── loaders.py                 # Dataset loading utilities
│   ├── synthetic_dataset.py       # Synthetic pattern generator
│   ├── preprocessors.py           # Data preprocessing
│   ├── splitters.py               # Train/val/test splitting
│   │
│   └── datasets/                  # Dataset storage
│       ├── synthetic_patterns_v1.csv   # Generated synthetic data
│       ├── xor.json               # XOR dataset (for consistency)
│       └── README.md              # Dataset documentation
│
├── utils/                         # Utilities (ENHANCED)
│   ├── __init__.py
│   ├── logger.py                  # Basic logger (existing)
│   ├── structured_logger.py       # Structured logging (JSON/CSV) (NEW)
│   ├── visualizer.py              # Visualization tools (existing)
│   ├── checkpoint_manager.py      # Checkpointing system (NEW)
│   ├── experiment_tracker.py      # Experiment tracking (NEW)
│   ├── seed_manager.py            # Reproducibility/seed control (NEW)
│   ├── environment.py             # Environment validation (NEW)
│   └── metrics.py                 # Metrics computation (NEW)
│
├── tasks/                         # Task definitions
│   ├── __init__.py
│   ├── xor.py                     # XOR task (refactored)
│   ├── synthetic_patterns.py      # Synthetic pattern task (NEW)
│   └── base_task.py               # Abstract task interface (NEW)
│
├── tests/                         # Unit tests (NEW)
│   ├── __init__.py
│   ├── test_genome.py             # Genome tests
│   ├── test_mutations.py          # Mutation tests
│   ├── test_network.py            # Network tests
│   ├── test_evolution.py          # Evolution tests
│   ├── test_learning_rules.py     # Learning rule tests
│   ├── test_data_loaders.py       # Data loading tests
│   └── test_reproducibility.py    # Reproducibility tests
│
├── experiments/                   # Experiment outputs (NEW)
│   ├── .gitignore                 # Ignore experiment outputs
│   ├── runs/                      # Individual experiment runs
│   │   ├── run_20231210_143022/   # Timestamped run directory
│   │   │   ├── config.yaml        # Config snapshot
│   │   │   ├── logs/              # Log files
│   │   │   ├── checkpoints/       # Model checkpoints
│   │   │   ├── results/           # Final results
│   │   │   └── visualizations/    # Plots and graphs
│   │   └── ...
│   └── experiments.db             # SQLite database for tracking (optional)
│
├── notebooks/                     # Jupyter notebooks (existing)
│   ├── analysis.ipynb             # Result analysis
│   ├── visualization.ipynb        # Visualization demos
│   └── experiments.ipynb          # Experiment notebooks
│
├── scripts/                       # Utility scripts (NEW)
│   ├── generate_synthetic_data.py # Generate synthetic datasets
│   ├── validate_environment.py    # Check environment setup
│   ├── run_experiment.py          # Run configured experiment
│   └── analyze_results.py         # Analyze experiment results
│
├── backend/                       # Web backend (existing)
│   ├── app.py
│   └── ...
│
├── frontend/                      # Web frontend (existing)
│   └── ...
│
├── assets/                        # Static assets (existing)
│   └── ...
│
└── .venv/                         # Virtual environment (gitignored)
```

---

## 📂 Directory Descriptions

### **Core Directories**

#### `configs/`
- **Purpose**: Externalized configuration files
- **Format**: YAML (human-readable, supports comments)
- **Structure**: Base configs + task-specific overrides
- **Version Control**: ✅ Committed to Git

#### `core/`
- **Purpose**: Core framework logic
- **Responsibilities**: Genome, network, evolution, learning
- **Design**: Pure logic, no I/O or hardcoded values
- **Testing**: Fully unit tested

#### `training/`
- **Purpose**: Training orchestration
- **Responsibilities**: Training loops, fitness evaluation, callbacks
- **Design**: Decoupled from core (uses core as library)

#### `data/`
- **Purpose**: Data management
- **Responsibilities**: Loading, preprocessing, generation
- **Storage**: `datasets/` subdirectory for actual data files

#### `utils/`
- **Purpose**: Cross-cutting utilities
- **Responsibilities**: Logging, checkpointing, visualization, metrics
- **Design**: Reusable, framework-agnostic

#### `tasks/`
- **Purpose**: Task definitions
- **Responsibilities**: Task-specific fitness functions and data
- **Design**: Inherit from `base_task.py` interface

#### `tests/`
- **Purpose**: Automated testing
- **Framework**: `pytest` recommended
- **Coverage**: Aim for >80% code coverage

#### `experiments/`
- **Purpose**: Experiment outputs
- **Version Control**: ❌ Gitignored (too large)
- **Structure**: Timestamped run directories
- **Tracking**: Optional SQLite database for metadata

#### `scripts/`
- **Purpose**: Standalone utility scripts
- **Use Cases**: Data generation, environment validation, batch runs
- **Design**: CLI-friendly with argparse

---

## 🆕 New Files in v1.1

### Configuration
```
configs/base.yaml
configs/evolution.yaml
configs/network.yaml
configs/training.yaml
configs/tasks/xor.yaml
configs/tasks/synthetic_patterns.yaml
```

### Core Modules
```
core/mutations.py
core/config.py
core/evolution/engine.py
core/evolution/strategies.py
core/evolution/selection.py
core/evolution/population.py
```

### Training Infrastructure
```
training/trainer.py
training/fitness_functions.py
training/evaluator.py
training/callbacks.py
```

### Data Handling
```
data/loaders.py
data/synthetic_dataset.py
data/preprocessors.py
data/splitters.py
data/datasets/synthetic_patterns_v1.csv
```

### Utilities
```
utils/structured_logger.py
utils/checkpoint_manager.py
utils/experiment_tracker.py
utils/seed_manager.py
utils/environment.py
utils/metrics.py
```

### Testing
```
tests/test_genome.py
tests/test_mutations.py
tests/test_network.py
tests/test_evolution.py
tests/test_learning_rules.py
tests/test_data_loaders.py
tests/test_reproducibility.py
```

### Scripts
```
scripts/generate_synthetic_data.py
scripts/validate_environment.py
scripts/run_experiment.py
scripts/analyze_results.py
```

### Documentation
```
docs/REPRODUCIBILITY.md
docs/API.md
docs/TUTORIALS.md
```

---

## 🔄 Migration from v1.0 to v1.1

### Files to Refactor
- `core/evolution.py` → Split into `core/evolution/` module
- `tasks/xor.py` → Use new config system and trainer
- `utils/logger.py` → Keep, add `structured_logger.py`

### Files to Keep As-Is
- `core/genome.py` (minor updates)
- `core/network.py` (minor updates)
- `core/local_learning.py` → Rename to `core/learning_rules.py`
- `utils/visualizer.py` (keep existing)

### Files to Archive
- None (all v1.0 files remain useful)

---

## 🎯 Design Principles

1. **Separation of Concerns**: Core logic, training, data, and utilities are independent
2. **Configuration Over Code**: All parameters externalized to configs
3. **Testability**: Every module has corresponding tests
4. **Reproducibility**: Seed management and environment validation built-in
5. **Extensibility**: Abstract interfaces for strategies and tasks
6. **Documentation**: Every directory has a clear purpose

---

## 📦 Dependency Management

### `requirements.txt` (Pinned Versions)
```
torch==2.0.1
numpy==1.24.3
networkx==3.1
matplotlib==3.7.1
pyyaml==6.0
pytest==7.3.1
pandas==2.0.2
```

### Optional: `pyproject.toml` (Poetry)
```toml
[tool.poetry.dependencies]
python = "^3.9"
torch = "^2.0.0"
numpy = "^1.24.0"
...
```

---

## 🚀 Quick Start (v1.1)

```bash
# 1. Clone and setup
git clone <repo>
cd neurogen
git checkout dev/v1.1

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Validate environment
python scripts/validate_environment.py

# 5. Generate synthetic data
python scripts/generate_synthetic_data.py

# 6. Run experiment
python scripts/run_experiment.py --config configs/tasks/xor.yaml

# 7. Analyze results
python scripts/analyze_results.py --run experiments/runs/run_<timestamp>
```

---

**This structure supports scalability, reproducibility, and maintainability for Neurogen v1.1 and beyond! 🚀**
