# 📦 NEUROGEN v1.1 PRE-WORK PACKAGE - COMPLETE REFERENCE

**Generated:** 2025-12-10  
**Purpose:** Complete pre-development package for Neurogen v1.1  
**Status:** ✅ Ready for Development

---

## 📚 TABLE OF CONTENTS

1. [Dev Branch Setup](#1-dev-branch-setup)
2. [Current Architecture (v1.0)](#2-current-architecture-v10)
3. [Top 5 Structural Problems](#3-top-5-structural-problems)
4. [v1.1 Folder Structure](#4-v11-folder-structure)
5. [Synthetic Dataset Plan](#5-synthetic-dataset-plan)
6. [Reproducible Environment](#6-reproducible-environment)
7. [Quick Start Commands](#7-quick-start-commands)

---

## 1. DEV BRANCH SETUP

### Freeze v1.0.0 and Create dev/v1.1

```bash
# Step 1: Tag v1.0.0
git checkout main
git tag -a v1.0.0 -m "Release v1.0.0: Initial Neurogen implementation"
git push origin v1.0.0

# Step 2: Create dev branch
git checkout -b dev/v1.1
git push -u origin dev/v1.1

# Step 3: Verify
git tag  # Should show v1.0.0
git branch  # Should show * dev/v1.1
```

### Daily Workflow

```bash
# Start work
git checkout dev/v1.1
git pull origin dev/v1.1

# Make changes, commit frequently
git add <files>
git commit -m "feat: add config system"

# Push regularly
git push origin dev/v1.1
```

**📄 Full Details:** `docs/DEV_BRANCH_SETUP.md`

---

## 2. CURRENT ARCHITECTURE (v1.0)

### Core Components

1. **Genome** (`core/genome.py`)
   - Nodes: Input, Hidden, Output
   - Connections: Weighted edges
   - Mutations: Add/remove nodes/connections, perturb weights

2. **Network** (`core/network.py`)
   - Dynamic construction from genome
   - Topological evaluation order
   - Hebbian local learning

3. **Evolution** (`core/evolution.py`)
   - Population-based search
   - Elite selection + tournament
   - Mutation-only reproduction

4. **Local Learning** (`core/local_learning.py`)
   - Hebbian: Δw = η * pre * post
   - Oja: Normalized Hebbian
   - BCM: Sliding threshold

### Data Flow

```
Initialize Population → Evaluate Fitness → Select Elite → Mutate → Repeat
                           ↓
                    Build Network from Genome
                           ↓
                    Multi-Episode Evaluation
                           ↓
                    Apply Hebbian Learning
```

**📄 Full Details:** `docs/ARCHITECTURE.md`

---

## 3. TOP 5 STRUCTURAL PROBLEMS

### Problem 1: Hardcoded Configuration ⚠️
- **Issue:** Magic numbers scattered throughout code
- **Solution:** YAML config system (`configs/`, `core/config.py`)

### Problem 2: Insufficient Logging ⚠️
- **Issue:** No structured logs, no checkpointing
- **Solution:** JSON/CSV logging, checkpoint manager

### Problem 3: Poor Modularity ⚠️
- **Issue:** Tight coupling, mixed responsibilities
- **Solution:** Refactor into `core/evolution/`, `training/`, strategy patterns

### Problem 4: Limited Dataset Support ⚠️
- **Issue:** Only XOR, no data pipeline
- **Solution:** Synthetic dataset generator, data loaders

### Problem 5: No Reproducibility ⚠️
- **Issue:** Non-deterministic runs, no seed control
- **Solution:** Seed manager, pinned dependencies, environment validation

**📄 Full Details:** `TODO_v1.1.md`

---

## 4. v1.1 FOLDER STRUCTURE

```
neurogen/
├── configs/                    # NEW: YAML configurations
│   ├── base.yaml
│   ├── evolution.yaml
│   └── tasks/
│       └── xor.yaml
│
├── core/                       # REFACTORED
│   ├── genome.py
│   ├── mutations.py            # NEW
│   ├── network.py
│   ├── learning_rules.py
│   ├── config.py               # NEW
│   └── evolution/              # NEW
│       ├── engine.py
│       ├── strategies.py
│       └── selection.py
│
├── training/                   # NEW
│   ├── trainer.py
│   ├── fitness_functions.py
│   └── evaluator.py
│
├── data/                       # NEW
│   ├── loaders.py
│   ├── synthetic_dataset.py
│   ├── preprocessors.py
│   └── datasets/
│       └── synthetic_patterns_v1.csv
│
├── utils/                      # ENHANCED
│   ├── logger.py
│   ├── structured_logger.py    # NEW
│   ├── checkpoint_manager.py   # NEW
│   ├── seed_manager.py         # NEW
│   └── visualizer.py
│
├── tests/                      # NEW
│   ├── test_genome.py
│   ├── test_network.py
│   └── test_reproducibility.py
│
├── experiments/                # NEW
│   └── runs/
│       └── run_<timestamp>/
│
└── scripts/                    # NEW
    ├── generate_synthetic_data.py
    ├── validate_environment.py
    └── run_experiment.py
```

**📄 Full Details:** `docs/FOLDER_STRUCTURE_V1.1.md`

---

## 5. SYNTHETIC DATASET PLAN

### Specification
- **Size:** 100 samples (60 train, 20 val, 20 test)
- **Features:** 4 input dimensions
- **Task:** Binary classification
- **Patterns:** 4 types (25 samples each)

### Pattern Types

1. **Linear** (Easy): `y = 1 if (x1 + x2 > x3 + x4) else 0`
2. **XOR** (Medium): `y = 1 if (x1 > 0.5) XOR (x2 > 0.5) else 0`
3. **Polynomial** (Medium-Hard): `y = 1 if (x1² + x2² > 0.5) else 0`
4. **Interaction** (Hard): `y = 1 if (x1*x2 + x3*x4 > 0.25) else 0`

### File Format (CSV)

```csv
x1,x2,x3,x4,label,pattern_type,split
0.234,0.567,0.123,0.890,1,linear,train
0.789,0.234,0.456,0.678,0,xor,val
...
```

### Implementation Snippet

```python
def generate_synthetic_patterns(num_samples=100, seed=42):
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(num_samples, 4))
    # ... generate labels per pattern type
    return pd.DataFrame(data)
```

**📄 Full Details:** `docs/SYNTHETIC_DATASET_PLAN.md`

---

## 6. REPRODUCIBLE ENVIRONMENT

### Requirements (Pinned Versions)

```txt
torch==2.0.1
numpy==1.24.3
networkx==3.1
pandas==2.0.2
pyyaml==6.0.1
matplotlib==3.7.1
seaborn==0.12.2
pytest==7.3.1
pytest-cov==4.1.0
tqdm==4.65.0
```

### Setup Commands

```bash
# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Validate environment
python scripts/validate_environment.py
```

### Seed Management

```python
# utils/seed_manager.py
def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
```

**📄 Full Details:** `docs/REPRODUCIBLE_ENVIRONMENT.md`

---

## 7. QUICK START COMMANDS

### Complete Setup (Copy-Paste Ready)

```bash
# 1. Setup Git branches
git checkout main
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
git checkout -b dev/v1.1
git push -u origin dev/v1.1

# 2. Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create folder structure
mkdir -p configs/tasks
mkdir -p core/evolution
mkdir -p training
mkdir -p data/datasets
mkdir -p tests
mkdir -p experiments/runs
mkdir -p scripts

# 4. Validate environment
python scripts/validate_environment.py

# 5. Start development!
# Begin with P0 priorities: Config system, Seed manager, Dataset generator
```

---

## 📋 DEVELOPMENT ROADMAP

### Week 1: Foundation ⭐ P0
- [ ] Implement config system (`core/config.py`, `configs/*.yaml`)
- [ ] Implement seed manager (`utils/seed_manager.py`)
- [ ] Update folder structure

### Week 2: Data & Logging ⭐ P0
- [ ] Implement synthetic dataset generator (`data/synthetic_dataset.py`)
- [ ] Generate dataset (`synthetic_patterns_v1.csv`)
- [ ] Implement structured logging (`utils/structured_logger.py`)
- [ ] Implement checkpoint manager (`utils/checkpoint_manager.py`)

### Week 3: Refactoring ⭐ P1
- [ ] Extract mutations (`core/mutations.py`)
- [ ] Refactor evolution (`core/evolution/`)
- [ ] Create training module (`training/`)

### Week 4: Testing ⭐ P1
- [ ] Write unit tests (`tests/`)
- [ ] Validate reproducibility
- [ ] Test on synthetic dataset

### Week 5: Release ⭐
- [ ] Update documentation
- [ ] Merge to main
- [ ] Tag v1.1.0

---

## ✅ PRE-DEVELOPMENT CHECKLIST

Before coding, ensure:

- [ ] Read `ARCHITECTURE.md` (understand v1.0)
- [ ] Read `TODO_v1.1.md` (know what to fix)
- [ ] Read `FOLDER_STRUCTURE_V1.1.md` (know where files go)
- [ ] Created `dev/v1.1` branch
- [ ] Tagged `v1.0.0` release
- [ ] Created virtual environment
- [ ] Installed pinned dependencies
- [ ] Created folder structure

---

## 📁 DOCUMENT LOCATIONS

All documents are in your repository:

```
neurogen/
├── docs/
│   ├── V1.1_PREWORK_SUMMARY.md          # This summary
│   ├── ARCHITECTURE.md                  # v1.0 architecture
│   ├── DEV_BRANCH_SETUP.md              # Git workflow
│   ├── FOLDER_STRUCTURE_V1.1.md         # Folder layout
│   ├── SYNTHETIC_DATASET_PLAN.md        # Dataset plan
│   └── REPRODUCIBLE_ENVIRONMENT.md      # Environment setup
├── TODO_v1.1.md                         # Top 5 problems
└── requirements.txt                     # Pinned dependencies
```

---

## 🎯 SUCCESS CRITERIA

v1.1 is successful when:

✅ All configs externalized to YAML  
✅ Deterministic runs with seed control  
✅ Structured logging (JSON/CSV)  
✅ Synthetic dataset with 100 samples  
✅ Checkpoint/resume functionality  
✅ Modular architecture  
✅ >80% test coverage  
✅ Reproducible experiments  

---

## 🚀 YOU'RE READY!

This package provides everything needed to start Neurogen v1.1 development:

✅ **Clear baseline** (v1.0 architecture)  
✅ **Identified problems** (top 5 issues)  
✅ **Organized structure** (folder layout)  
✅ **Reproducible environment** (pinned deps, seeds)  
✅ **Benchmark dataset** (synthetic patterns)  
✅ **Git workflow** (branch strategy)  

**Next Step:** Start implementing! Begin with config system and seed management.

---

**Happy Coding! 🎉**
