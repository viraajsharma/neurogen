# Neurogen v1.1 TODO: Top 5 Structural Problems

## 🎯 Mission
Address the critical engineering issues in v1.0 to create a production-ready, maintainable, and extensible framework for v1.1.

---

## ❌ Problem 1: Hardcoded Configuration Values

### **Issue**
Configuration parameters are scattered throughout the codebase with magic numbers and hardcoded values.

**Examples:**
- Mutation rates defined in multiple places (`genome.py`, `evolution.py`)
- Training hyperparameters hardcoded in task files (`xor.py`)
- Network parameters (weight ranges, clamping limits) embedded in code
- No single source of truth for configuration

**Impact:**
- ❌ Difficult to reproduce experiments
- ❌ Can't compare runs with different settings
- ❌ No way to version configurations
- ❌ Error-prone manual parameter changes

### **v1.1 Solution**
- [ ] Create `configs/` directory with YAML/JSON config files
- [ ] Implement `ConfigManager` class to load and validate configs
- [ ] Define config schemas for:
  - [ ] Evolution parameters (population size, elite size, mutation rates)
  - [ ] Network parameters (weight ranges, activation functions)
  - [ ] Training parameters (learning rates, episodes, generations)
  - [ ] Task-specific parameters (dataset paths, batch sizes)
- [ ] Add config versioning and experiment tracking
- [ ] Support config inheritance (base + task-specific overrides)

**Files to Create:**
- `configs/base.yaml`
- `configs/xor_task.yaml`
- `core/config.py`

---

## ❌ Problem 2: Insufficient Logging and Monitoring

### **Issue**
Minimal logging infrastructure with no structured output or detailed tracking.

**Current State:**
- Basic console logging only
- No structured log formats (JSON, CSV)
- No per-generation metrics beyond fitness
- No weight/gradient tracking
- No intermediate checkpointing
- Can't resume interrupted runs

**Impact:**
- ❌ Can't debug training failures
- ❌ No visibility into evolution dynamics
- ❌ Lost progress if training crashes
- ❌ Can't analyze what mutations worked
- ❌ No data for scientific analysis

### **v1.1 Solution**
- [ ] Implement structured logging system:
  - [ ] JSON logs for programmatic analysis
  - [ ] CSV logs for spreadsheet analysis
  - [ ] Console logs for human readability
- [ ] Track detailed metrics:
  - [ ] Per-generation: best/avg/worst fitness, topology stats
  - [ ] Per-individual: mutation history, parent lineage
  - [ ] Per-network: weight distributions, activation patterns
- [ ] Add checkpointing system:
  - [ ] Save population state every N generations
  - [ ] Support resume from checkpoint
  - [ ] Auto-save on crash/interrupt
- [ ] Create experiment tracking:
  - [ ] Unique run IDs
  - [ ] Config snapshots per run
  - [ ] Results database (SQLite)

**Files to Create:**
- `utils/structured_logger.py`
- `utils/checkpoint_manager.py`
- `utils/experiment_tracker.py`

---

## ❌ Problem 3: Poor Modularity and Code Organization

### **Issue**
Monolithic files with mixed responsibilities and tight coupling.

**Current Problems:**
- `evolution.py` handles both evolution logic AND fitness evaluation
- `network.py` mixes network construction with local learning
- No clear separation of concerns
- Hard to test individual components
- Can't swap out evolution strategies or learning rules easily

**Impact:**
- ❌ Difficult to extend with new features
- ❌ Hard to unit test
- ❌ Can't reuse components independently
- ❌ Tight coupling prevents experimentation

### **v1.1 Solution**
- [ ] Refactor into clean modules:
  - [ ] `core/genome.py` - Pure genome representation (no mutation logic)
  - [ ] `core/mutations.py` - Mutation operators as strategies
  - [ ] `core/network.py` - Network construction only
  - [ ] `core/learning_rules.py` - Local learning as plugins
  - [ ] `core/evolution/` - Evolution strategies (separate from fitness)
  - [ ] `training/` - Training loops and fitness functions
- [ ] Implement strategy pattern for:
  - [ ] Mutation strategies (configurable)
  - [ ] Selection strategies (elite, tournament, roulette)
  - [ ] Learning rules (Hebbian, Oja, BCM as plugins)
- [ ] Add abstract base classes for extensibility
- [ ] Create clean interfaces between components

**Files to Create:**
- `core/mutations.py`
- `core/evolution/strategies.py`
- `core/evolution/selection.py`
- `training/fitness_functions.py`
- `training/trainer.py`

---

## ❌ Problem 4: Limited Dataset Support and No Data Pipeline

### **Issue**
Only works with tiny hardcoded datasets (XOR). No data loading, preprocessing, or augmentation.

**Current Limitations:**
- XOR data hardcoded as 4-sample tensor
- No support for external datasets
- No train/val/test splits
- No data normalization or preprocessing
- No batch loading for larger datasets
- Can't scale beyond toy problems

**Impact:**
- ❌ Can't test on real-world problems
- ❌ No way to validate generalization
- ❌ Can't benchmark against other methods
- ❌ Limited to trivial tasks

### **v1.1 Solution**
- [ ] Create `data/` module:
  - [ ] `data/loaders.py` - Dataset loading utilities
  - [ ] `data/synthetic_dataset.py` - Synthetic pattern generator
  - [ ] `data/preprocessors.py` - Normalization, scaling
  - [ ] `data/splitters.py` - Train/val/test splitting
- [ ] Implement synthetic dataset generator:
  - [ ] 50-100 samples with known patterns
  - [ ] Multiple difficulty levels
  - [ ] Configurable noise levels
  - [ ] Support for classification and regression
- [ ] Add data pipeline:
  - [ ] Lazy loading for large datasets
  - [ ] Batch iteration
  - [ ] Data augmentation hooks
  - [ ] Caching for repeated access
- [ ] Support standard formats:
  - [ ] CSV, JSON, NumPy arrays
  - [ ] PyTorch Dataset/DataLoader compatibility

**Files to Create:**
- `data/loaders.py`
- `data/synthetic_dataset.py`
- `data/preprocessors.py`
- `data/datasets/synthetic_patterns_v1.csv`

---

## ❌ Problem 5: No Reproducibility Guarantees

### **Issue**
Runs are non-deterministic with no seed control or environment management.

**Current Problems:**
- No random seed setting
- Non-deterministic PyTorch operations
- No version pinning for dependencies
- No environment specification
- Can't reproduce published results
- No documentation of system requirements

**Impact:**
- ❌ Can't reproduce experiments
- ❌ Can't debug specific failures
- ❌ Can't verify scientific claims
- ❌ Collaboration is difficult
- ❌ No baseline for comparisons

### **v1.1 Solution**
- [ ] Implement deterministic execution:
  - [ ] Global seed setting (Python, NumPy, PyTorch)
  - [ ] Deterministic PyTorch operations
  - [ ] Seed tracking in configs and logs
  - [ ] Reproducible initialization
- [ ] Create environment management:
  - [ ] Pin exact dependency versions
  - [ ] Document Python version requirements
  - [ ] Provide `requirements.txt` with versions
  - [ ] Optional: `pyproject.toml` for Poetry
  - [ ] Docker container for full reproducibility (optional)
- [ ] Add reproducibility utilities:
  - [ ] `utils/seed_manager.py` - Centralized seed control
  - [ ] Environment validation on startup
  - [ ] Version logging (Python, PyTorch, NumPy)
  - [ ] Hardware info logging (CPU/GPU)
- [ ] Documentation:
  - [ ] Installation guide
  - [ ] Environment setup instructions
  - [ ] Reproducibility checklist

**Files to Create:**
- `utils/seed_manager.py`
- `utils/environment.py`
- `requirements.txt` (updated with versions)
- `docs/REPRODUCIBILITY.md`

---

## 📊 Priority Matrix

| Problem | Impact | Effort | Priority |
|---------|--------|--------|----------|
| 1. Hardcoded Config | High | Medium | **P0** |
| 2. Logging/Monitoring | High | Medium | **P0** |
| 3. Modularity | Medium | High | **P1** |
| 4. Dataset Support | High | Medium | **P0** |
| 5. Reproducibility | High | Low | **P0** |

---

## 🎯 v1.1 Success Criteria

### Must Have (P0)
- ✅ All configs externalized to YAML files
- ✅ Structured logging with JSON/CSV output
- ✅ Synthetic dataset with 50-100 samples
- ✅ Deterministic runs with seed control
- ✅ Checkpoint/resume functionality

### Should Have (P1)
- ✅ Modular architecture with strategy patterns
- ✅ Unit tests for core components
- ✅ Experiment tracking database
- ✅ Clean separation of concerns

### Nice to Have (P2)
- ⭕ Docker container for reproducibility
- ⭕ Multiple evolution strategies
- ⭕ Advanced data augmentation
- ⭕ Web-based experiment dashboard

---

## 🚀 Implementation Order

1. **Week 1**: Reproducibility + Config System
   - Set up seed management
   - Create config infrastructure
   - Pin dependencies

2. **Week 2**: Logging + Checkpointing
   - Implement structured logging
   - Add checkpoint system
   - Create experiment tracker

3. **Week 3**: Dataset + Data Pipeline
   - Build synthetic dataset generator
   - Create data loaders
   - Add preprocessing utilities

4. **Week 4**: Refactoring + Modularity
   - Extract mutation strategies
   - Separate fitness functions
   - Add abstract interfaces

5. **Week 5**: Testing + Documentation
   - Write unit tests
   - Update documentation
   - Validate reproducibility

---

**Let's build Neurogen v1.1 right! 🚀**
