# Neurogen v1.1 Reproducible Environment Setup

## 🎯 Objective
Ensure deterministic, reproducible experiments across different machines and runs.

---

## 📋 Requirements

### **Python Version**
- **Required**: Python 3.9 or 3.10
- **Recommended**: Python 3.10.12
- **Not Supported**: Python 3.11+ (PyTorch compatibility issues)

### **Operating System**
- ✅ Linux (Ubuntu 20.04+, Debian 11+)
- ✅ macOS (11.0+)
- ✅ Windows 10/11

---

## 📦 Dependencies (Pinned Versions)

### **requirements.txt**

```txt
# Core Dependencies
torch==2.0.1
numpy==1.24.3
networkx==3.1

# Data & Utilities
pandas==2.0.2
pyyaml==6.0.1

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Testing
pytest==7.3.1
pytest-cov==4.1.0

# Logging & Tracking
tqdm==4.65.0

# Optional: Jupyter
jupyter==1.0.0
ipykernel==6.23.1
```

### **Why Pinned Versions?**
- ✅ Ensures identical behavior across machines
- ✅ Prevents breaking changes from updates
- ✅ Enables exact reproduction of results
- ✅ Simplifies debugging

---

## 🛠️ Installation Instructions

### **1. Clone Repository**

```bash
git clone https://github.com/yourusername/neurogen.git
cd neurogen
git checkout dev/v1.1
```

### **2. Create Virtual Environment**

#### **Linux/macOS**
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### **Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### **Windows (CMD)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### **3. Upgrade pip**

```bash
pip install --upgrade pip setuptools wheel
```

### **4. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **5. Verify Installation**

```bash
python scripts/validate_environment.py
```

**Expected Output:**
```
✅ Python version: 3.10.12
✅ PyTorch version: 2.0.1
✅ NumPy version: 1.24.3
✅ All dependencies installed correctly
✅ Environment is ready for reproducible runs
```

---

## 🔒 Deterministic Execution

### **Seed Management**

Create `utils/seed_manager.py`:

```python
import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    Set global random seed for reproducibility.
    
    Args:
        seed: Random seed (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Global seed set to {seed}")


def get_rng_state():
    """Get current RNG state for checkpointing."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_rng_state(state):
    """Restore RNG state from checkpoint."""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])
```

### **Usage in Training Scripts**

```python
from utils.seed_manager import set_global_seed

# At the start of every script
set_global_seed(42)  # Or load from config
```

---

## 🧪 Environment Validation Script

Create `scripts/validate_environment.py`:

```python
#!/usr/bin/env python
"""
Validate Neurogen v1.1 environment setup.
"""

import sys
import importlib

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major == 3 and version.minor in [9, 10]:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro}")
        print("   Required: Python 3.9 or 3.10")
        return False


def check_package(package_name, required_version=None):
    """Check if package is installed with correct version."""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        
        if required_version and version != required_version:
            print(f"⚠️  {package_name} version: {version} (expected {required_version})")
            return False
        else:
            print(f"✅ {package_name} version: {version}")
            return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Neurogen v1.1 Environment Validation")
    print("=" * 60)
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    
    # Core dependencies
    checks.append(check_package('torch', '2.0.1'))
    checks.append(check_package('numpy', '1.24.3'))
    checks.append(check_package('networkx', '3.1'))
    checks.append(check_package('pandas', '2.0.2'))
    checks.append(check_package('yaml'))
    checks.append(check_package('matplotlib', '3.7.1'))
    
    # Testing
    checks.append(check_package('pytest', '7.3.1'))
    
    print("=" * 60)
    
    if all(checks):
        print("✅ All checks passed! Environment is ready.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

**Make executable:**
```bash
chmod +x scripts/validate_environment.py
```

---

## 📊 Reproducibility Checklist

### **Before Every Experiment**

- [ ] Virtual environment activated
- [ ] Dependencies installed from `requirements.txt`
- [ ] Global seed set in config
- [ ] PyTorch deterministic mode enabled
- [ ] Config file version controlled
- [ ] Hardware info logged (CPU/GPU model)

### **During Experiment**

- [ ] Seed logged in experiment metadata
- [ ] Config snapshot saved to experiment directory
- [ ] RNG state saved in checkpoints
- [ ] Python/PyTorch/NumPy versions logged

### **After Experiment**

- [ ] Results saved with full metadata
- [ ] Config and seed documented
- [ ] Environment info recorded
- [ ] Reproducibility verified (re-run with same seed)

---

## 🔧 Configuration Template

### **configs/base.yaml**

```yaml
# Neurogen v1.1 Base Configuration

# Reproducibility
seed: 42
deterministic: true

# Hardware
device: cpu  # or 'cuda' for GPU

# Logging
log_level: INFO
log_dir: experiments/runs
save_checkpoints: true
checkpoint_interval: 50  # generations

# Environment
python_version: "3.10"
torch_version: "2.0.1"
numpy_version: "1.24.3"
```

---

## 🐳 Docker Setup (Optional)

For maximum reproducibility, use Docker:

### **Dockerfile**

```dockerfile
FROM python:3.10.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "scripts/run_experiment.py"]
```

### **Build and Run**

```bash
# Build image
docker build -t neurogen:v1.1 .

# Run experiment
docker run -v $(pwd)/experiments:/app/experiments neurogen:v1.1 \
    python scripts/run_experiment.py --config configs/tasks/xor.yaml
```

---

## 🧪 Testing Reproducibility

### **Test Script** (`tests/test_reproducibility.py`)

```python
import pytest
import torch
from utils.seed_manager import set_global_seed
from core.genome import Genome

def test_seed_reproducibility():
    """Test that same seed produces same results."""
    
    # Run 1
    set_global_seed(42)
    genome1 = Genome(num_inputs=2, num_outputs=1)
    genome1.mutate()
    weights1 = [c.weight for c in genome1.connections]
    
    # Run 2 (same seed)
    set_global_seed(42)
    genome2 = Genome(num_inputs=2, num_outputs=1)
    genome2.mutate()
    weights2 = [c.weight for c in genome2.connections]
    
    # Should be identical
    assert weights1 == weights2, "Same seed should produce identical results"


def test_different_seeds():
    """Test that different seeds produce different results."""
    
    # Run 1
    set_global_seed(42)
    genome1 = Genome(num_inputs=2, num_outputs=1)
    genome1.mutate()
    weights1 = [c.weight for c in genome1.connections]
    
    # Run 2 (different seed)
    set_global_seed(123)
    genome2 = Genome(num_inputs=2, num_outputs=1)
    genome2.mutate()
    weights2 = [c.weight for c in genome2.connections]
    
    # Should be different
    assert weights1 != weights2, "Different seeds should produce different results"
```

**Run tests:**
```bash
pytest tests/test_reproducibility.py -v
```

---

## 📝 Logging Environment Info

### **Auto-log at Experiment Start**

```python
import sys
import torch
import numpy as np
import platform

def log_environment_info(logger):
    """Log environment information for reproducibility."""
    
    logger.info("=" * 60)
    logger.info("ENVIRONMENT INFO")
    logger.info("=" * 60)
    
    # System
    logger.info(f"OS: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version}")
    
    # Dependencies
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"NumPy: {np.__version__}")
    
    # Hardware
    logger.info(f"CPU: {platform.processor()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA: {torch.version.cuda}")
    else:
        logger.info("GPU: None (CPU only)")
    
    logger.info("=" * 60)
```

---

## 🚀 Quick Start Commands

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Validate setup
python scripts/validate_environment.py

# 3. Run reproducible experiment
python scripts/run_experiment.py --config configs/tasks/xor.yaml --seed 42

# 4. Verify reproducibility (run again with same seed)
python scripts/run_experiment.py --config configs/tasks/xor.yaml --seed 42

# 5. Compare results (should be identical)
python scripts/compare_runs.py experiments/runs/run_001 experiments/runs/run_002
```

---

## ⚠️ Common Issues

### **Issue 1: PyTorch Non-Determinism**
**Solution:** Ensure `torch.backends.cudnn.deterministic = True`

### **Issue 2: Different Results on GPU vs CPU**
**Solution:** Use CPU for strict reproducibility, or document GPU model

### **Issue 3: Version Mismatches**
**Solution:** Always use `pip install -r requirements.txt` (not `pip install torch`)

### **Issue 4: Floating Point Precision**
**Solution:** Use `torch.float32` consistently, avoid mixed precision

---

## 📚 References

- [PyTorch Reproducibility Guide](https://pytorch.org/docs/stable/notes/randomness.html)
- [NumPy Random Seed](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html)
- [Python Random Module](https://docs.python.org/3/library/random.html)

---

**With this setup, Neurogen v1.1 experiments are fully reproducible! 🚀**
