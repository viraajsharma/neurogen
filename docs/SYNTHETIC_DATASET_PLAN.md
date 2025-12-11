# Neurogen v1.1 Synthetic Dataset Plan

## 🎯 Objective
Create a small, controlled synthetic dataset (50-100 samples) to validate Neurogen v1.1's learning capabilities on known patterns beyond XOR.

---

## 📊 Dataset Specification

### **Dataset Name**: `synthetic_patterns_v1`

### **Size**: 
- **Training**: 60 samples
- **Validation**: 20 samples
- **Test**: 20 samples
- **Total**: 100 samples

### **Task Type**: Binary Classification

### **Input Dimensions**: 4 features (expandable to 8 for harder variants)

### **Output**: Single binary label (0 or 1)

---

## 🧬 Pattern Types

The dataset will contain **4 distinct learnable patterns** with controlled difficulty:

### **Pattern 1: Linear Separable (Easy)**
- **Description**: Simple linear boundary
- **Rule**: `y = 1 if (x1 + x2 > x3 + x4) else 0`
- **Samples**: 25 (25%)
- **Difficulty**: ⭐ Easy
- **Purpose**: Baseline sanity check

### **Pattern 2: XOR-Like (Medium)**
- **Description**: Non-linear, requires hidden layer
- **Rule**: `y = 1 if (x1 > 0.5) XOR (x2 > 0.5) else 0`
- **Samples**: 25 (25%)
- **Difficulty**: ⭐⭐ Medium
- **Purpose**: Test non-linear learning (known to work in v1.0)

### **Pattern 3: Polynomial Boundary (Medium-Hard)**
- **Description**: Quadratic decision boundary
- **Rule**: `y = 1 if (x1^2 + x2^2 > 0.5) else 0`
- **Samples**: 25 (25%)
- **Difficulty**: ⭐⭐⭐ Medium-Hard
- **Purpose**: Test capacity for complex boundaries

### **Pattern 4: Multi-Feature Interaction (Hard)**
- **Description**: Requires all 4 features
- **Rule**: `y = 1 if (x1 * x2 + x3 * x4 > threshold) else 0`
- **Samples**: 25 (25%)
- **Difficulty**: ⭐⭐⭐⭐ Hard
- **Purpose**: Test feature interaction learning

---

## 🔧 Generation Strategy

### **Feature Distribution**
- All features sampled from **uniform distribution [0, 1]**
- Ensures balanced feature ranges
- No feature dominance

### **Label Generation**
```python
def generate_label(x1, x2, x3, x4, pattern_type):
    if pattern_type == 'linear':
        return 1 if (x1 + x2 > x3 + x4) else 0
    
    elif pattern_type == 'xor':
        return 1 if (x1 > 0.5) != (x2 > 0.5) else 0
    
    elif pattern_type == 'polynomial':
        return 1 if (x1**2 + x2**2 > 0.5) else 0
    
    elif pattern_type == 'interaction':
        threshold = 0.25
        return 1 if (x1 * x2 + x3 * x4 > threshold) else 0
```

### **Noise Injection (Optional)**
- **Label Noise**: 5% random label flips (configurable)
- **Feature Noise**: Gaussian noise with σ=0.05 (configurable)
- **Purpose**: Test robustness

### **Class Balance**
- Ensure ~50% positive, ~50% negative samples per pattern
- Adjust thresholds if needed to maintain balance

---

## 📁 File Format

### **CSV Format** (`data/datasets/synthetic_patterns_v1.csv`)

```csv
x1,x2,x3,x4,label,pattern_type,split
0.234,0.567,0.123,0.890,1,linear,train
0.789,0.234,0.456,0.678,0,xor,train
0.123,0.456,0.789,0.234,1,polynomial,val
0.567,0.890,0.345,0.123,0,interaction,test
...
```

**Columns:**
- `x1, x2, x3, x4`: Input features (float, range [0, 1])
- `label`: Binary output (0 or 1)
- `pattern_type`: Pattern category (linear, xor, polynomial, interaction)
- `split`: Data split (train, val, test)

### **JSON Format** (Alternative)

```json
{
  "metadata": {
    "version": "1.0",
    "num_samples": 100,
    "num_features": 4,
    "task": "binary_classification",
    "patterns": ["linear", "xor", "polynomial", "interaction"]
  },
  "data": [
    {
      "features": [0.234, 0.567, 0.123, 0.890],
      "label": 1,
      "pattern_type": "linear",
      "split": "train"
    },
    ...
  ]
}
```

---

## 🛠️ Implementation: `data/synthetic_dataset.py`

### **Core Functions**

```python
import numpy as np
import pandas as pd
from typing import Tuple, List

def generate_synthetic_patterns(
    num_samples: int = 100,
    num_features: int = 4,
    noise_level: float = 0.0,
    label_noise: float = 0.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic pattern dataset.
    
    Args:
        num_samples: Total number of samples
        num_features: Number of input features (4 or 8)
        noise_level: Gaussian noise std for features
        label_noise: Probability of label flip
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with features, labels, pattern types, and splits
    """
    np.random.seed(seed)
    
    # Generate features
    X = np.random.uniform(0, 1, size=(num_samples, num_features))
    
    # Generate labels for each pattern type
    patterns = ['linear', 'xor', 'polynomial', 'interaction']
    samples_per_pattern = num_samples // len(patterns)
    
    data = []
    for i, pattern in enumerate(patterns):
        start_idx = i * samples_per_pattern
        end_idx = start_idx + samples_per_pattern
        
        for j in range(start_idx, end_idx):
            x = X[j]
            label = _generate_label(x, pattern)
            
            # Add label noise
            if np.random.rand() < label_noise:
                label = 1 - label
            
            # Add feature noise
            if noise_level > 0:
                x = x + np.random.normal(0, noise_level, size=x.shape)
                x = np.clip(x, 0, 1)  # Keep in [0, 1]
            
            # Assign split
            split = _assign_split(j, num_samples)
            
            data.append({
                'x1': x[0], 'x2': x[1], 'x3': x[2], 'x4': x[3],
                'label': label,
                'pattern_type': pattern,
                'split': split
            })
    
    return pd.DataFrame(data)


def _generate_label(x: np.ndarray, pattern: str) -> int:
    """Generate label based on pattern type."""
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]
    
    if pattern == 'linear':
        return 1 if (x1 + x2 > x3 + x4) else 0
    
    elif pattern == 'xor':
        return 1 if (x1 > 0.5) != (x2 > 0.5) else 0
    
    elif pattern == 'polynomial':
        return 1 if (x1**2 + x2**2 > 0.5) else 0
    
    elif pattern == 'interaction':
        return 1 if (x1 * x2 + x3 * x4 > 0.25) else 0
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def _assign_split(idx: int, total: int) -> str:
    """Assign train/val/test split."""
    ratio = idx / total
    if ratio < 0.6:
        return 'train'
    elif ratio < 0.8:
        return 'val'
    else:
        return 'test'


def save_dataset(df: pd.DataFrame, filepath: str):
    """Save dataset to CSV."""
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from CSV."""
    df = pd.read_csv(filepath)
    X = df[['x1', 'x2', 'x3', 'x4']].values
    y = df['label'].values
    return X, y


def load_dataset_by_split(filepath: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """Load specific split from dataset."""
    df = pd.read_csv(filepath)
    df_split = df[df['split'] == split]
    X = df_split[['x1', 'x2', 'x3', 'x4']].values
    y = df_split['label'].values
    return X, y
```

---

## 🚀 Usage Example

### **Generate Dataset**

```python
from data.synthetic_dataset import generate_synthetic_patterns, save_dataset

# Generate dataset
df = generate_synthetic_patterns(
    num_samples=100,
    num_features=4,
    noise_level=0.05,  # 5% feature noise
    label_noise=0.02,  # 2% label noise
    seed=42
)

# Save to file
save_dataset(df, 'data/datasets/synthetic_patterns_v1.csv')
```

### **Load Dataset for Training**

```python
from data.synthetic_dataset import load_dataset_by_split
import torch

# Load training data
X_train, y_train = load_dataset_by_split('data/datasets/synthetic_patterns_v1.csv', split='train')

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

print(f"Training samples: {len(X_train)}")
print(f"Feature shape: {X_train.shape}")
print(f"Label shape: {y_train.shape}")
```

---

## 📊 Dataset Statistics

### **Expected Distribution**

| Pattern Type | Train | Val | Test | Total |
|--------------|-------|-----|------|-------|
| Linear       | 15    | 5   | 5    | 25    |
| XOR          | 15    | 5   | 5    | 25    |
| Polynomial   | 15    | 5   | 5    | 25    |
| Interaction  | 15    | 5   | 5    | 25    |
| **Total**    | **60**| **20**| **20**| **100**|

### **Class Balance**
- Target: 50% positive, 50% negative per pattern
- Adjust thresholds if imbalance > 60/40

---

## 🧪 Validation Criteria

### **Dataset Quality Checks**
- [ ] All features in range [0, 1]
- [ ] No missing values
- [ ] Class balance within 40-60% per pattern
- [ ] Train/val/test splits are 60/20/20
- [ ] Patterns are learnable (test with simple baseline)

### **Baseline Performance**
Test with a simple 2-layer MLP (with backprop) to ensure patterns are learnable:
- **Linear**: >90% accuracy
- **XOR**: >85% accuracy
- **Polynomial**: >75% accuracy
- **Interaction**: >70% accuracy

---

## 🔄 Integration with Neurogen v1.1

### **Fitness Function**

```python
def evaluate_synthetic_patterns_fitness(network, X, y, num_episodes=5, hebbian_lr=0.01):
    """
    Evaluate network on synthetic patterns dataset.
    """
    network.eval()
    best_accuracy = 0.0
    
    for episode in range(num_episodes):
        with torch.no_grad():
            outputs = network(X)
        
        # Apply Hebbian learning
        if episode < num_episodes - 1:
            network.apply_local_learning(learning_rate=hebbian_lr)
        
        # Compute accuracy
        predictions = (outputs > 0.5).float()
        correct = (predictions == y).float().sum().item()
        accuracy = correct / len(y)
        
        best_accuracy = max(best_accuracy, accuracy)
    
    return best_accuracy
```

### **Task Configuration** (`configs/tasks/synthetic_patterns.yaml`)

```yaml
task:
  name: synthetic_patterns
  dataset_path: data/datasets/synthetic_patterns_v1.csv
  
evolution:
  population_size: 200
  num_generations: 300
  elite_size: 20
  
network:
  num_inputs: 4
  num_outputs: 1
  
training:
  num_episodes: 5
  hebbian_lr: 0.01
  target_fitness: 0.85  # 85% accuracy
```

---

## 📈 Success Metrics

### **v1.1 Goals**
- [ ] Neurogen achieves >80% accuracy on linear patterns
- [ ] Neurogen achieves >75% accuracy on XOR patterns
- [ ] Neurogen achieves >65% accuracy on polynomial patterns
- [ ] Neurogen achieves >60% accuracy on interaction patterns
- [ ] Overall accuracy >70% across all patterns

---

## 🔮 Future Extensions

### **v1.2+**
- Increase to 500-1000 samples
- Add regression tasks
- Multi-class classification (3+ classes)
- Time-series patterns
- Real-world datasets (MNIST, UCI)

---

**This synthetic dataset provides a controlled, reproducible benchmark for Neurogen v1.1! 🚀**
