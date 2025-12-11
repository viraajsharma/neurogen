# NeuroGen v1.1.0 — Modular Self-Organization Update

**Summary**
NeuroGen v1.1.0 introduces a stable, modular foundation for self-organizing neural networks. This release focuses on "alive vs dead" dynamics, ensuring that networks can process input, update weights via local rules (Hebbian), and maintain stability without exploding.

**Key Features**
- **Modular Architecture**: Split into `core` (neuron, network, energy), `data`, `training`, and `utils`.
- **Synthetic Dataset**: Built-in generator for 8x8 binary patterns and geometric shapes.
- **Stability Verification**: Automated tests (`stability_test.py`) prove energy minimization and weight bounds.
- **CSV Logging**: detailed per-iteration metrics for energy, activation, and weight changes.
- **Determinism**: Fully reproducible runs via `utils/seeds.py`.

**Performance Results**
- **Stability**: Networks consistently minimize energy (e.g., from 1.91 to -12.58) over 50 iterations.
- **Convergence**: Weights stabilize to bounded values (max 0.81) using Hebbian learning.
- **Reproducibility**: 100% identical results across runs with fixed seeds.

**Usage**
Run the trainer:
```bash
python training/trainer.py
```

Run verification tests:
```bash
python tests/stability_test.py
python tests/full_dataset_test.py
```

**Next Steps (v1.2)**
- Integration of v1.1 self-organization into the evolutionary loop (v1.0).
- Advanced local rules (Oja, BCM) implementation in the new modular structure.
