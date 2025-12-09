# NeuroGen

![NeuroGen Logo](assets/logo_placeholder.png)

**NeuroGen** is a research‑grade open‑source AI framework that replaces backpropagation with evolutionary search, genome‑driven network construction, and local learning rules (Hebbian, Oja, BCM, STDP). It enables the evolution of neural architectures and dynamic topology mutations without gradients.

---

## Why NeuroGen?

Traditional deep learning relies on backpropagation, which suffers from:
- **Biological implausibility** – gradients are not a mechanism used by the brain.
- **Vanishing/exploding gradients** – limits depth and stability.
- **Rigid architecture** – requires manual design of network topology.

NeuroGen offers a **backprop‑free** alternative that:
- Evolves architectures automatically.
- Learns locally through biologically‑inspired rules.
- Adapts topology dynamically during training.

---

## Key Features (V1)
- Evolutionary search over neural architectures.
- Genome‑driven network construction.
- Local learning rules: Hebbian, Oja, BCM, STDP.
- Dynamic topology mutations (add/remove nodes & edges).
- Demonstrated ability to evolve a network that solves the XOR problem.

---

## Screenshots

<div align="center">
  <img src="assets/fitness_curve_placeholder.png" alt="Fitness Curve" width="45%" />
  <img src="assets/network_graph_placeholder.png" alt="Network Graph" width="45%" />
</div>

*Placeholders – replace with actual images in the `/assets/` directory.*

---

## Installation

```bash
git clone https://github.com/yourorg/NeuroGen.git
cd NeuroGen
pip install -r requirements.txt
```

## Quickstart

```python
import neurogen
from neurogen.tasks import xor

# Run the XOR evolutionary demo
xor.run_demo()
```

---

## Running the XOR Demo

The repository includes a ready‑to‑run XOR demo:

```bash
python -m neurogen.tasks.xor
```

You will see a fitness curve and the evolved network topology printed to the console.

---

## Roadmap

| Version | Milestones |
|---------|------------|
| **V1** (Current) | Evolutionary XOR demo, basic local rules |
| **V2** | Additional learning rules, novelty search, symbolic tasks |
| **V3** | Speciation, surrogate models, annealing schedules |
| **V4** | Modular growth, gating mechanisms, meta‑parameter optimization |

---

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Ensure code follows the existing style and passes tests.
4. Submit a pull request with a clear description of changes.

---

## Citation

If you use NeuroGen in your research, please cite:

```
@software{NeuroGen2025,
  author = {Sharma, Viraaj},
  title = {NeuroGen: Backprop‑Free Evolutionary AI Framework},
  year = {2025},
  url = {https://github.com/yourorg/NeuroGen},
}
```

---

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
