# n2v

Python toolbox for neural network verification.

This toolbox verifies properties of neural networks using sound set-based reachability analysis and probabilistic verification via conformal inference. It supports PyTorch models and ONNX networks.

## Related Tools and Software

This toolbox is a translation of the MATLAB [NNV](https://github.com/verivital/nnv) tool by the VeriVITAL research group. It makes use of the neural network model transformation tool [onnx2torch](https://github.com/sammsaski/onnx2torch) for loading and converting ONNX models to PyTorch.

---

## Installation

### Step 1: Clone with Submodules

```bash
git clone --recurse-submodules <repository-url>

# If already cloned without submodules:
git submodule update --init --recursive
```

### Step 2: Install

```bash
cd n2v
pip install -r requirements.txt
pip install -e third_party/onnx2torch
pip install -e .
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- CVXPY >= 1.2.0
- onnx2torch (from submodule)

---

## Getting Started

### Quick Start

```python
import torch.nn as nn
import numpy as np
import n2v
from n2v.sets import Star

# Define a PyTorch model
model = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)
model.eval()

# Create input set (L-inf ball)
center = np.array([0.5, 0.5, 0.5])
epsilon = 0.1
input_star = Star.from_bounds(center - epsilon, center + epsilon)

# Compute reachable output set
net = n2v.NeuralNetwork(model)
output_stars = net.reach(input_star, method='exact')

# Extract bounds
for star in output_stars:
    lb, ub = star.get_ranges()
    print(f"Output bounds: [{lb.flatten()}, {ub.flatten()}]")
```

### Documentation

| Document | Description |
|----------|-------------|
| [docs/theoretical_foundations.md](docs/theoretical_foundations.md) | Mathematical details for all set types, layers, and relaxations |
| [docs/development_status.md](docs/development_status.md) | Feature inventory, layer support tables, and roadmap |
| [docs/probabilistic_verification.md](docs/probabilistic_verification.md) | Conformal inference theory and API |
| [docs/lp_solvers.md](docs/lp_solvers.md) | LP solver selection and comparison |
| [examples/](examples/) | Examples guide with tutorials and benchmarks |

---

## Features

### Set Representations

- **Star**: Exact polytopic constraints `x = c + V*alpha, C*alpha <= d`
- **Zonotope**: Efficient over-approximations `x = c + V*alpha, alpha in [-1,1]`
- **Box**: Fast interval bounds `lb <= x <= ub`
- **ImageStar / ImageZono**: Image-aware variants for CNNs
- **Hexatope / Octatope**: DCS/UTVPI-constrained zonotopes with strongly polynomial optimization
- **ProbabilisticBox**: Box with conformal inference coverage guarantees

### Layer Support (20+ types)

- **Linear**: Linear, Conv1D, Conv2D, BatchNorm, AvgPool2D, GlobalAvgPool, Flatten, Pad, Upsample, Transpose, Reshape, Reduce, Concat, Slice, Split
- **Nonlinear**: ReLU (exact/approx), LeakyReLU (exact/approx), Sigmoid (approx), Tanh (approx), Sign (exact/approx), MaxPool2D (exact/approx)
- **ONNX**: Add, Sub, Mul, Div, MatMul, Neg, Cast, and more via graph execution

See [docs/development_status.md](docs/development_status.md) for the full layer support matrix.

### Verification Methods

| Method | Guarantee | Description |
|--------|-----------|-------------|
| `exact` | Sound and complete | Star splitting at nonlinear layers |
| `approx` | Sound (over-approximate) | Triangle/S-curve relaxation, no splitting |
| `probabilistic` | Coverage with confidence | Conformal inference, model-agnostic |
| `hybrid` | Mixed | Exact until threshold, then probabilistic fallback |

### Falsification

Counterexample search via random sampling and Projected Gradient Descent (PGD) to quickly identify property violations before running expensive reachability analysis.

---

## Contributors

<!-- List contributors here -->
- [Samuel Sasaki](https://sammsaski.github.io/)

## References

The methods implemented in n2v are based upon or used in the following papers:

- Diego Manzanas Lopez, Sung Woo Choi, Hoang-Dung Tran, Taylor T. Johnson, "NNV 2.0: The Neural Network Verification Tool". In: Enea, C., Lal, A. (eds) Computer Aided Verification. CAV 2023. Lecture Notes in Computer Science, vol 13965. Springer, Cham. [https://doi.org/10.1007/978-3-031-37703-7_19]

- Navid Hashemi, Samuel Sasaki, Ipek Oguz, Meiyi Ma, Taylor T. Johnson, "Scaling Data-Driven Probabilistic Robustness Analysis for Semantic Segmentation Neural Networks", 38th Conference on Neural Information Processing Systems (NeurIPS), 2025.

- Samuel Sasaki, Diego Manzanas Lopez, Preston K. Robinette, Taylor T. Johnson, "Robustness Verification of Video Classification Neural Networks", IEEE/ACM 13th International Conference on Formal Methods in Software Engineering (FormaliSE), 2025. [https://doi.org/10.1109/FormaliSE66629.2025.00009]

- Lucas C. Cordeiro, Matthew L. Daggitt, Julien Girard-Satabin, Omri Isac, Taylor T. Johnson, Guy Katz, Ekaterina Komendantskaya, Augustin Lemesle, Edoardo Manino, Artjoms Sinkarovs, Haoze Wu, "Neural Network Verification is a Programming Language Challenge", 34th European Symposium on Programming (ESOP), 2025. [https://doi.org/10.1007/978-3-031-91118-7_9]

- Diego Manzanas Lopez, Samuel Sasaki, Taylor T. Johnson, "NNV: A Star Set Reachability Approach (Competition Contribution)", 7th Workshop on Formal Methods for ML-Enabled Autonomous Systems (SAIV), 2025. [https://doi.org/10.1007/978-3-031-99991-8_15]

- Hoang-Dung Tran, Neelanjana Pal, Patrick Musau, Xiaodong Yang, Nathaniel P. Hamilton, Diego Manzanas Lopez, Stanley Bak, Taylor T. Johnson, "Robustness Verification of Semantic Segmentation Neural Networks using Relaxed Reachability", In 33rd International Conference on Computer-Aided Verification (CAV), Springer, 2021. [http://www.taylortjohnson.com/research/tran2021cav.pdf]

- Hoang-Dung Tran, Stanley Bak, Weiming Xiang, Taylor T. Johnson, "Towards Verification of Large Convolutional Neural Networks Using ImageStars", 32nd International Conference on Computer-Aided Verification (CAV), 2020. [http://taylortjohnson.com/research/tran2020cav.pdf]

- Hoang-Dung Tran, Patrick Musau, Diego Manzanas Lopez, Xiaodong Yang, Luan Viet Nguyen, Weiming Xiang, Taylor T.Johnson, "Star-Based Reachability Analysis for Deep Neural Networks", The 23rd International Symposium on Formal Methods (FM), Porto, Portugal, 2019, Acceptance Rate 30%. . [http://taylortjohnson.com/research/tran2019fm.pdf]

**VNN-COMP Competition Reports**

- Konstantin Kaulen, Tobias Ladner, Stanley Bak, Christopher Brix, Hai Duong, Thomas Flinkow, Taylor T. Johnson, Lukas Koller, Edoardo Manino, ThanhVu H Nguyen, Haoze Wu, "The 6th International Verification of Neural Networks Competition (VNN-COMP 2025): Summary and Results", arXiv:2512.19007, 2025. [https://arxiv.org/abs/2512.19007]

- Christopher Brix, Stanley Bak, Taylor T. Johnson, Haoze Wu, "The Fifth International Verification of Neural Networks Competition (VNN-COMP 2024): Summary and Results", arXiv:2412.19985, 2024. [https://doi.org/10.48550/arXiv.2412.19985]

- Christopher Brix, Stanley Bak, Changliu Liu, Taylor T. Johnson, "The Fourth International Verification of Neural Networks Competition (VNN-COMP 2023): Summary and Results", arXiv:2312.16760, 2023. [https://arxiv.org/abs/2312.16760]

- Mark Niklas Müller, Christopher Brix, Stanley Bak, Changliu Liu, Taylor T. Johnson, "The Third International Verification of Neural Networks Competition (VNN-COMP 2022): Summary and Results", arXiv:2212.10376, 2022. [https://doi.org/10.48550/arXiv.2212.10376]

- Stanley Bak, Changliu Liu, Taylor T. Johnson, "The Second International Verification of Neural Networks Competition (VNN-COMP 2021): Summary and Results", arXiv:2109.00498, 2021. [https://arxiv.org/abs/2109.00498]

- Christopher Brix, Mark Niklas Müller, Stanley Bak, Taylor T. Johnson, Changliu Liu, "First Three Years of the International Verification of Neural Networks Competition (VNN-COMP)", Int J Softw Tools Technol Transfer 25, 329-339, 2023. [https://doi.org/10.1007/s10009-023-00703-4]

### Cite

TBD.

<!-- ```bibtex
@inproceedings{nnv2_cav2023,
    author = {Lopez, Diego Manzanas and Choi, Sung Woo
              and Tran, Hoang-Dung and Johnson, Taylor T.},
    title = {{NNV} 2.0: The Neural Network Verification Tool},
    booktitle = {Computer Aided Verification (CAV)},
    year = {2023},
    publisher = {Springer},
    doi = {10.1007/978-3-031-37703-7_19},
}
``` -->

## Acknowledgements

<!-- Fill in funding sources / acknowledgements here -->
This work is supported in part by AFOSR, DARPA, NSF.

## Contact

For any questions related to n2v (or NNV), please add them to the issues or contact [Samuel Sasaki](mailto:samuel.sasaki@vanderbilt.edu).

