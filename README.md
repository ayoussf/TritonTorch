<p align="center">
    <img src="./assets/tritonhub.jpg" alt="TritonHub" style="object-fit: cover;"/>
</p>

<p align="center">
    <a href="https://github.com/ayoussf/Triton-Hub/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch 2.0+"></a>
    <a href="https://github.com/triton-lang/triton"><img src="https://img.shields.io/badge/Triton-Enabled-green.svg" alt="Triton"></a>
</p>

<!-- --- -->

## Overview

**TritonHub** is a *fully differentiable*, efficient, and modular open-source library of PyTorch neural network modules and operations implemented in [Triton](https://github.com/triton-lang/triton). It provides GPU-accelerated primitives that leverage Triton's low-level control and parallelism, enabling seamless integration of deep learning building blocks into your workflows.

<!-- --- -->

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Modules](#supported-modules)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

<!-- --- -->

## Installation

### Prerequisites

- Linux operating system (WSL for Windows users)
- CUDA-capable GPU
- Python 3.8+
- PyTorch 2.0+
- Triton (installed via pip or from [source](https://github.com/triton-lang/triton))

### From Source

```bash
git clone https://github.com/ayoussf/Triton-Hub.git
cd Triton-Hub
pip install .
```

### Development Installation

```bash
pip install -e '.[dev]'
```

<!-- --- -->

## Quick Start

```python
import torch
from TritonHub.Normalization import LayerNorm
from TritonHub.Activations import ReLU

# Configuration
batch, length, dim = 2, 100, 128
device = "cuda"
dtype = torch.float16  # Or torch.float32

# Initialize input tensor
x = torch.randn(batch, length, dim, device=device, dtype=dtype)

# Create modules
layernorm = LayerNorm(dim, eps=1e-6, elementwise_affine=True, 
                      bias=True, device=device, dtype=dtype)
relu = ReLU()

# Forward pass
x = layernorm(x)
x = relu(x)
```

<!-- --- -->

## Supported Modules

All modules support both **<span style="color:green"><strong>forward</strong></span>** and **<span style="color:red"><strong>backward</strong></span>** passes for full differentiability.

> [!NOTE]
> Some modules (e.g., Conv1d, Conv2d) are not fully optimized yet. It is still to be determined whether this is due to kernel implementation or autotuning configuration.

### Activation Functions

| Function | Description |
|----------|-------------|
| GeLU | Gaussian Error Linear Unit (with/without tanh approximation) |
| ReLU | Rectified Linear Unit |
| LeakyReLU | Leaky Rectified Linear Unit |
| ReLU6 | ReLU clamped at 6 |
| Sigmoid | Sigmoid activation |
| Tanh | Hyperbolic tangent |
| Mish | Mish activation function |
| SiLU | Sigmoid Linear Unit (Swish) |
| Softmax | Softmax normalization |
| LogSoftmax | Log-Softmax |
| Softmin | Softmin normalization |
| Softplus | Smooth approximation of ReLU |
| Threshold | Thresholded activation |

### Normalization Layers

| Layer | Description |
|-------|-------------|
| LayerNorm | Layer normalization |
| RMSNorm | Root Mean Square normalization |
| BatchNorm | Batch normalization *(In Progress)* |

### Neural Network Layers

| Layer | Description |
|-------|-------------|
| Linear | Fully connected layer |
| Dropout | Dropout regularization |
| MLP | Multi-Layer Perceptron (Gated-MLP / FFN) |
| Multi-Head Attention | Scaled dot-product attention |
| Conv1d | 1D convolution |
| Conv2d | 2D convolution |

### Operations

| Operation | Description |
|-----------|-------------|
| BMM | Batched matrix multiplication (supports unbatched inputs) |
| Normalize | L1, L2, and p-norm tensor normalization |
| Norm | Matrix/vector L1, L2, and p-norms |
| Pairwise Cosine Similarity | Distance computation between vectors |

<!-- --- -->

## Testing

See [tests/README.md](tests/README.md) for full documentation on running tests, CLI options, and test structure.

<!-- --- -->

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add unit tests under the `tests/` directory
4. Ensure compatibility with PyTorch and Triton
5. Submit a pull request

Found a bug or have a suggestion? Please [Open an issue](https://github.com/ayoussf/Triton-Hub/issues) or submit a [Pull Request](https://github.com/ayoussf/Triton-Hub/pulls).

<!-- --- -->

## License

TritonHub is released under the [MIT License](LICENSE). You are free to use, modify, and distribute it.

<!-- --- -->

## Acknowledgments

Special thanks to the authors of [Mamba](https://github.com/state-spaces/mamba). Their work has been a valuable reference for parts of this repository.