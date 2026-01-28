# Tests

## Table of Contents

- [Running Tests](#running-tests)
- [CLI Options](#cli-options)
- [Structure](#structure)
- [Adding New Tests](#adding-new-tests)
- [Overridable Methods](#overridable-methods)

## Running Tests

```bash
# Basic
pytest                                    # Run all tests
pytest -s                                 # With visible output (needed for timing)
pytest -s --show-timing                   # Show CUDA timing

# Specific tests
pytest tests/activations/                 # Run specific folder
pytest tests/activations/test_gelu.py    # Run specific file
pytest tests/activations/test_gelu.py::TestGELU::test_gelu  # Run specific test

# With options
pytest -s --show-timing --dtype float16
pytest -s --show-timing --dtypes float16,float32
pytest -s --show-timing --test-mode forward
pytest -s --show-timing --warmup-iterations 5
```

## CLI Options

<div align="center">

| Option | Description |
|:------:|:-----------:|
| `--test-mode [forward\|backward\|full]` | Test mode (default: full) |
| `--warmup-iterations N` | Warmup iterations before timing (default: 3) |
| `--show-timing` | Display CUDA timing |
| `--dtype [float16\|float32\|float64]` | Filter by single dtype |
| `--dtypes dtype1,dtype2,...` | Filter by multiple dtypes |

</div>

## Structure

```
tests/
├── base.py          # TritonKernelTest base class
├── utils.py         # CUDATimer, parametrize helpers
├── conftest.py      # Pytest fixtures and CLI options
├── activations/
├── layers/
├── normalization/
├── ops/
└── distance/
```

## Adding New Tests

1. Create a test file in the appropriate folder (e.g., `tests/activations/test_newactivation.py`)

2. Create a helper class that extends `TritonKernelTest`, for example:

```python
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class NewActivationTest(TritonKernelTest):
    def __init__(self, config: RunConfig, B: int, M: int, N: int, D: int):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        super().__init__(config)

    def setup_modules(self) -> None:
        self.triton_module = YourTritonModule()
        self.torch_module = nn.Reference()

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.triton_module(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.torch_module(x)
```

3. Create a pytest class that uses the helper:

```python
class TestNewActivation:
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 256, 256

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([32, 64, 128, 256, 512, 1024, 2048])
    def test_new_activation(
        self,
        dtype: torch.dtype,
        dim: int,
        base_shape: tuple[int, int, int],
        test_mode: str,
        warmup_iterations: int,
        show_timing: bool,
        device: str,
        seed: int):
        B, M, N = base_shape
        config = RunConfig(
            dtype=dtype,
            device=device,
            seed=seed,
            warmup_iterations=warmup_iterations,
            test_mode=test_mode,
            show_timing=show_timing)
        test = NewActivationTest(config, B, M, N, dim)
        test.run()
```

## Overridable Methods

The `TritonKernelTest` base class handles warmup, timing, and comparison automatically.

Some tests may require custom handling (e.g., multiple inputs, different assertions). In that case, you can override methods in the helper class that extends `TritonKernelTest`. See [test_cosine_similarity.py](distance/test_cosine_similarity.py) or [test_dropout.py](layers/test_dropout.py) for an example.