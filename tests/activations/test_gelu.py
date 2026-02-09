from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonTorch.Activations import GeLU
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class GELUTest(TritonKernelTest):
    """
    Test class for GELU activation.
    """
    def __init__(self, config: RunConfig, B: int, M: int, N: int, D: int):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        super().__init__(config)

    def setup_modules(self) -> None:
        self.gelu_triton = GeLU()
        self.gelu_torch = nn.GELU()

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu_torch(x)

class TestGELU:
    """
    Pytest test class for GELU activation.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 256, 256

    @parametrize_dtypes([torch.float16, torch.float32, torch.float64])
    @parametrize_dims([32, 64, 128, 256, 512, 1024, 2048])
    def test_gelu(self,
                  dtype: torch.dtype,
                  dim: int,
                  base_shape: tuple[int, int, int],
                  test_mode: str,
                  warmup_iterations: int,
                  show_timing: bool,
                  device: str,
                  seed: int,):
        """
        Test GELU forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = GELUTest(config, B, M, N, dim)
        test.run()