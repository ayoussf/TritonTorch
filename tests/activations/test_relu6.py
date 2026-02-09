from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonTorch.Activations import ReLU6
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class ReLU6Test(TritonKernelTest):
    """
    Test class for ReLU6 activation.
    """
    def __init__(self, config: RunConfig, B: int, M: int, N: int, D: int):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        super().__init__(config)

    def setup_modules(self) -> None:
        self.relu6_triton = ReLU6()
        self.relu6_torch = nn.ReLU6()

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu6_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu6_torch(x)

class TestReLU6:
    """
    Pytest test class for ReLU6 activation.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 256, 256

    @parametrize_dtypes([torch.float16, torch.float32, torch.float64])
    @parametrize_dims([32, 64, 128, 256, 512, 1024, 2048])
    def test_relu6(self,
                   dtype: torch.dtype,
                   dim: int,
                   base_shape: tuple[int, int, int],
                   test_mode: str,
                   warmup_iterations: int,
                   show_timing: bool,
                   device: str,
                   seed: int,):
        """
        Test ReLU6 forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = ReLU6Test(config, B, M, N, dim)
        test.run()