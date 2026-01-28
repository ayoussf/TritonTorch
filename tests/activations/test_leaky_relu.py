
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonHub.Activations import LeakyReLU
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class LeakyReLUTest(TritonKernelTest):
    """
    Test class for LeakyReLU activation.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 M: int,
                 N: int,
                 D: int,
                 negative_slope: float = 0.01,):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        self.negative_slope = negative_slope
        super().__init__(config)

    def setup_modules(self) -> None:
        self.leaky_relu_triton = LeakyReLU(negative_slope=self.negative_slope)
        self.leaky_relu_torch = nn.LeakyReLU(negative_slope=self.negative_slope)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.leaky_relu_torch(x)

class TestLeakyReLU:
    """
    Pytest test class for LeakyReLU activation.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 256, 256

    @parametrize_dtypes([torch.float16, torch.float32, torch.float64])
    @parametrize_dims([32, 64, 128, 256, 512, 1024, 2048])
    def test_leaky_relu(self,
                        dtype: torch.dtype,
                        dim: int,
                        base_shape: tuple[int, int, int],
                        test_mode: str,
                        warmup_iterations: int,
                        show_timing: bool,
                        device: str,
                        seed: int,):
        """
        Test LeakyReLU forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = LeakyReLUTest(config, B, M, N, dim)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize("negative_slope", [0.01, 0.1, 0.2])
    def test_leaky_relu_slopes(self,
                               dtype: torch.dtype,
                               negative_slope: float,
                               base_shape: tuple[int, int, int],
                               test_mode: str,
                               warmup_iterations: int,
                               show_timing: bool,
                               device: str,
                               seed: int,):
        """
        Test LeakyReLU with different negative slopes.
        """
        B, M, N = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = LeakyReLUTest(config, B, M, N, D, negative_slope=negative_slope)
        test.run()