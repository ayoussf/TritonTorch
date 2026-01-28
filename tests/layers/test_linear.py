from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonHub.Layers import Linear
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class LinearTest(TritonKernelTest):
    """
    Test class for Linear layer.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 M: int,
                 N: int,
                 D: int,
                 out_features: int | None = None,
                 bias: bool = True,):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        self.out_features = out_features if out_features is not None else 2 * D
        self.bias = bias
        super().__init__(config)

    def setup_modules(self) -> None:
        self.linear_triton = Linear(in_features=self.D,
                                   out_features=self.out_features,
                                   bias=self.bias,
                                   device=self.config.device,
                                   dtype=self.config.dtype,)
        self.linear_torch = nn.Linear(in_features=self.D,
                                      out_features=self.out_features,
                                      bias=self.bias,
                                      device=self.config.device,
                                      dtype=self.config.dtype,)
        # Copy weights to ensure same initialization
        with torch.no_grad():
            self.linear_torch.weight.copy_(self.linear_triton.weight.T)
            if self.bias:
                self.linear_torch.bias.copy_(self.linear_triton.bias)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_torch(x)

class TestLinear:
    """
    Pytest test class for Linear layer.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 256, 256

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([32, 64, 128, 256, 512, 1024, 2048])
    def test_linear(self,
                    dtype: torch.dtype,
                    dim: int,
                    base_shape: tuple[int, int, int],
                    test_mode: str,
                    warmup_iterations: int,
                    show_timing: bool,
                    device: str,
                    seed: int,):
        """
        Test Linear forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = LinearTest(config, B, M, N, dim)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("bias", [True, False])
    def test_linear_bias(self,
                         dtype: torch.dtype,
                         bias: bool,
                         base_shape: tuple[int, int, int],
                         test_mode: str,
                         warmup_iterations: int,
                         show_timing: bool,
                         device: str,
                         seed: int,):
        """
        Test Linear with and without bias.
        """
        B, M, N = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = LinearTest(config, B, M, N, D, bias=bias)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("out_features", [128, 512, 1024])
    def test_linear_out_features(self,
                                 dtype: torch.dtype,
                                 out_features: int,
                                 base_shape: tuple[int, int, int],
                                 test_mode: str,
                                 warmup_iterations: int,
                                 show_timing: bool,
                                 device: str,
                                 seed: int,):
        """
        Test Linear with different output feature sizes.
        """
        B, M, N = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = LinearTest(config, B, M, N, D, out_features=out_features)
        test.run()