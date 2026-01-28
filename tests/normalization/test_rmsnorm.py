from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonHub.Normalization import RMSNorm
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class RMSNormTest(TritonKernelTest):
    """
    Test class for RMSNorm normalization.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 M: int,
                 N: int,
                 D: int,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        super().__init__(config)

    def setup_modules(self) -> None:
        self.rmsnorm_triton = RMSNorm(dimension=self.D,
                                      eps=self.eps,
                                      elementwise_affine=self.elementwise_affine,
                                      device=self.config.device,
                                      dtype=self.config.dtype,)
        self.rmsnorm_torch = nn.RMSNorm(self.D,
                                        eps=self.eps,
                                        elementwise_affine=self.elementwise_affine,
                                        device=self.config.device,
                                        dtype=self.config.dtype,)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm_torch(x)

class TestRMSNorm:
    """
    Pytest test class for RMSNorm normalization.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 256, 256

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([32, 64, 128, 256, 512, 1024, 2048])
    def test_rmsnorm(self,
                     dtype: torch.dtype,
                     dim: int,
                     base_shape: tuple[int, int, int],
                     test_mode: str,
                     warmup_iterations: int,
                     show_timing: bool,
                     device: str,
                     seed: int,):
        """
        Test RMSNorm forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = RMSNormTest(config, B, M, N, dim)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("eps", [1e-5, 1e-6, 1e-8])
    def test_rmsnorm_eps(self,
                         dtype: torch.dtype,
                         eps: float,
                         base_shape: tuple[int, int, int],
                         test_mode: str,
                         warmup_iterations: int,
                         show_timing: bool,
                         device: str,
                         seed: int,):
        """
        Test RMSNorm with different epsilon values.
        """
        B, M, N = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = RMSNormTest(config, B, M, N, D, eps=eps)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("elementwise_affine", [True, False])
    def test_rmsnorm_affine(self,
                            dtype: torch.dtype,
                            elementwise_affine: bool,
                            base_shape: tuple[int, int, int],
                            test_mode: str,
                            warmup_iterations: int,
                            show_timing: bool,
                            device: str,
                            seed: int,):
        """
        Test RMSNorm with and without elementwise affine.
        """
        B, M, N = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = RMSNormTest(config, B, M, N, D, elementwise_affine=elementwise_affine)
        test.run()