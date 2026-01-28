from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonHub.Activations import Threshold
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class ThresholdTest(TritonKernelTest):
    """
    Test class for Threshold activation.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 M: int,
                 N: int,
                 D: int,
                 threshold: float = 0.5,
                 value: float = 0.0,):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        self.threshold = threshold
        self.value = value
        super().__init__(config)

    def setup_modules(self) -> None:
        self.threshold_triton = Threshold(threshold=self.threshold, value=self.value)
        self.threshold_torch = nn.Threshold(threshold=self.threshold, value=self.value)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.threshold_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.threshold_torch(x)

class TestThreshold:
    """
    Pytest test class for Threshold activation.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 256, 256

    @parametrize_dtypes([torch.float16, torch.float32, torch.float64])
    @parametrize_dims([32, 64, 128, 256, 512, 1024, 2048])
    def test_threshold(self,
                       dtype: torch.dtype,
                       dim: int,
                       base_shape: tuple[int, int, int],
                       test_mode: str,
                       warmup_iterations: int,
                       show_timing: bool,
                       device: str,
                       seed: int,):
        """
        Test Threshold forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = ThresholdTest(config, B, M, N, dim)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32, torch.float64])
    @pytest.mark.parametrize("threshold,value", [(0.0, 0.0), (0.5, 0.1), (1.0, -1.0)])
    def test_threshold_params(self,
                              dtype: torch.dtype,
                              threshold: float,
                              value: float,
                              base_shape: tuple[int, int, int],
                              test_mode: str,
                              warmup_iterations: int,
                              show_timing: bool,
                              device: str,
                              seed: int,):
        """
        Test Threshold with different threshold and value parameters.
        """
        B, M, N = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = ThresholdTest(config, B, M, N, D, threshold=threshold, value=value)
        test.run()