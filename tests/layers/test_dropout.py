from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonHub.Layers import Dropout
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims


class DropoutTest(TritonKernelTest):
    """
    Test class for Dropout layer.

    Note: Dropout is stochastic, so we test deterministic edge cases (p=0, p=1)
    and verify statistical properties for other values.
    """
    def __init__(self, config: RunConfig, B: int, M: int, N: int, D: int, p: float = 0.0):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        self.p = p
        super().__init__(config)

    def setup_modules(self) -> None:
        self.dropout_triton = Dropout(p=self.p)
        self.dropout_triton.train()
        self.dropout_torch = nn.Dropout(p=self.p)
        self.dropout_torch.train()

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        self._original_data = data
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout_torch(x)

    def _assert_forward(self,
                        triton_output: torch.Tensor,
                        torch_output: torch.Tensor) -> None:
        """
        Custom forward assertion for dropout.
        - p=0: outputs should match input exactly
        - p=1: outputs should be all zeros
        - other p: check statistical properties
        """
        if self.p == 0.0:
            # No dropout - should match input
            assert torch.allclose(triton_output, self._original_data,
                                  rtol=self.config.rtol, atol=self.config.atol)
        elif self.p == 1.0:
            # Full dropout - should be all zeros
            assert torch.all(triton_output == 0)
        else:
            # Check proportion of zeros (allow 10% tolerance)
            zero_proportion = (triton_output == 0).float().mean().item()
            assert abs(zero_proportion - self.p) < 0.1, \
                   f"Zero proportion {zero_proportion:.3f} deviates from p={self.p}"
            # Check scaling of non-zero values
            non_zero_mask = triton_output != 0
            if non_zero_mask.any():
                expected_scale = 1.0 / (1.0 - self.p)
                actual_values = triton_output[non_zero_mask]
                input_values = self._original_data[non_zero_mask]
                scale_ratio = (actual_values / input_values).mean().item()
                assert abs(scale_ratio - expected_scale) < 0.01, \
                       f"Scale ratio {scale_ratio:.3f} doesn't match expected {expected_scale:.3f}"

    def _assert_backward(self,
                         triton_inputs: tuple[torch.Tensor, ...],
                         torch_inputs: tuple[torch.Tensor, ...]) -> None:
        """
        Custom backward assertion for dropout.
        """
        triton_grad = triton_inputs[0].grad
        if self.p == 0.0:
            # No dropout - gradients should match
            torch_grad = torch_inputs[0].grad
            assert torch.allclose(triton_grad, torch_grad,
                                  rtol=self.config.rtol, atol=self.config.atol)
        elif self.p == 1.0:
            # Full dropout - gradients should be all zeros
            assert torch.all(triton_grad == 0)
        else:
            # Check proportion of zeros in gradient
            zero_proportion = (triton_grad == 0).float().mean().item()
            assert abs(zero_proportion - self.p) < 0.1, \
                   f"Gradient zero proportion {zero_proportion:.3f} deviates from p={self.p}"

class TestDropout:
    """
    Pytest test class for Dropout layer.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 4, 128, 128

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([64, 128, 256, 512, 1024])
    def test_dropout_no_drop(self,
                             dtype: torch.dtype,
                             dim: int,
                             base_shape: tuple[int, int, int],
                             test_mode: str,
                             warmup_iterations: int,
                             show_timing: bool,
                             device: str,
                             seed: int):
        """
        Test Dropout with p=0 (no dropout, deterministic).
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing)
        test = DropoutTest(config, B, M, N, dim, p=0.0)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5, 0.7])
    def test_dropout_statistical(self,
                                 dtype: torch.dtype,
                                 p: float,
                                 base_shape: tuple[int, int, int],
                                 test_mode: str,
                                 warmup_iterations: int,
                                 show_timing: bool,
                                 device: str,
                                 seed: int):
        """
        Test Dropout with various p values (statistical check).
        """
        B, M, N = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing)
        test = DropoutTest(config, B, M, N, D, p=p)
        test.run()