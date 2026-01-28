from __future__ import annotations
import pytest
import torch
from TritonHub.Ops import norm
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class NormTest(TritonKernelTest):
    """
    Test class for norm operation.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 M: int,
                 D: int,
                 p: int = 2,
                 eps: float = 1e-12,):
        self.B = B
        self.M = M
        self.D = D
        self.p = p
        self.eps = eps
        super().__init__(config)

    def setup_modules(self) -> None:
        self.norm_triton = norm(p=self.p, eps=self.eps)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, p=self.p, dim=-1, keepdim=True).squeeze(-1)

    def run_backward(self,
                     triton_output: torch.Tensor,
                     torch_output: torch.Tensor,
                     triton_inputs: tuple[torch.Tensor, ...],
                     torch_inputs: tuple[torch.Tensor, ...],
                     do_warmup: bool = True,) -> None:
        """
        Override to handle the keepdim difference in gradient.
        """
        if do_warmup:
            self._warmup_backward(triton_inputs, torch_inputs)

        # Need to match dimensions for backward
        grad = torch.randn_like(triton_output)
        grad_torch = grad.unsqueeze(-1)  # Add dimension for torch norm

        # Time Triton backward
        with self.timer:
            triton_output.backward(grad)
        self.results.backward.triton_time_ms = self.timer.elapsed_ms

        # For torch, we need to compute the norm again with keepdim=True
        torch_norm_output = torch.norm(torch_inputs[0], p=self.p, dim=-1, keepdim=True)

        # Time PyTorch backward
        with self.timer:
            torch_norm_output.backward(grad_torch)
        self.results.backward.torch_time_ms = self.timer.elapsed_ms

        # Compute gradient differences
        tri_inp = triton_inputs[0]
        tor_inp = torch_inputs[0]

        diff = (tri_inp.grad - tor_inp.grad).abs()
        self.results.backward.mean_diff = diff.mean().item()
        self.results.backward.max_diff = diff.max().item()

class TestNorm:
    """
    Pytest test class for norm operation.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int]:
        return 1, 1000

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([32, 64, 128, 256, 512, 1024])
    def test_norm(self,
                  dtype: torch.dtype,
                  dim: int,
                  base_shape: tuple[int, int],
                  test_mode: str,
                  warmup_iterations: int,
                  show_timing: bool,
                  device: str,
                  seed: int,):
        """
        Test norm forward and backward.
        """
        B, M = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = NormTest(config, B, M, dim)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("p", [1, 2, 6])
    def test_norm_p_values(self,
                           dtype: torch.dtype,
                           p: float,
                           base_shape: tuple[int, int],
                           test_mode: str,
                           warmup_iterations: int,
                           show_timing: bool,
                           device: str,
                           seed: int,):
        """
        Test norm with different p values.
        """
        B, M = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = NormTest(config, B, M, D, p=int(p))
        test.run()