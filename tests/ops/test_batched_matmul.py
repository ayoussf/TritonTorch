from __future__ import annotations
import pytest
import torch
from TritonHub.Ops import bmm
from tests.base import TritonKernelTest, RunConfig
from tests.utils import RunConfig, BenchmarkResult, parametrize_dtypes, parametrize_dims

class BMMTest(TritonKernelTest):
    """
    Test class for batched matrix multiplication operation.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 M: int,
                 N: int,
                 D: int,):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        super().__init__(config)

    def setup_modules(self) -> None:
        pass  # No modules to set up for bmm

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data_x = torch.randn(self.B,self.M, self.D,
                             device=self.config.device,
                             dtype=self.config.dtype,)
        data_y = torch.randn(self.B, self.N, self.D,
                             device=self.config.device,
                             dtype=self.config.dtype,)
        triton_x = data_x.clone().detach().requires_grad_(True)
        triton_y = data_y.clone().detach().requires_grad_(True)
        torch_x = data_x.clone().detach().requires_grad_(True)
        torch_y = data_y.clone().detach().requires_grad_(True)
        return (triton_x, triton_y), (torch_x, torch_y)

    def forward_triton(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return bmm(x, y)

    def forward_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bmd,bnd->bmn', x, y)

    def run_backward(self,
                     triton_output: torch.Tensor,
                     torch_output: torch.Tensor,
                     triton_inputs: tuple[torch.Tensor, ...],
                     torch_inputs: tuple[torch.Tensor, ...],
                     do_warmup: bool = True,) -> None:
        """Override to handle multiple input gradients."""
        if do_warmup:
            self._warmup_backward(triton_inputs, torch_inputs)

        grad = torch.randn_like(triton_output)

        # Time Triton backward
        with self.timer:
            triton_output.backward(grad)
        self.results.backward.triton_time_ms = self.timer.elapsed_ms

        # Time PyTorch backward
        with self.timer:
            torch_output.backward(grad)
        self.results.backward.torch_time_ms = self.timer.elapsed_ms

        # Compute gradient differences for x
        tri_x, tri_y = triton_inputs
        tor_x, tor_y = torch_inputs

        diff_x = (tri_x.grad - tor_x.grad).abs()
        self.results.backward = BenchmarkResult(triton_time_ms=self.results.backward.triton_time_ms,
                                                torch_time_ms=self.results.backward.torch_time_ms,
                                                mean_diff=diff_x.mean().item(),
                                                max_diff=diff_x.max().item(),)
        
        # Compute gradient differences for y
        diff_y = (tri_y.grad - tor_y.grad).abs()
        self.results.backward_extra["input_y"] = BenchmarkResult(triton_time_ms=self.results.backward.triton_time_ms,
                                                                 torch_time_ms=self.results.backward.torch_time_ms,
                                                                 mean_diff=diff_y.mean().item(),
                                                                 max_diff=diff_y.max().item(),)
    
    def _assert_backward(self,
                         triton_inputs: tuple[torch.Tensor, ...],
                         torch_inputs: tuple[torch.Tensor, ...],) -> None:
        """
        Assert backward pass correctness for both inputs.
        """
        tri_x, tri_y = triton_inputs
        tor_x, tor_y = torch_inputs

        assert torch.allclose(tri_x.grad, tor_x.grad,
                              rtol=self.config.rtol, atol=self.config.atol), \
               f"Backward pass mismatch for input x: mean_diff={self.results.backward.mean_diff:.6e}"

        assert torch.allclose(tri_y.grad, tor_y.grad,
                              rtol=self.config.rtol, atol=self.config.atol), \
               f"Backward pass mismatch for input y: mean_diff={self.results.backward_extra['input_y'].mean_diff:.6e}"

class TestBMM:
    """
    Pytest test class for batched matrix multiplication operation.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 10, 10

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([32, 64, 128, 256, 512])
    def test_bmm(self,
                 dtype: torch.dtype,
                 dim: int,
                 base_shape: tuple[int, int, int],
                 test_mode: str,
                 warmup_iterations: int,
                 show_timing: bool,
                 device: str,
                 seed: int,):
        """
        Test bmm forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-1 if dtype == torch.float32 else 8e-3,)
        test = BMMTest(config, B, M, N, dim)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_bmm_batch_sizes(self,
                             dtype: torch.dtype,
                             batch_size: int,
                             base_shape: tuple[int, int, int],
                             test_mode: str,
                             warmup_iterations: int,
                             show_timing: bool,
                             device: str,
                             seed: int,):
        """
        Test bmm with different batch sizes.
        """
        _, M, N = base_shape
        D = 128
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-1,)
        test = BMMTest(config, batch_size, M, N, D)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("m,n", [(10, 10), (32, 32), (64, 64), (128, 64)])
    def test_bmm_shapes(self,
                        dtype: torch.dtype,
                        m: int,
                        n: int,
                        test_mode: str,
                        warmup_iterations: int,
                        show_timing: bool,
                        device: str,
                        seed: int,):
        """
        Test bmm with different M and N dimensions.
        """
        B = 1
        D = 128
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-1,)
        test = BMMTest(config, B, m, n, D)
        test.run()