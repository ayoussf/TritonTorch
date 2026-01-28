from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from tests.utils import RunConfig, BenchmarkResult, CUDATimer, TestResults, clear_cuda_cache

class TritonKernelTest(ABC):
    """
    Base class for testing Triton kernels against PyTorch implementations.

    Subclasses must implement:
    - setup_modules(): Initialize Triton and PyTorch modules
    - create_inputs(): Create input tensors for testing
    - forward_triton(): Run Triton forward pass
    - forward_torch(): Run PyTorch forward pass
    """
    def __init__(self, config: RunConfig):
        self.config = config
        self.timer = CUDATimer()
        self.results = TestResults()

        # Set random seed
        torch.manual_seed(config.seed)

        # Setup modules
        self.setup_modules()

    @abstractmethod
    def setup_modules(self) -> None:
        """
        Initialize Triton and PyTorch modules for comparison.
        """
        pass

    @abstractmethod
    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """
        Create input tensors for testing.

        Returns:
            Tuple of (triton_inputs, torch_inputs).
        """
        pass

    @abstractmethod
    def forward_triton(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Run Triton forward pass.
        """
        pass

    @abstractmethod
    def forward_torch(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Run PyTorch forward pass.
        """
        pass

    def _warmup_forward(self,
                        triton_inputs: tuple[torch.Tensor, ...],
                        torch_inputs: tuple[torch.Tensor, ...],) -> None:
        """
        Perform warmup iterations for forward pass.
        """
        for _ in range(self.config.warmup_iterations):
            with torch.no_grad():
                _ = self.forward_triton(*triton_inputs)
                _ = self.forward_torch(*torch_inputs)
            torch.cuda.synchronize()

    def _warmup_backward(self,
                         triton_inputs: tuple[torch.Tensor, ...],
                         torch_inputs: tuple[torch.Tensor, ...],) -> None:
        """
        Perform warmup iterations for backward pass.
        """
        for _ in range(self.config.warmup_iterations):
            # Create new inputs for warmup to avoid grad accumulation issues
            tri_warmup = tuple(t.clone().detach().requires_grad_(t.requires_grad)
                               for t in triton_inputs)
            tor_warmup = tuple(t.clone().detach().requires_grad_(t.requires_grad)
                               for t in torch_inputs)

            # Run forward and backward
            tri_out = self.forward_triton(*tri_warmup)
            tor_out = self.forward_torch(*tor_warmup)

            grad = torch.randn_like(tri_out)
            tri_out.backward(grad)
            tor_out.backward(grad)
            torch.cuda.synchronize()

    def run_forward(self,
                    triton_inputs: tuple[torch.Tensor, ...],
                    torch_inputs: tuple[torch.Tensor, ...],
                    do_warmup: bool = True,) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run and time forward passes.

        Args:
            triton_inputs: Input tensors for Triton.
            torch_inputs: Input tensors for PyTorch.
            do_warmup: Whether to perform warmup.

        Returns:
            Tuple of (triton_output, torch_output).
        """
        if do_warmup:
            self._warmup_forward(triton_inputs, torch_inputs)

        # Time Triton forward
        with self.timer:
            triton_output = self.forward_triton(*triton_inputs)
        self.results.forward.triton_time_ms = self.timer.elapsed_ms

        # Time PyTorch forward
        with self.timer:
            torch_output = self.forward_torch(*torch_inputs)
        self.results.forward.torch_time_ms = self.timer.elapsed_ms

        # Compute differences
        diff = (triton_output - torch_output).abs()
        self.results.forward.mean_diff = diff.mean().item()
        self.results.forward.max_diff = diff.max().item()

        return triton_output, torch_output

    def run_backward(self,
                     triton_output: torch.Tensor,
                     torch_output: torch.Tensor,
                     triton_inputs: tuple[torch.Tensor, ...],
                     torch_inputs: tuple[torch.Tensor, ...],
                     do_warmup: bool = True,) -> None:
        """
        Run and time backward passes.

        Args:
            triton_output: Output from Triton forward pass.
            torch_output: Output from PyTorch forward pass.
            triton_inputs: Input tensors (for gradient comparison).
            torch_inputs: Input tensors (for gradient comparison).
            do_warmup: Whether to perform warmup.
        """
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

        # Compute gradient differences for each input
        for i, (tri_inp, tor_inp) in enumerate(zip(triton_inputs, torch_inputs)):
            if tri_inp.grad is not None and tor_inp.grad is not None:
                diff = (tri_inp.grad - tor_inp.grad).abs()
                result = BenchmarkResult(triton_time_ms=self.results.backward.triton_time_ms,
                                         torch_time_ms=self.results.backward.torch_time_ms,
                                         mean_diff=diff.mean().item(),
                                         max_diff=diff.max().item(),)
                if i == 0:
                    self.results.backward = result
                else:
                    self.results.backward_extra[f"input_{i}"] = result

    def run(self) -> TestResults:
        """
        Run the complete test according to the configured test mode.
        
        Returns:
            TestResults object with all timing and accuracy information.
        """
        # Clear CUDA cache for consistent memory state
        clear_cuda_cache()

        triton_inputs, torch_inputs = self.create_inputs()

        if self.config.test_mode == "forward":
            # Forward only: warmup then test forward
            triton_output, torch_output = self.run_forward(triton_inputs,
                                                           torch_inputs,
                                                           do_warmup=True)
            self._assert_forward(triton_output, torch_output)

        elif self.config.test_mode == "backward":
            # Backward only: need forward first (no timing), then warmup backward
            with torch.no_grad():
                pass  # No pre-forward needed, warmup does its own

            # Create fresh inputs for the actual backward test
            triton_inputs, torch_inputs = self.create_inputs()
            triton_output = self.forward_triton(*triton_inputs)
            torch_output = self.forward_torch(*torch_inputs)

            self.run_backward(triton_output,
                              torch_output,
                              triton_inputs,
                              torch_inputs,
                              do_warmup=True)
            self._assert_backward(triton_inputs, torch_inputs)

        else:  # "full"
            # Full test: warmup both forward and backward, then time both
            # Warmup forward+backward together (uses fresh inputs each iteration)
            self._warmup_backward(triton_inputs, torch_inputs)

            # Create fresh inputs for the actual timed test
            triton_inputs, torch_inputs = self.create_inputs()

            # Timed forward (no additional warmup needed)
            triton_output, torch_output = self.run_forward(triton_inputs,
                                                           torch_inputs,
                                                           do_warmup=False)
            self._assert_forward(triton_output, torch_output)

            # Timed backward (no additional warmup needed)
            self.run_backward(triton_output,
                              torch_output,
                              triton_inputs,
                              torch_inputs,
                              do_warmup=False)
            self._assert_backward(triton_inputs, torch_inputs)

        if self.config.show_timing:
            self.print_results()

        return self.results

    def _assert_forward(self,
                        triton_output: torch.Tensor,
                        torch_output: torch.Tensor) -> None:
        """
        Assert forward pass correctness.
        """
        assert torch.allclose(triton_output, torch_output, 
                              rtol=self.config.rtol, atol=self.config.atol), \
        (f"Forward pass mismatch: "
         f"mean_diff={self.results.forward.mean_diff:.6e}, "
         f"max_diff={self.results.forward.max_diff:.6e}")

    def _assert_backward(self,
                         triton_inputs: tuple[torch.Tensor, ...],
                         torch_inputs: tuple[torch.Tensor, ...],) -> None:
        """
        Assert backward pass correctness.
        """
        for i, (tri_inp, tor_inp) in enumerate(zip(triton_inputs, torch_inputs)):
            if tri_inp.grad is not None and tor_inp.grad is not None:
                assert torch.allclose(tri_inp.grad, tor_inp.grad,
                                      rtol=self.config.rtol, atol=self.config.atol), \
                (f"Backward pass mismatch for input {i}: "
                 f"mean_diff={self.results.backward.mean_diff:.6e}, "
                 f"max_diff={self.results.backward.max_diff:.6e}")

    def print_results(self) -> None:
        """
        Print timing results.
        """
        fwd = self.results.forward
        bwd = self.results.backward

        def speedup(torch_ms: float, triton_ms: float) -> str:
            return f"{torch_ms / triton_ms:.2f}x" if triton_ms > 0 else "N/A"

        sections = []

        if self.config.test_mode in ("forward", "full"):
            sections.append(f"Fwd: Triton {fwd.triton_time_ms:.2f}ms | Torch {fwd.torch_time_ms:.2f}ms | {speedup(fwd.torch_time_ms, fwd.triton_time_ms)} | diff {fwd.max_diff:.2e}")

        if self.config.test_mode in ("backward", "full"):
            sections.append(f"Bwd: Triton {bwd.triton_time_ms:.2f}ms | Torch {bwd.torch_time_ms:.2f}ms | {speedup(bwd.torch_time_ms, bwd.triton_time_ms)} | diff {bwd.max_diff:.2e}")

        print(" || ".join(sections))