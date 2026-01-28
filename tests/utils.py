from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Any
import pytest
import torch

@dataclass
class RunConfig:
    """Configuration for a test run."""
    dtype: torch.dtype
    device: str = "cuda"
    seed: int = 42
    warmup_iterations: int = 3
    test_mode: str = "full"  # "forward", "backward", or "full"
    show_timing: bool = False
    rtol: float | None = None
    atol: float | None = None

    def __post_init__(self):
        if self.rtol is None or self.atol is None:
            self.rtol, self.atol = get_tolerances(self.dtype)

@dataclass
class BenchmarkResult:
    """
    Container for benchmark timing results.
    """
    triton_time_ms: float = 0.0
    torch_time_ms: float = 0.0
    mean_diff: float = 0.0
    max_diff: float = 0.0

    @property
    def speedup(self) -> float:
        """
        Calculate speedup ratio (torch_time / triton_time).
        """
        if self.triton_time_ms == 0:
            return 0.0
        return self.torch_time_ms / self.triton_time_ms

@dataclass
class TestResults:
    """
    Container for complete test results including forward and backward passes.
    """
    forward: BenchmarkResult = field(default_factory=BenchmarkResult)
    backward: BenchmarkResult = field(default_factory=BenchmarkResult)
    backward_extra: dict[str, BenchmarkResult] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """
        Summary dictionary of all results.
        """
        return {"forward": dataclasses.asdict(self.forward),
                "backward": dataclasses.asdict(self.backward),
                "backward_extra": {k: dataclasses.asdict(v) for k, v in self.backward_extra.items()},}

class CUDATimer:
    """
    CUDA event-based timer for GPU timing.
    """
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self) -> CUDATimer:
        self.start_event.record()
        return self

    def __exit__(self, *args) -> None:
        self.end_event.record()
        torch.cuda.synchronize()

    @property
    def elapsed_ms(self) -> float:
        """
        Return elapsed time in milliseconds.
        """
        return self.start_event.elapsed_time(self.end_event)

def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    """
    rtol and atol values for a given dtype.

    Args:
        dtype: PyTorch dtype.

    Returns:
        Tuple of (rtol, atol).
    """
    if dtype == torch.float16:
        return 1e-2, 5e-2
    elif dtype == torch.bfloat16:
        return 1e-2, 5e-2
    elif dtype == torch.float32:
        return 3e-4, 1e-3
    elif dtype == torch.float64:
        return 1e-5, 1e-5
    else:
        return 1e-3, 1e-2

def clear_cuda_cache() -> None:
    """
    Clear CUDA cache and synchronize.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def parametrize_dtypes(dtypes: list[torch.dtype] | None = None):
    """
    Decorator to parametrize tests with dtypes.

    Args:
        dtypes: List of torch dtypes to parametrize. If None, uses [float16, float32].
    """
    if dtypes is None:
        dtypes = [torch.float16, torch.float32]
    return pytest.mark.parametrize("dtype", dtypes, ids=lambda d: str(d).split(".")[-1])


def parametrize_dims(dims: list[int] | None = None):
    """
    Decorator to parametrize tests with dimensions.

    Args:
        dims: List of dimension sizes to parametrize. If None, uses [32, 64, 128, 256, 512].
    """
    if dims is None:
        dims = [32, 64, 128, 256, 512]
    return pytest.mark.parametrize("dim", dims)
