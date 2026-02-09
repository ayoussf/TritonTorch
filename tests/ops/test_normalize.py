from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F
from TritonTorch.Ops import normalize
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class NormalizeTest(TritonKernelTest):
    """
    Test class for normalize operation.
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
        self.normalize_triton = normalize(p=self.p, eps=self.eps)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=self.p, dim=-1, eps=self.eps)

class TestNormalize:
    """
    Pytest test class for normalize operation.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int]:
        return 1, 1000

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([32, 64, 128, 256, 512, 1024])
    def test_normalize(self,
                       dtype: torch.dtype,
                       dim: int,
                       base_shape: tuple[int, int],
                       test_mode: str,
                       warmup_iterations: int,
                       show_timing: bool,
                       device: str,
                       seed: int,):
        """
        Test normalize forward and backward.
        """
        B, M = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = NormalizeTest(config, B, M, dim)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("p", [1, 2])
    def test_normalize_p_values(self,
                                dtype: torch.dtype,
                                p: int,
                                base_shape: tuple[int, int],
                                test_mode: str,
                                warmup_iterations: int,
                                show_timing: bool,
                                device: str,
                                seed: int,):
        """
        Test normalize with different p values.
        """
        B, M = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = NormalizeTest(config, B, M, D, p=p)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("eps", [1e-12, 1e-8, 1e-6])
    def test_normalize_eps(self,
                           dtype: torch.dtype,
                           eps: float,
                           base_shape: tuple[int, int],
                           test_mode: str,
                           warmup_iterations: int,
                           show_timing: bool,
                           device: str,
                           seed: int,):
        """
        Test normalize with different epsilon values.
        """
        B, M = base_shape
        D = 256
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,)
        test = NormalizeTest(config, B, M, D, eps=eps)
        test.run()