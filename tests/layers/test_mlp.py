from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonTorch.Layers import MLP
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes, parametrize_dims

class TorchGatedMLP(nn.Module):
    """PyTorch MLP."""
    def __init__(self, in_features, hidden_features, out_features, activation, bias, device, dtype):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device, dtype=dtype)
        self.activation = activation

    def forward(self, x):
        y = self.fc1(x)
        y, gate = y.chunk(2, dim=-1)
        y = y * self.activation(gate)
        y = self.fc2(y)
        return y

class TorchFFN(nn.Module):
    """PyTorch FFN."""
    def __init__(self, in_features, hidden_features, out_features, activation, bias, device, dtype):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device, dtype=dtype)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device, dtype=dtype)
        self.activation = activation

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y

class MLPTest(TritonKernelTest):
    """
    Test class for MLP layer.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 M: int,
                 N: int,
                 D: int,
                 hidden_features: int | None = None,
                 out_features: int | None = None,
                 mlp_type: str = "gated_mlp",
                 bias: bool = False):
        self.B = B
        self.M = M
        self.N = N
        self.D = D
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.mlp_type = mlp_type
        self.bias = bias
        super().__init__(config)

    def setup_modules(self) -> None:
        self.mlp_triton = MLP(in_features=self.D,
                              hidden_features=self.hidden_features,
                              out_features=self.out_features,
                              activation="SiLU",
                              bias=self.bias,
                              mlp_type=self.mlp_type,
                              device=self.config.device,
                              dtype=self.config.dtype)
        
        out_features = self.out_features if self.out_features is not None else self.D
        hidden_features = self.hidden_features if self.hidden_features is not None else int(8 * self.D / 3)
        hidden_features = (hidden_features + 128 - 1) // 128 * 128
        
        torch_activation = nn.SiLU()
        if self.mlp_type == "gated_mlp":
            self.mlp_torch = TorchGatedMLP(self.D, hidden_features, out_features,
                                           torch_activation, self.bias,
                                           self.config.device, self.config.dtype)
        else:
            self.mlp_torch = TorchFFN(self.D, hidden_features, out_features,
                                      torch_activation, self.bias,
                                      self.config.device, self.config.dtype)

        # Copy weights (Triton Linear stores weights transposed)
        with torch.no_grad():
            self.mlp_torch.fc1.weight.copy_(self.mlp_triton.fc1.weight.T)
            self.mlp_torch.fc2.weight.copy_(self.mlp_triton.fc2.weight.T)
            if self.bias:
                self.mlp_torch.fc1.bias.copy_(self.mlp_triton.fc1.bias)
                self.mlp_torch.fc2.bias.copy_(self.mlp_triton.fc2.bias)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.M, self.N, self.D,
                           device=self.config.device,
                           dtype=self.config.dtype)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_torch(x)


class TestMLP:
    """
    Pytest test class for MLP layer.
    """
    @pytest.fixture
    def base_shape(self) -> tuple[int, int, int]:
        return 1, 64, 64

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([128, 256, 512, 1024])
    def test_gated_mlp(self,
                       dtype: torch.dtype,
                       dim: int,
                       base_shape: tuple[int, int, int],
                       test_mode: str,
                       warmup_iterations: int,
                       show_timing: bool,
                       device: str,
                       seed: int):
        """
        Test gated MLP forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2)
        test = MLPTest(config, B, M, N, dim, mlp_type="gated_mlp")
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @parametrize_dims([128, 256, 512, 1024])
    def test_ffn(self,
                 dtype: torch.dtype,
                 dim: int,
                 base_shape: tuple[int, int, int],
                 test_mode: str,
                 warmup_iterations: int,
                 show_timing: bool,
                 device: str,
                 seed: int):
        """
        Test FFN forward and backward.
        """
        B, M, N = base_shape
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2)
        test = MLPTest(config, B, M, N, dim, mlp_type="ffn")
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("bias", [True, False])
    def test_mlp_bias(self,
                      dtype: torch.dtype,
                      bias: bool,
                      base_shape: tuple[int, int, int],
                      test_mode: str,
                      warmup_iterations: int,
                      show_timing: bool,
                      device: str,
                      seed: int):
        """
        Test MLP with and without bias.
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
                           atol=1e-2)
        test = MLPTest(config, B, M, N, D, bias=bias)
        test.run()