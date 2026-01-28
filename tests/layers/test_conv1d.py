from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonHub.Layers.conv1d import Conv1d
from tests.base import TritonKernelTest 
from tests.utils import RunConfig, parametrize_dtypes

class Conv1dTest(TritonKernelTest):
    """
    Test class for Conv1d layer.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 C_in: int,
                 C_out: int,
                 L: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias: bool = True,):
        self.B = B
        self.C_in = C_in
        self.C_out = C_out
        self.L = L
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        super().__init__(config)

    def setup_modules(self) -> None:
        self.conv_triton = Conv1d(in_channels=self.C_in,
                                  out_channels=self.C_out,
                                  kernel_size=self.kernel_size,
                                  stride=self.stride,
                                  padding=self.padding,
                                  dilation=self.dilation,
                                  bias=self.bias,
                                  device=self.config.device,
                                  dtype=self.config.dtype,)
        self.conv_torch = nn.Conv1d(in_channels=self.C_in,
                                    out_channels=self.C_out,
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    bias=self.bias,
                                    device=self.config.device,
                                    dtype=self.config.dtype,)
        # Copy weights to ensure same initialization
        with torch.no_grad():
            self.conv_torch.weight.copy_(self.conv_triton.weight)
            if self.bias:
                self.conv_torch.bias.copy_(self.conv_triton.bias)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        data = torch.randn(self.B, self.C_in, self.L,
                           device=self.config.device,
                           dtype=self.config.dtype,)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_triton(x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_torch(x)

class TestConv1d:
    """
    Pytest test class for Conv1d layer.
    """
    @pytest.fixture
    def base_config(self) -> dict:
        return {"B": 1,
                "L": 1000,
                "C_in": 32,
                "C_out": 64,
                "kernel_size": 3}

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("c_in,c_out", [(32, 64), (64, 128), (128, 256), (256, 512), (512, 1024)])
    def test_conv1d_channels(self,
                             dtype: torch.dtype,
                             c_in: int,
                             c_out: int,
                             base_config: dict,
                             test_mode: str,
                             warmup_iterations: int,
                             show_timing: bool,
                             device: str,
                             seed: int,):
        """
        Test Conv1d with different channel configurations.
        """
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = Conv1dTest(config,
                          B=base_config["B"],
                          C_in=c_in,
                          C_out=c_out,
                          L=base_config["L"],
                          kernel_size=base_config["kernel_size"],)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("padding", [0, 5, 10])
    def test_conv1d_padding(self,
                            dtype: torch.dtype,
                            padding: int,
                            base_config: dict,
                            test_mode: str,
                            warmup_iterations: int,
                            show_timing: bool,
                            device: str,
                            seed: int,):
        """
        Test Conv1d with different padding values.
        """
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = Conv1dTest(config,
                          B=base_config["B"],
                          C_in=base_config["C_in"],
                          C_out=base_config["C_out"],
                          L=base_config["L"],
                          kernel_size=base_config["kernel_size"],
                          padding=padding,)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("stride", [1, 2])
    def test_conv1d_stride(self,
                           dtype: torch.dtype,
                           stride: int,
                           base_config: dict,
                           test_mode: str,
                           warmup_iterations: int,
                           show_timing: bool,
                           device: str,
                           seed: int,):
        """
        Test Conv1d with different stride values.
        """
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = Conv1dTest(config,
                          B=base_config["B"],
                          C_in=base_config["C_in"],
                          C_out=base_config["C_out"],
                          L=base_config["L"],
                          kernel_size=base_config["kernel_size"],
                          stride=stride,)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("dilation", [1, 2])
    def test_conv1d_dilation(self,
                             dtype: torch.dtype,
                             dilation: int,
                             base_config: dict,
                             test_mode: str,
                             warmup_iterations: int,
                             show_timing: bool,
                             device: str,
                             seed: int,):
        """
        Test Conv1d with different dilation values.
        """
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = Conv1dTest(config,
                          B=base_config["B"],
                          C_in=base_config["C_in"],
                          C_out=base_config["C_out"],
                          L=base_config["L"],
                          kernel_size=base_config["kernel_size"],
                          dilation=dilation,)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("bias", [True, False])
    def test_conv1d_bias(self,
                         dtype: torch.dtype,
                         bias: bool,
                         base_config: dict,
                         test_mode: str,
                         warmup_iterations: int,
                         show_timing: bool,
                         device: str,
                         seed: int,):
        """
        Test Conv1d with and without bias.
        """
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2,)
        test = Conv1dTest(config,
                          B=base_config["B"],
                          C_in=base_config["C_in"],
                          C_out=base_config["C_out"],
                          L=base_config["L"],
                          kernel_size=base_config["kernel_size"],
                          bias=bias,)
        test.run()