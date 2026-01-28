from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from TritonHub.Layers import MHA
from tests.base import TritonKernelTest
from tests.utils import RunConfig, parametrize_dtypes

class MHATest(TritonKernelTest):
    """
    Test class for Multi-Head Attention layer.

    Note: Dropout is set to 0 for deterministic comparison.
    """
    def __init__(self,
                 config: RunConfig,
                 B: int,
                 seq_len: int,
                 embed_dim: int,
                 num_heads: int,
                 bias: bool = True):
        self.B = B
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias
        super().__init__(config)

    def setup_modules(self) -> None:
        self.mha_triton = MHA(embed_dim=self.embed_dim,
                              num_heads=self.num_heads,
                              qkv_proj_bias=self.bias,
                              out_proj_bias=self.bias,
                              dropout=0.0,
                              device=self.config.device,
                              dtype=self.config.dtype)
        self.mha_torch = nn.MultiheadAttention(embed_dim=self.embed_dim,
                                               num_heads=self.num_heads,
                                               bias=self.bias,
                                               dropout=0.0,
                                               batch_first=True,
                                               device=self.config.device,
                                               dtype=self.config.dtype)
        
        # Copy weights from Triton to PyTorch
        with torch.no_grad():
            if hasattr(self.mha_triton, 'in_proj'):
                self.mha_torch.in_proj_weight.copy_(self.mha_triton.in_proj.weight.T)
                if self.bias:
                    self.mha_torch.in_proj_bias.copy_(self.mha_triton.in_proj.bias)
            else:
                E = self.embed_dim
                self.mha_torch.in_proj_weight[:E].copy_(self.mha_triton.q_proj.weight.T)
                self.mha_torch.in_proj_weight[E:2*E].copy_(self.mha_triton.k_proj.weight.T)
                self.mha_torch.in_proj_weight[2*E:].copy_(self.mha_triton.v_proj.weight.T)
                if self.bias:
                    self.mha_torch.in_proj_bias[:E].copy_(self.mha_triton.q_proj.bias)
                    self.mha_torch.in_proj_bias[E:2*E].copy_(self.mha_triton.k_proj.bias)
                    self.mha_torch.in_proj_bias[2*E:].copy_(self.mha_triton.v_proj.bias)
            
            self.mha_torch.out_proj.weight.copy_(self.mha_triton.out_proj.weight.T)
            if self.bias:
                self.mha_torch.out_proj.bias.copy_(self.mha_triton.out_proj.bias)

    def create_inputs(self) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        # Self-attention: q = k = v
        data = torch.randn(self.B, self.seq_len, self.embed_dim,
                           device=self.config.device,
                           dtype=self.config.dtype)
        triton_input = data.clone().detach().requires_grad_(True)
        torch_input = data.clone().detach().requires_grad_(True)
        return (triton_input,), (torch_input,)

    def forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        return self.mha_triton(x, x, x)

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.mha_torch(x, x, x, need_weights=False)
        return output

class TestMHA:
    """
    Pytest test class for Multi-Head Attention layer.
    """
    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def seq_len(self) -> int:
        return 64

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("embed_dim,num_heads", [(256, 4), (512, 8), (768, 12)])
    def test_mha_self_attention(self,
                                dtype: torch.dtype,
                                embed_dim: int,
                                num_heads: int,
                                batch_size: int,
                                seq_len: int,
                                test_mode: str,
                                warmup_iterations: int,
                                show_timing: bool,
                                device: str,
                                seed: int):
        """
        Test MHA self-attention forward and backward.
        """
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2)
        test = MHATest(config, batch_size, seq_len, embed_dim, num_heads)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("bias", [True, False])
    def test_mha_bias(self,
                      dtype: torch.dtype,
                      bias: bool,
                      batch_size: int,
                      seq_len: int,
                      test_mode: str,
                      warmup_iterations: int,
                      show_timing: bool,
                      device: str,
                      seed: int):
        """
        Test MHA with and without bias.
        """
        embed_dim = 256
        num_heads = 4
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2)
        test = MHATest(config, batch_size, seq_len, embed_dim, num_heads, bias=bias)
        test.run()

    @parametrize_dtypes([torch.float16, torch.float32])
    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256])
    def test_mha_seq_lengths(self,
                             dtype: torch.dtype,
                             seq_len: int,
                             batch_size: int,
                             test_mode: str,
                             warmup_iterations: int,
                             show_timing: bool,
                             device: str,
                             seed: int):
        """
        Test MHA with different sequence lengths.
        """
        embed_dim = 256
        num_heads = 4
        config = RunConfig(dtype=dtype,
                           device=device,
                           seed=seed,
                           warmup_iterations=warmup_iterations,
                           test_mode=test_mode,
                           show_timing=show_timing,
                           rtol=0.0,
                           atol=1e-2)
        test = MHATest(config, batch_size, seq_len, embed_dim, num_heads)
        test.run()