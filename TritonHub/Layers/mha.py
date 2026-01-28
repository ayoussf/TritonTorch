import torch
import torch.nn as nn
import math
from einops import rearrange
from typing import Optional
from TritonHub.Layers import Linear, Dropout
from TritonHub.Layers.linear import _linear_fwd
from TritonHub.Ops import bmm
from TritonHub.Activations.softmax import Softmax

class MHA(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_heads_kv=None,
                 head_dim=None,
                 qkv_proj_bias=True,
                 out_proj_bias=True,
                 dropout=0.1,
                 softmax_scale=None,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads # GQA or MQA.
        assert (self.num_heads % self.num_heads_kv == 0), "num_heads must be divisible by num_heads_kv"
        if head_dim is None:
            assert (self.embed_dim % num_heads == 0), "embed_dim must be divisible by num_heads"
        self.head_dim = head_dim if head_dim is not None else self.embed_dim // num_heads
        q_dim = self.head_dim * self.num_heads
        k_dim = self.head_dim * self.num_heads_kv
        v_dim = self.head_dim * self.num_heads_kv
        out_dim = self.head_dim * self.num_heads
        self.softmax = Softmax()
        self.softmax_scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(self.head_dim)
        if k_dim == embed_dim and v_dim == embed_dim:
            self.in_proj = Linear(embed_dim, q_dim + k_dim + v_dim, bias=qkv_proj_bias, **factory_kwargs)
        else:
            self.q_proj = Linear(embed_dim, q_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.k_proj = Linear(embed_dim, k_dim, bias=qkv_proj_bias, **factory_kwargs)
            self.v_proj = Linear(embed_dim, v_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = Linear(out_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        if hasattr(self, 'in_proj'):
            nn.init.xavier_uniform_(self.in_proj.weight)
            if self.in_proj.bias is not None:
                nn.init.constant_(self.in_proj.bias, 0.)
        else:
            nn.init.xavier_uniform_(self.q_proj.weight)
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            if self.q_proj.bias is not None:
                nn.init.constant_(self.q_proj.bias, 0.)
                nn.init.constant_(self.k_proj.bias, 0.)
                nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def _qkv_projection(self, trgt_q: torch.Tensor, src_k: torch.Tensor, src_v: torch.Tensor):
        if hasattr(self, 'in_proj'):
            E = trgt_q.shape[-1]
            if torch.equal(trgt_q, src_k) and torch.equal(src_k, src_v):
                qkv = self.in_proj(trgt_q)
                q, k, v = torch.split(qkv, [E, E, E], dim=-1)
            elif torch.equal(src_k, src_v):
                w_q, w_kv = self.in_proj.weight.split([E, 2 * E])
                if self.in_proj.bias is None:
                    b_q, b_kv = None, None
                else:
                    b_q, b_kv = self.in_proj.bias.split([E, 2 * E])
                q = _linear_fwd(trgt_q,  w_q, b_q)
                kv = _linear_fwd(src_k,  w_kv, b_kv)
                k, v = torch.split(kv, [E, E], dim=-1)
            else:
                w_q, w_k, w_v = self.in_proj.weight.chunk(3, dim=1)
                if self.in_proj.bias is None:
                    b_q, b_k, b_v = None, None, None
                else:
                    b_q, b_k, b_v = self.in_proj.bias.chunk(3)
                q = _linear_fwd(trgt_q, w_q, b_q)
                k = _linear_fwd(src_k, w_k, b_k)
                v = _linear_fwd(src_v, w_v, b_v)
        else:
            q = self.q_proj(trgt_q)
            k = self.k_proj(src_k)
            v = self.v_proj(src_v)
        return q, k, v
    
    def _process_mask(self, mask: torch.Tensor, q: torch.Tensor, k: torch.Tensor):
        if mask is None:
            return None
        if not (mask.is_floating_point() or mask.dtype == torch.bool):
            raise ValueError("Attention mask must be of type torch.bool or floating point")
        if mask.is_floating_point():
            assert torch.all((mask == 0) | (mask == 1)), "Floating point Attention mask must be binary (0 or 1)"
            mask = mask.bool()
        assert mask.dtype == torch.bool, "Attention mask must be of type torch.bool"
        if mask.dim() == 2: # Key_padding_mask
            assert mask.shape[-1] == k.shape[1], "Mask must have same sequence length as keys"
            mask = mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, -1, -1)
        elif mask.dim() == 3: # Attn_mask
            assert mask.shape[-2] == q.shape[1], "Mask query dimension must match query sequence length"
            assert mask.shape[-1] == k.shape[1], "Mask key dimension must match key sequence length"
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        elif mask.dim() == 4: # Multihead_attn_mask
            assert mask.shape[1] == 1 or mask.shape[1] == self.num_heads, "Mask head dimension invalid"
            if mask.shape[1] == 1: mask = mask.expand(-1, self.num_heads, -1, -1)
            assert mask.shape[-2] == q.shape[1], "Mask query dimension must match query sequence length"
            assert mask.shape[-1] == k.shape[1], "Mask key dimension must match key sequence length"
        else:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")
        final_mask = torch.zeros_like(mask, dtype=q.dtype, device=q.device)
        final_mask = final_mask.masked_fill_(mask, float("-inf"))
        final_mask = rearrange(final_mask, "b h l s -> (b h) l s")
        return final_mask
    
    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                mask: Optional[torch.Tensor]=None,
                is_causal: bool=False) -> torch.Tensor:
        b = q.shape[0]
        q, k, v = self._qkv_projection(q, k, v)
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (hkv d) -> ... hkv d", d=self.head_dim)
        v = rearrange(v, "... (hkv d) -> ... hkv d", d=self.head_dim)
        if self.num_heads != self.num_heads_kv:
            k = k.repeat_interleave(self.num_heads // self.num_heads_kv, dim=2)
            v = v.repeat_interleave(self.num_heads // self.num_heads_kv, dim=2)
        mask = self._process_mask(mask, q, k)
        q = rearrange(q, "b l h d -> (b h) l d").contiguous()
        k = rearrange(k, "b s h d -> (b h) s d").contiguous()
        v = rearrange(v, "b s h d -> (b h) d s").contiguous()
        qk = bmm(q, k) * self.softmax_scale
        if is_causal:
            causal_mask = torch.tril(torch.zeros((qk.shape[0], qk.shape[1], qk.shape[2]), 
                                                  dtype=torch.bool, device=qk.device))
            if mask is not None:
                mask = mask + causal_mask
            else:
                mask = causal_mask
        if mask is not None:
            qk = qk + mask
        qk = self.softmax(qk)
        qk = self.dropout(qk)
        context = bmm(qk, v)
        context = rearrange(context, "(b h) l d -> b l (h d)", b=b)
        out = self.out_proj(context)
        return out