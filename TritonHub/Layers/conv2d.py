import torch
import triton
import triton.language as tl
import math
from einops import rearrange
from TritonHub.utils import custom_fwd, custom_bwd
from TritonHub.autotune import get_cuda_autotune_config

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K'], include_group_size=True, include_fp8_configs=True, include_extra_configs=True),
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv2d_kernel_fwd(X,
                       W,
                       Y,
                       M,
                       N,
                       K,
                       stride_am, stride_ak,
                       stride_bk, stride_bn,
                       stride_cm, stride_cn,
                       B,
                       BLOCK_SIZE_M: tl.constexpr,
                       BLOCK_SIZE_N: tl.constexpr,
                       BLOCK_SIZE_K: tl.constexpr,
                       GROUP_SIZE_M: tl.constexpr,
                       HAS_BIAS: tl.constexpr):
    
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    X = X + (offs_am[:, None] * stride_am  + offs_k[None, :] * stride_ak)
    W = W + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    y = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(X, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(W, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        y += tl.dot(x, w)
        X += BLOCK_SIZE_K * stride_ak
        W += BLOCK_SIZE_K * stride_bk
    if HAS_BIAS:
        y += tl.load(B + offs_bn, mask=offs_bn < N)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    Y = Y + stride_cm  * offs_cm[:, None] + stride_cn  * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(Y, y, mask=c_mask)

def _conv2d_fwd(x, weight, bias, stride, padding, dilation):
    if x.stride(-1) != 1:
        x = x.contiguous()
    assert x.stride(-1) == 1, 'expect input to be row-major'
    weight = weight.contiguous()
    assert weight.stride(-1) == 1, 'expect weight to be row-major'
    batch, in_c, in_h, in_w = x.shape
    out_c, _, k_h, k_w = weight.shape

    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    padding_h, padding_w = (padding, padding) if isinstance(padding, int) else padding
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    out_h = (in_h + 2 * padding_h - dilation_h * (k_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * padding_w - dilation_w * (k_w - 1) - 1) // stride_w + 1

    M, N, K = batch * out_h * out_w, out_c, in_c * k_h * k_w

    x = torch.nn.functional.unfold(x,
                                   kernel_size=(k_h, k_w),
                                   dilation=(dilation_h, dilation_w),
                                   padding=(padding_h, padding_w),
                                   stride=(stride_h, stride_w))
    x = rearrange(x, 'b (c kh kw) (oh ow) -> (b oh ow) (c kh kw)',
                  c=in_c, kh=k_h, kw=k_w, oh=out_h, ow=out_w).contiguous()

    weight = rearrange(weight, 'oc ic kh kw -> (ic kh kw) oc').contiguous()

    out = torch.empty((M, N), memory_format=torch.contiguous_format, dtype=x.dtype, device=x.device)
    assert out.stride(-1) == 1, 'expect output to be row-major'
    HAS_BIAS = True if bias is not None else False
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    with torch.cuda.device(x.device.index):
        _conv2d_kernel_fwd[grid](x,
                                 weight,
                                 out,
                                 M,
                                 N,
                                 K,
                                 x.stride(0), x.stride(1),
                                 weight.stride(0), weight.stride(1),
                                 out.stride(0), out.stride(1),
                                 bias,
                                 HAS_BIAS=HAS_BIAS)
    return rearrange(out, '(b oh ow) c -> b c oh ow', b=batch, oh=out_h, ow=out_w).contiguous()

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K'], include_group_size=True, include_fp8_configs=True, include_extra_configs=True),
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv2d_kernel_bwd_dx(DOUT, W, DX,
                          M, N, K,
                          stride_DOUT_m, stride_DOUT_n,
                          stride_W_k, stride_W_n,
                          stride_DX_m, stride_DX_k,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr,
                          GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = offs_bk
    
    DOUT_ptr = DOUT + (offs_am[:, None] * stride_DOUT_m + offs_n[None, :] * stride_DOUT_n)
    W_ptr = W + (offs_k[None, :] * stride_W_k + offs_n[:, None] * stride_W_n)
    
    DX_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        g = tl.load(DOUT_ptr, mask=offs_n[None, :] < N - n * BLOCK_SIZE_N, other=0.0)
        w = tl.load(W_ptr, mask=offs_n[:, None] < N - n * BLOCK_SIZE_N, other=0.0)
        DX_acc += tl.dot(g, w)
        DOUT_ptr += BLOCK_SIZE_N * stride_DOUT_n
        W_ptr += BLOCK_SIZE_N * stride_W_n

    offs_m = offs_am[:, None]
    offs_k = offs_bk[None, :]
    DX_ptr = DX + (offs_m * stride_DX_m + offs_k * stride_DX_k)
    mask = (offs_m < M) & (offs_k < K)
    tl.store(DX_ptr, DX_acc, mask=mask)

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N', 'K'], include_group_size=True, include_fp8_configs=True, include_extra_configs=True),
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv2d_kernel_bwd_dw(X, DOUT, DW,
                          M, N, K,
                          stride_X_m, stride_X_k,
                          stride_DOUT_m, stride_DOUT_n,
                          stride_DW_k, stride_DW_n,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr,
                          GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_k = group_id * GROUP_SIZE_M
    group_size_k = min(num_pid_k - first_pid_k, GROUP_SIZE_M)
    pid_k = first_pid_k + ((pid % num_pid_in_group) % group_size_k)
    pid_n = (pid % num_pid_in_group) // group_size_k

    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    
    X_ptr = X + (offs_m[None, :] * stride_X_m + offs_k[:, None] * stride_X_k)
    DOUT_ptr = DOUT + (offs_m[:, None] * stride_DOUT_m + offs_n[None, :] * stride_DOUT_n)
    
    DW_acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        a = tl.load(X_ptr, mask=offs_m[None, :] < M - m * BLOCK_SIZE_M, other=0.0)
        b = tl.load(DOUT_ptr, mask=offs_m[:, None] < M - m * BLOCK_SIZE_M, other=0.0)
        DW_acc += tl.dot(a, b)
        X_ptr += BLOCK_SIZE_M * stride_X_m
        DOUT_ptr += BLOCK_SIZE_M * stride_DOUT_m
    
    offs_k_out = offs_k[:, None]
    offs_n_out = offs_n[None, :]
    DW_ptr = DW + (offs_k_out * stride_DW_k + offs_n_out * stride_DW_n)
    mask = (offs_k_out < K) & (offs_n_out < N)
    tl.store(DW_ptr, DW_acc, mask=mask)

@triton.autotune(
    configs=get_cuda_autotune_config(block_keys=['M', 'N'], include_group_size=False, include_fp8_configs=True),
    key=['M', 'N'],
)
@triton.jit
def _conv2d_kernel_bwd_db(DOUT, DB,
                          M, N,
                          stride_DOUT_m, stride_DOUT_n,
                          BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    
    DB_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        offs_am = m * BLOCK_SIZE_M + offs_m
        DOUT_ptr = DOUT + (offs_am[:, None] * stride_DOUT_m + offs_n[None, :] * stride_DOUT_n)
        g = tl.load(DOUT_ptr, mask=(offs_am[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        DB_acc += tl.sum(g, axis=0)
    tl.store(DB + offs_n, DB_acc, mask=offs_n < N)

def _conv2d_bwd(x, dout, weight, bias, stride, padding, dilation):
    if x.stride(-1) != 1:
        x = x.contiguous()
    assert x.stride(-1) == 1, 'expect input to be row-major'
    weight = weight.contiguous()
    assert weight.stride(-1) == 1, 'expect weight to be row-major'
    batch, in_c, in_h, in_w = x.shape
    _, out_c, out_h, out_w = dout.shape
    _, _, k_h, k_w = weight.shape

    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    padding_h, padding_w = (padding, padding) if isinstance(padding, int) else padding
    dilation_h, dilation_w = (dilation, dilation) if isinstance(dilation, int) else dilation

    M, N, K = batch * out_h * out_w, out_c, in_c * k_h * k_w

    x = torch.nn.functional.unfold(x,
                                   kernel_size=(k_h, k_w),
                                   dilation=(dilation_h, dilation_w),
                                   padding=(padding_h, padding_w),
                                   stride=(stride_h, stride_w))
    x = rearrange(x, 'b (c kh kw) (oh ow) -> (b oh ow) (c kh kw)',
                  c=in_c, kh=k_h, kw=k_w, oh=out_h, ow=out_w).contiguous()
    dout = rearrange(dout, 'b c oh ow -> (b oh ow) c').contiguous()
    weight = rearrange(weight, 'oc ic kh kw -> (ic kh kw) oc').contiguous()
    
    dx = torch.empty_like(x, memory_format=torch.contiguous_format, dtype=x.dtype, device=x.device)
    dw = torch.empty_like(weight, memory_format=torch.contiguous_format, dtype=x.dtype, device=x.device)

    grid_dx = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(K, META['BLOCK_SIZE_K']),)
    grid_dw = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)

    with torch.cuda.device(x.device.index):
        _conv2d_kernel_bwd_dx[grid_dx](dout, weight, dx,
                                       M, N, K,
                                       dout.stride(0), dout.stride(1),
                                       weight.stride(0), weight.stride(1),
                                       dx.stride(0), dx.stride(1))
        _conv2d_kernel_bwd_dw[grid_dw](x, dout, dw,
                                       M, N, K,
                                       x.stride(0), x.stride(1),
                                       dout.stride(0), dout.stride(1),
                                       dw.stride(0), dw.stride(1))
        if bias is not None:
            db = torch.empty_like(bias, memory_format=torch.contiguous_format, dtype=bias.dtype, device=bias.device)
            grid_db = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']),)
            _conv2d_kernel_bwd_db[grid_db](dout, db,
                                           M, N,
                                           dout.stride(0), dout.stride(1),)
        else:
            db = None
    
    dx = rearrange(dx, '(b oh ow) (c kh kw) -> b (c kh kw) (oh ow)',
                   b=batch, oh=out_h, ow=out_w, c=in_c, kh=k_h, kw=k_w).contiguous()
    dx = torch.nn.functional.fold(dx,
                                  output_size=(in_h, in_w),
                                  kernel_size=(k_h, k_w),
                                  dilation=(dilation_h, dilation_w),
                                  padding=(padding_h, padding_w),
                                  stride=(stride_h, stride_w))
    dw = rearrange(dw, '(ic kh kw) oc -> oc ic kh kw', ic=in_c, kh=k_h, kw=k_w).contiguous()
    return dx, dw, db

class conv2d(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        output = _conv2d_fwd(x, weight, bias, stride, padding, dilation)
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        x, weight, bias = ctx.saved_tensors
        dx, dw, db = _conv2d_bwd(x, dout, weight, bias, ctx.stride, ctx.padding, ctx.dilation)
        return dx, dw, db, None, None, None

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels,
                                                     self.kernel_size[0], self.kernel_size[1],
                                                     **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.conv2d_fn = conv2d.apply

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return self.conv2d_fn(x, self.weight, self.bias, self.stride, self.padding, self.dilation)