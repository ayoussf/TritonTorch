import torch
import torch.nn as nn
from TritonHub.Layers.conv1d import Conv1d
from tabulate import tabulate as tb

class Conv1dUnitTest:
    def __init__(self, B=4, C_in=32, C_out=64, L=256, K=3, stride=1, padding=1, dilation=1,
                 dtype=torch.float32, print_tb=False, bias=True):
        self.B = B
        self.C_in = C_in
        self.C_out = C_out
        self.L = L
        self.K = K
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dtype = dtype
        self.print_tb = print_tb
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

        # Triton Conv1d and Torch Conv1d
        self.Conv1d = Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=K,
                             stride=stride, padding=padding, dilation=dilation,
                             bias=bias, device="cuda", dtype=dtype)
        self.Conv1d_torch = nn.Conv1d(in_channels=C_in, out_channels=C_out, kernel_size=K,
                                      stride=stride, padding=padding, dilation=dilation,
                                      bias=bias, device="cuda", dtype=dtype)
        with torch.no_grad():
            self.Conv1d_torch.weight.copy_(self.Conv1d.weight)
            if bias:
                self.Conv1d_torch.bias.copy_(self.Conv1d.bias)
        assert torch.allclose(self.Conv1d.weight, self.Conv1d_torch.weight)

    def run(self):
        torch.manual_seed(42)
        input_data = torch.randn(self.B, self.C_in, self.L, device='cuda', dtype=self.dtype)

        input_tri = input_data.clone().detach().requires_grad_(True)
        input_tor = input_data.clone().detach().requires_grad_(True)

        rtol, atol = (0.0, 1e-2)
        self.forward(input_tri, input_tor, atol, rtol)

    def forward(self, input_tri, input_tor, atol, rtol):
        self.start.record()
        output_tri = self.Conv1d(input_tri)
        self.end.record()
        torch.cuda.synchronize()
        self.triton_time_fwd = self.start.elapsed_time(self.end)

        self.start.record()
        output_tor = self.Conv1d_torch(input_tor)
        self.end.record()
        torch.cuda.synchronize()
        self.torch_time_fwd = self.start.elapsed_time(self.end)

        assert torch.allclose(output_tri, output_tor, atol=atol, rtol=rtol), 'Error in forward pass'
        if self.print_tb:
            self.diff_f = (output_tri - output_tor).abs()
        self.backward(input_tri, input_tor, output_tri, output_tor, atol, rtol)

    def backward(self, input_tri, input_tor, output_tri, output_tor, atol, rtol):
        g = torch.randn_like(output_tri, device='cuda')

        self.start.record()
        output_tri.backward(g)
        self.end.record()
        torch.cuda.synchronize()
        self.triton_time_bwd = self.start.elapsed_time(self.end)

        self.start.record()
        output_tor.backward(g)
        self.end.record()
        torch.cuda.synchronize()
        self.torch_time_bwd = self.start.elapsed_time(self.end)
        assert torch.allclose(input_tri.grad, input_tor.grad, atol=atol, rtol=rtol), 'Error in backward pass'
        if self.print_tb:
            self.diff_b = (input_tri.grad - input_tor.grad).abs()
            self.table()
    
    def table(self):
        print(tb([[self.dtype, self.C_in, self.C_out, self.K,
                   self.padding, self.dilation, self.stride,
                   self.diff_f.mean().item(), self.diff_f.max().item(),
                   self.diff_b.mean().item(), self.diff_b.max().item(),
                   self.triton_time_fwd, self.torch_time_fwd,
                   self.triton_time_bwd, self.torch_time_bwd]],
                 headers=['Dtype', 'C_in', 'C_out', 'K', 'Pad',
                          'Dil.', 'Str.',
                          'Forward Mean Diff', 'Forward Max Diff',
                          'Backward Mean Diff', 'Backward Max Diff',
                          'Triton Fwd Time', 'Torch Fwd Time',
                          'Triton Bwd Time', 'Torch Bwd Time'], tablefmt='orgtbl'))

if __name__ == '__main__':
    B, L = 1, 1000
    C_in = 32
    C_out = 64
    kernel_size = 3
    configs = [(int(C_in * i), int(C_out * i), kernel_size) for i in [1, 2, 4, 8, 16]]
    for bias in [True, False]:
        for padding in [0, 5, 10]:
            for dilation in [1, 2]:
                for stride in [1, 2]:
                    for C_in, C_out, K in configs:
                        for i in range(2):
                            if i == 0:
                                print('First iteration slow due to Triton autotune'); print_tb = False
                            else:
                                print_tb = True
                            for dtype in [torch.float16, torch.float32]:
                                runner = Conv1dUnitTest(B, C_in, C_out, L, K, dtype=dtype, stride=stride, padding=padding, dilation=dilation, print_tb=print_tb, bias=bias)
                                runner.run()
                                del runner
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
    print('All tests passed!')