import torch
import torch.nn as nn
from TritonHub.Normalization import RMSNorm
from tabulate import tabulate as tb

class RMSNormUnitTest:
    def __init__(self, B=4, N=512, M=512, D=256, dtype=torch.float32, print_tb=False, eps=1e-5, bias=True, elementwise_affine=True):
        self.B = B
        self.N = N
        self.M = M
        self.D = D
        self.dtype = dtype
        self.print_tb = print_tb

        # Triton RMSNorm and Torch RMSNorm
        self.RMSNorm = RMSNorm(dimension=D, eps=eps, elementwise_affine=elementwise_affine, device='cuda', dtype=dtype)
        self.RMSNorm_torch = nn.RMSNorm(D, eps=eps, elementwise_affine=elementwise_affine, device='cuda', dtype=dtype)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def run(self):
        torch.manual_seed(42)
        # Create the input tensor. (This is an example of an "image" tensor with B H W C layout, however it can be any tensors since
        # interally tensors get flattened to 2D tensors (-1, C) before RMSNorm computation)
        input_data = torch.randn(self.B, self.M, self.N, self.D, device='cuda', dtype=self.dtype)

        input_tri = input_data.clone().detach().requires_grad_(True)
        input_tor = input_data.clone().detach().requires_grad_(True)

        # Set the tolerance for the comparison
        rtol, atol = (3e-4, 1e-3) if dtype == torch.float32 else ((1e-2, 5e-2) if dtype == torch.float else (5e-3, 1e-2))
        self.forward(input_tri, input_tor, atol, rtol)

    def forward(self, input_tri, input_tor, atol, rtol):
        self.start.record()
        output_tri = self.RMSNorm(input_tri)
        self.end.record()
        torch.cuda.synchronize()
        self.triton_time_fwd = self.start.elapsed_time(self.end)

        self.start.record()
        output_tor = self.RMSNorm_torch(input_tor)
        self.end.record()
        torch.cuda.synchronize()
        self.torch_time_fwd = self.start.elapsed_time(self.end)

        assert torch.allclose(output_tri, output_tor, atol=atol, rtol=rtol), 'Error in forward pass'
        if self.print_tb:
            self.diff_f = (output_tri - output_tor).abs()
        self.backward(input_tri, input_tor, output_tri, output_tor, atol, rtol)

    def backward(self, input_tri, input_tor, output_tri, output_tor, atol, rtol):
        g = torch.randn_like(output_tri, device="cuda") 

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
        print(tb([[self.dtype, self.D, 
                   self.diff_f.mean().item(), self.diff_f.max().item(), 
                   self.diff_b.mean().item(), self.diff_b.max().item(),
                   self.triton_time_fwd, self.torch_time_fwd,
                   self.triton_time_bwd, self.torch_time_bwd]],
                headers=['Dype', 'Dim', 
                         'Forward Mean Diff', 'Forward Max Diff', 
                         'Backward Mean Diff', 'Backward Max Diff',
                         'Triton Fwd Time', 'Torch Fwd Time', 
                         'Triton Bwd Time', 'Torch Bwd Time'], tablefmt='orgtbl'))

if __name__ == '__main__':
    B, N, M = 1, 256, 256
    eps = 1e-5
    bias = True
    print_tb = True
    elementwise_affine = True
    for D in [32, 64, 128, 256, 512, 1024, 2048]:
        for i in range(2):
            if i ==0: print('First iteration Slow due to Triton Autotune'); print_tb=False 
            else: print_tb=True
            for dtype in [torch.float16, torch.float32]:
                runner = RMSNormUnitTest(B, N, M, D, dtype, print_tb, eps, bias, elementwise_affine)
                runner.run()
                del runner
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    print('All tests passed!')