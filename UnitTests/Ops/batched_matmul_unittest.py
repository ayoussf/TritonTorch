import torch
from TritonHub.Ops import bmm
from tabulate import tabulate as tb

class BMMUnitTest:
    def __init__(self, B=4, N=512, M=512, D=256, dtype=torch.float32, print_tb=False):
        self.B = B
        self.N = N
        self.M = M
        self.D = D
        self.dtype = dtype
        self.print_tb = print_tb
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def run(self):
        torch.manual_seed(42)
        input_x = torch.randn(self.B, self.M, self.D, device='cuda', dtype=self.dtype)
        input_y = torch.randn(self.B, self.N, self.D, device='cuda', dtype=self.dtype)

        # Create separate tensors for input and input_ref using the same data and ensure gradient computation
        input_x_tri = input_x.clone().detach().requires_grad_(True)
        input_x_tor = input_x.clone().detach().requires_grad_(True)
        input_y_tri = input_y.clone().detach().requires_grad_(True)
        input_y_tor = input_y.clone().detach().requires_grad_(True)

        # Set the tolerance for the comparison
        rtol, atol = (0.0, 1e-1) if dtype == torch.float32 else (1e-2, 8e-3)
        self.forward(input_x_tri, input_y_tri, input_x_tor, input_y_tor, atol, rtol)

    def forward(self, input_x_tri, input_y_tri, input_x_tor, input_y_tor, atol, rtol):
        self.start.record()
        output_tri = bmm(input_x_tri, input_y_tri)
        self.end.record()
        torch.cuda.synchronize()
        self.triton_time_fwd = self.start.elapsed_time(self.end)

        self.start.record()
        output_tor = torch.einsum('bmd,bnd->bmn', input_x_tor, input_y_tor)
        self.end.record()
        torch.cuda.synchronize()
        self.torch_time_fwd = self.start.elapsed_time(self.end)
        
        assert torch.allclose(output_tri, output_tor, atol=atol, rtol=rtol), 'Error in forward pass'
        if self.print_tb:
            self.diff_f = (output_tri - output_tor).abs()
        self.backward(input_x_tri, input_y_tri, input_x_tor, input_y_tor, output_tri, output_tor, atol, rtol)

    def backward(self, input_x_tri, input_y_tri, input_x_tor, input_y_tor, output_tri, output_tor, atol, rtol):
        g = torch.rand_like(output_tri, device="cuda")

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

        assert torch.allclose(input_x_tri.grad, input_x_tor.grad, atol=atol, rtol=rtol), 'Error in backward pass'
        assert torch.allclose(input_y_tri.grad, input_y_tor.grad, atol=atol, rtol=rtol), 'Error in backward pass'
        if self.print_tb:
            self.diff_xb = (input_x_tri.grad - input_x_tor.grad).abs()
            self.diff_yb = (input_y_tri.grad - input_y_tor.grad).abs()
            self.table()
    
    def table(self):
        print(tb([[self.dtype, self.D, 
                   self.diff_f.mean().item(), self.diff_f.max().item(), 
                   self.diff_xb.mean().item(), self.diff_xb.max().item(),
                   self.diff_yb.mean().item(), self.diff_yb.max().item(),
                   self.triton_time_fwd, self.torch_time_fwd,
                   self.triton_time_bwd, self.torch_time_bwd]],
                headers=['Dype', 'Dim', 
                         'Forward Mean Diff', 'Forward Max Diff', 
                         'Backward X Mean Diff', 'Backward X Max Diff',
                         'Backward Y Mean Diff', 'Backward Y Max Diff',
                         'Triton Fwd Time', 'Torch Fwd Time', 
                         'Triton Bwd Time', 'Torch Bwd Time'], tablefmt='orgtbl'))

if __name__ == '__main__':
    B, N, M = 1, 10, 10
    print_tb = True
    for D in [32, 64, 128, 256, 512]:
        for i in range(2):
            if i == 0: print('First iteration Slow due to Triton Autotune'); print_tb=False 
            else: print_tb=True
            for dtype in [torch.float16, torch.float32]:
                runner = BMMUnitTest(B, N, M, D, dtype, print_tb)
                runner.run()
                del runner
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    print('All tests passed!')