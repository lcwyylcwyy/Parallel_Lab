import os
os.environ['PYTHONPATH'] = '../..'
os.environ['TORCH_CUDA_ARCH_LIST'] = 'Ampere'
os.environ['MAX_JOBS'] = '10'
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.cpp_extension import load

def set_torch_random_seed(seed=0):
    import torch
    torch.manual_seed(seed)
    # set CUDA seeds
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

# Load the CUDA kernel as a python module
lib = load(name='rms_norm_lib_lc', 
            sources=[os.path.join(os.path.dirname(__file__),'rms_norm.cu')], 
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                "--use_fast_math"
            ], 
            verbose=True,
            extra_cflags=['-std=c++17'])


set_torch_random_seed()

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LC_RMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = 1e-6
    
    def forward(self, x):
        input_dtype = x.dtype
        var = x.pow(2).mean(-1, keepdim=True)
        hs = x * torch.rsqrt(var + self.eps)
        return self.gamma * hs


import numpy as np

class RMSNorm:
    def __init__(self, dim, eps=1e-8):
        self.eps = eps
        self.dim = dim
        self.weight = np.ones((dim,), dtype=np.float32)
        self.x = None
        self.rms = None

    def forward(self, x):
        # x: (batch, head_num, seq_len, head_dim)
        self.x = x
        self.rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        out = x / self.rms * self.weight
        return out

    def backward(self, dout):
        # dout: same shape as x
        x = self.x
        rms = self.rms
        weight = self.weight

        N = x.shape[-1]
        dx = (dout * weight) / rms - (x * np.sum(dout * weight * x, axis=-1, keepdims=True)) / (rms ** 3 * N)
        dweight = np.sum(dout * x / rms, axis=tuple(range(len(x.shape)-1)))
        return dx, dweight

if __name__ == "__main__":
    # 生成Attention Tensor
    batch, head_num, seq_len, head_dim = 1, 1, 64, 64
    np.random.seed(0)
    x_np = np.random.randn(batch, head_num, seq_len, head_dim).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)
    x_cuda = torch.tensor(x_np).reshape(batch * head_num * seq_len, head_dim).cuda().float().contiguous()
    out_cuda = torch.zeros_like(x_cuda).cuda().float().contiguous()

    # PyTorch RMSNorm
    class TorchRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-8):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight



    # Forward
    rmsnorm_np = RMSNorm(head_dim)
    out_np = rmsnorm_np.forward(x_np)

    rmsnorm_torch = TorchRMSNorm(head_dim)
    rmsnorm_torch_qw = Qwen2RMSNorm(head_dim)
    rmsnorm_torch_lc = LC_RMSNorm(head_dim)
    lib.rms_norm_f32x4(x_cuda, out_cuda, torch.tensor(rmsnorm_np.weight).cuda())

    with torch.no_grad():
        rmsnorm_torch.weight.copy_(torch.tensor(rmsnorm_np.weight))
        rmsnorm_torch_qw.weight.data.copy_(torch.tensor(rmsnorm_np.weight))
        rmsnorm_torch_lc.gamma.data.copy_(torch.tensor(rmsnorm_np.weight))
    out_torch = rmsnorm_torch(x_torch)
    out_qw = rmsnorm_torch_qw(x_torch)
    out_lc = rmsnorm_torch_lc(x_torch)

    print("Forward diff:", np.max(np.abs(out_np - out_torch.detach().numpy())))
    print("Forward QW diff:", np.max(np.abs(out_np - out_qw.detach().numpy())))
    print("Forward LC diff:", np.max(np.abs(out_np - out_lc.detach().numpy())))
    print("Forward diff between cuda and numpy version:", 
            np.allclose(out_np, out_cuda.reshape(batch, head_num, seq_len, head_dim).cpu().detach().numpy(), atol=1e-5))

    # Backward
    dout = np.random.randn(*x_np.shape).astype(np.float32)
    dx_np, dweight_np = rmsnorm_np.backward(dout)

    dout_torch = torch.tensor(dout)
    out_torch.backward(dout_torch)
    dx_torch = x_torch.grad
    dweight_torch = rmsnorm_torch.weight.grad

    print("Backward dx diff:", np.max(np.abs(dx_np - dx_torch.numpy())))
    print("Backward dweight diff:", np.max(np.abs(dweight_np - dweight_torch.numpy())))
