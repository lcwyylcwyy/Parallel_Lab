import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.cpp_extension import load
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = 'Ampere'
os.environ['MAX_JOBS'] = '10'
# Load the CUDA kernel as a python module
lib = load(name='layer_norm_lib_lc', 
            sources=[os.path.join(os.path.dirname(__file__),'layer_norm.cu')], 
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

class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        # normalized_shape: int or tuple, e.g. head_dim
        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self.beta = np.zeros(normalized_shape, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        # x: (batch, head_num, seq_len, head_dim)
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma * self.x_hat + self.beta
        return out

    def backward(self, dout):
        # dout: same shape as x
        N = self.x.shape[-1]
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) + \
                dvar * np.mean(-2 * (self.x - self.mean), axis=-1, keepdims=True)
        dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2 * (self.x - self.mean) / N + dmean / N
        dgamma = np.sum(dout * self.x_hat, axis=(0, 1, 2))
        dbeta = np.sum(dout, axis=(0, 1, 2))
        return dx, dgamma, dbeta


def validate_numpy_layer_norm():
    # 随机输入
    np.random.seed(0)
    # batch, head_num, seq_len, head_dim = 2, 3, 4, 5
    batch_size = 2
    head_num = 12
    seq_len = 192
    head_dim = 64
    # x = torch.randn(batch_size, head_num, seq_len, head_dim)
    x_np = np.random.randn(batch_size, head_num, seq_len, head_dim).astype(np.float32)
    dout_np = np.random.randn(batch_size, head_num, seq_len, head_dim).astype(np.float32)
    
    # lib.layer_norm_f32(x_np)
    # Numpy实现
    ln = LayerNorm(head_dim)
    out_np = ln.forward(x_np)
    dx_np, dgamma_np, dbeta_np = ln.backward(dout_np)


    # # CUDA实现
    x_cuda = torch.tensor(x_np).reshape(batch_size * head_num * seq_len, head_dim).cuda().float().contiguous()
    out_cuda = torch.zeros_like(x_cuda).cuda().float().contiguous()
    # lib.layer_norm_f32(x_cuda, out_cuda, torch.tensor(ln.gamma).cuda(), torch.tensor(ln.beta).cuda())
    lib.layer_norm_f32x4(x_cuda, out_cuda, torch.tensor(ln.gamma).cuda(), torch.tensor(ln.beta).cuda())
    print("Forward diff between cuda and numpy version:", 
            np.allclose(out_np, out_cuda.reshape(batch_size, head_num, seq_len, head_dim).cpu().detach().numpy(), atol=1e-5))


    # PyTorch实现
    x = torch.tensor(x_np, requires_grad=True)
    layer_norm = nn.LayerNorm(head_dim, eps=1e-5, elementwise_affine=True)
    with torch.no_grad():
        layer_norm.weight.copy_(torch.tensor(ln.gamma))
        layer_norm.bias.copy_(torch.tensor(ln.beta))
    out_torch = layer_norm(x)
    dout = torch.tensor(dout_np)
    out_torch.backward(dout)

    # 对比
    print("Forward diff between torch and numpy:", np.allclose(out_np, out_torch.detach().numpy(), atol=1e-5))
    print("dx diff:", np.allclose(dx_np, x.grad.numpy(), atol=1e-5))
    print("dgamma diff:", np.allclose(dgamma_np, layer_norm.weight.grad.numpy(), atol=1e-3))
    print("dbeta diff:", np.allclose(dbeta_np, layer_norm.bias.grad.numpy(), atol=1e-3))


if __name__ == "__main__":
    validate_numpy_layer_norm()