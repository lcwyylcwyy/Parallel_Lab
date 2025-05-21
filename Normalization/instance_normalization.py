import numpy as np
import torch
import torch.nn as nn


class InstanceNorm:
    def __init__(self, num_channels, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, num_channels, 1, 1))
        self.beta = np.zeros((1, num_channels, 1, 1))
        self.cache = None

    def forward(self, x):
        # x: (N, C, H, W)
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        self.cache = (x, x_hat, mean, var)
        return out

    def backward(self, dout):
        # dout: (N, C, H, W)
        x, x_hat, mean, var = self.cache
        N, C, H, W = x.shape
        HW = H * W

        # Gradients for gamma and beta
        dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        # Gradient for x
        dx_hat = dout * self.gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=(2, 3), keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=(2, 3), keepdims=True) + \
                dvar * np.sum(-2 * (x - mean), axis=(2, 3), keepdims=True) / HW
        dx = dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / HW + dmean / HW

        return dx, dgamma, dbeta



# 随机生成输入
N, C, H, W = 2, 3, 4, 4
np.random.seed(42)
x_np = np.random.randn(N, C, H, W).astype(np.float32)
x_torch = torch.tensor(x_np, requires_grad=True)

# PyTorch InstanceNorm
inst_norm = nn.InstanceNorm2d(C, affine=True, eps=1e-5)
with torch.no_grad():
    inst_norm.weight.copy_(torch.ones_like(inst_norm.weight))
    inst_norm.bias.copy_(torch.zeros_like(inst_norm.bias))

# Forward
y_torch = inst_norm(x_torch)
y_np = InstanceNorm(C).forward(x_np)

print("Forward difference:", np.abs(y_torch.detach().numpy() - y_np).max())

# Backward
dy = np.random.randn(*y_np.shape).astype(np.float32)
y_torch.backward(torch.tensor(dy))
dx_np, dgamma_np, dbeta_np = InstanceNorm(C).forward(x_np), None, None
# 这里只演示forward和数值对比，backward可以用数值梯度验证