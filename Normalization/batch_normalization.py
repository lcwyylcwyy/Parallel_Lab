import torch
import torch.nn as nn

class ManualBatchNorm2d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(num_features, requires_grad=True)
        self.beta = torch.zeros(num_features, requires_grad=True)
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x, training=True):
        N, C, H, W = x.shape
        if training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)
        else:
            mean = self.running_mean.view(1, C, 1, 1)
            var = self.running_var.view(1, C, 1, 1)
        self.x_centered = x - mean
        self.std_inv = torch.rsqrt(var + self.eps)
        self.x_norm = self.x_centered * self.std_inv
        out = self.gamma.view(1, C, 1, 1) * self.x_norm + self.beta.view(1, C, 1, 1)
        return out

    def backward(self, dout):
        N, C, H, W = dout.shape
        x_norm = self.x_norm
        std_inv = self.std_inv
        x_centered = self.x_centered

        dbeta = dout.sum(dim=(0, 2, 3))
        dgamma = (dout * x_norm).sum(dim=(0, 2, 3))

        dx_norm = dout * self.gamma.view(1, C, 1, 1)
        dvar = (-0.5 * (dx_norm * x_centered).sum(dim=(0, 2, 3), keepdim=True) *
                (std_inv ** 3))
        dmean = (-dx_norm * std_inv).sum(dim=(0, 2, 3), keepdim=True) + \
                dvar * (-2.0 * x_centered).mean(dim=(0, 2, 3), keepdim=True)
        dx = dx_norm * std_inv + dvar * 2 * x_centered / (N * H * W) + dmean / (N * H * W)
        return dx, dgamma, dbeta

# 验证
torch.manual_seed(0)

channel = 3
size = 2
batch_size = 2
x = torch.randn(batch_size, channel, size, size, requires_grad=True)
manual_bn = ManualBatchNorm2d(channel)
y_manual = manual_bn.forward(x)
loss_manual = y_manual.sum()
dx_manual, dgamma_manual, dbeta_manual = manual_bn.backward(torch.ones_like(y_manual))

# PyTorch内置
bn = nn.BatchNorm2d(channel, affine=True, track_running_stats=False)
with torch.no_grad():
    bn.weight.copy_(manual_bn.gamma)
    bn.bias.copy_(manual_bn.beta)
y_torch = bn(x)
loss_torch = y_torch.sum()
loss_torch.backward()

print("Manual output close to PyTorch:", torch.allclose(y_manual, y_torch, atol=1e-5))
print("Manual dx close to PyTorch:", torch.allclose(dx_manual, x.grad, atol=1e-5))
print("Manual dgamma close to PyTorch:", torch.allclose(dgamma_manual, bn.weight.grad, atol=1e-4))
print("Manual dbeta close to PyTorch:", torch.allclose(dbeta_manual, bn.bias.grad, atol=1e-5))