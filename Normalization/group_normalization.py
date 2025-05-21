import numpy as np
import torch

class GroupNormNP:
    def __init__(self, num_groups, num_channels, eps=1e-5):
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = np.ones((1, num_channels, 1, 1), dtype=np.float32)
        self.beta = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
        # cache for backward
        self.cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.num_groups
        x = x.reshape(N, G, C // G, H, W)
        mean = x.mean(axis=(2, 3, 4), keepdims=True)
        var = x.var(axis=(2, 3, 4), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        x_norm = x_norm.reshape(N, C, H, W)
        out = self.gamma * x_norm + self.beta
        self.cache = (x, x_norm, mean, var, self.gamma)
        return out

    def backward(self, dout):
        x, x_norm, mean, var, gamma = self.cache
        N, G, Cg, H, W = x.shape
        m = Cg * H * W

        dx_norm = dout * gamma
        dx_norm = dx_norm.reshape(N, G, Cg, H, W)
        x_norm = x_norm.reshape(N, G, Cg, H, W)  # 修正：reshape x_norm

        std_inv = 1. / np.sqrt(var + self.eps)
        dx = (1. / m) * std_inv * (
            m * dx_norm
            - dx_norm.sum(axis=(2, 3, 4), keepdims=True)
            - x_norm * (dx_norm * x_norm).sum(axis=(2, 3, 4), keepdims=True)
        )
        dx = dx.reshape(N, G * Cg, H, W)

        dgamma = (dout * x_norm.reshape(N, G * Cg, H, W)).sum(axis=(0, 2, 3), keepdims=True)
        dbeta = dout.sum(axis=(0, 2, 3), keepdims=True)
        return dx, dgamma, dbeta
# 验证代码
def validate_groupnorm():
    np.random.seed(0)
    torch.manual_seed(0)
    N, C, H, W = 2, 6, 4, 4
    G = 3
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)

    # PyTorch GroupNorm
    gn = torch.nn.GroupNorm(G, C, affine=True)
    gn.weight.data.fill_(1.0)
    gn.bias.data.zero_()
    y_torch = gn(x_torch)
    loss = y_torch.sum()
    loss.backward()
    grad_torch = x_torch.grad.detach().numpy()

    # Numpy GroupNorm
    gn_np = GroupNormNP(G, C)
    y_np = gn_np.forward(x_np)
    dout = np.ones_like(y_np)
    dx_np, dgamma, dbeta = gn_np.backward(dout)

    print("Forward diff:", np.abs(y_np - y_torch.detach().numpy()).max())
    print("Backward diff:", np.abs(dx_np - grad_torch).max())

if __name__ == "__main__":
    validate_groupnorm()