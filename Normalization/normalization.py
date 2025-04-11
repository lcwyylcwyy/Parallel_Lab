import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from utils import set_torch_random_seed

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


class LayerNormTorch(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Layer Normalization implementation

        Args:
            normalized_shape: input shape from an expected input
                of size [*, normalized_shape[0], normalized_shape[1], ...]
            eps: a small constant for numerical stability
            elementwise_affine: whether to use learnable affine parameters
        """
        super(LayerNormTorch, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)


    def forward(self, x):
        """
        Forward pass for layer normalization

        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]

        Returns:
            Normalized tensor with same shape as input
        """
        # Calculate mean and variance along the last dimensions (feature dimensions)
        mean = x.mean(dim=-1, keepdim=True)
        # Calculate variance and add epsilon for numerical stability
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

        # Normalize input
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply scale and shift if using affine transform
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias

        return x_norm


class LayerNormalizationNumpy:
    def __init__(self, features_dim, eps=1e-5):
        """
        Pure NumPy implementation of Layer Normalization

        Args:
            features_dim: number of features in input
            eps: small constant for numerical stability
        """
        self.eps = eps
        self.gamma = np.ones(features_dim)  # Scale parameter
        self.beta = np.zeros(features_dim)  # Shift parameter

        # For storing values needed in backpropagation
        self.x_norm = None
        self.var = None
        self.mean = None
        self.x = None

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input of shape [batch_size, features_dim]

        Returns:
            Layer-normalized output with same shape as input
        """
        self.x = x
        # Compute mean and variance along the features dimension
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)

        # Scale and shift
        out = self.gamma * self.x_norm + self.beta

        return out

    def backward(self, dout):
        """
        Backward pass

        Args:
            dout: Gradient of loss with respect to layer output

        Returns:
            Gradient with respect to layer input
        """
        batch_size = dout.shape[0]
        features_dim = dout.shape[1]

        # Gradients for gamma and beta
        self.dgamma = np.sum(dout * self.x_norm, axis=0)
        self.dbeta = np.sum(dout, axis=0)

        # Gradient with respect to x_norm
        dx_norm = dout * self.gamma

        # Gradient with respect to x
        # This is the complex part of layer normalization backprop
        dvar = -0.5 * np.sum(dx_norm * (self.x - self.mean) * np.power(self.var + self.eps, -1.5), axis=-1,
                             keepdims=True)
        dmean = -np.sum(dx_norm / np.sqrt(self.var + self.eps), axis=-1, keepdims=True) + \
                dvar * -2 * np.mean(self.x - self.mean, axis=-1, keepdims=True)

        dx = dx_norm / np.sqrt(self.var + self.eps) + \
             dvar * 2 * (self.x - self.mean) / features_dim + \
             dmean / features_dim

        return dx


def validate_numpy_layer_norm():
    # Set parameter
    batch_size = 10
    feature_dim = 256

    x_np = np.random.randn(batch_size, feature_dim)
    x_torch = torch.tensor(x_np, dtype=torch.float32)

    # 实例化 PyTorch 内置的 LayerNorm
    torch_ln = nn.LayerNorm(feature_dim)

    # 将 PyTorch 权重转换为 NumPy
    gamma = torch_ln.weight.detach().numpy()
    beta = torch_ln.bias.detach().numpy()

    # 实例化 NumPy 实现的 LayerNorm
    numpy_ln = LayerNormalizationNumpy(feature_dim)
    numpy_ln.gamma = gamma
    numpy_ln.beta = beta

    # 计算输出
    torch_output = torch_ln(x_torch).detach().numpy()
    numpy_output = numpy_ln.forward(x_np)

    # 计算差异
    diff = np.abs(torch_output - numpy_output).max()

    print(f"Maximum absolute difference (NumPy vs PyTorch): {diff:.8f}")

    return diff < 1e-6


def validate_layer_norm():
    batch_size = 6
    head_num = 12
    seq_len = 192
    head_dim = 64
    x = torch.randn(batch_size, head_num, seq_len, head_dim)

    # instantiate the PyTorch LayerNorm (norm along with the last dim)
    torch_ln = nn.LayerNorm(head_dim)

    # instantiate the LayerNorm implemented above
    custom_ln = LayerNormTorch(head_dim)

    # make sure the weight and bias are same between the torch and custom implementation
    custom_ln.weight = torch.nn.Parameter(torch_ln.weight.clone())
    custom_ln.bias = torch.nn.Parameter(torch_ln.bias.clone())

    torch_output = torch_ln(x)
    custom_output = custom_ln(x)

    # get difference
    diff = torch.abs(torch_output - custom_output).max().item()

    print(f"Maximum absolute difference: {diff:.8f}")
    print(f"PyTorch LayerNorm Output Shape: {torch_output.shape}")
    print(f"The two implementations are {'same' if torch.allclose(torch_output, custom_output,  rtol=0, atol=1e-05) else 'different'}")

    return diff < 1e-6


if __name__ == "__main__":
    if validate_layer_norm():
        print("Custom PyTorch LayerNorm implementation validated successfully!")
    else:
        print("Custom PyTorch LayerNorm implementation has discrepancies.")
