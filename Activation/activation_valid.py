import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import erf  # Import erf from SciPy

# Custom implementations
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))


def gelu(x):
    # Approximation of GELU
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_derivative(x):
    # Derivative of the GELU approximation
    term = np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)
    tanh_term = np.tanh(term)
    sech_squared = 1 - tanh_term ** 2
    inner_derivative = np.sqrt(2 / np.pi) * (1 + 0.134145 * x ** 2)
    return 0.5 * (1 + tanh_term + x * sech_squared * inner_derivative)


def swish(x):
    return x * sigmoid(x)


def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


def tanh_custom(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def glu(x, split_dim=-1):
    # GLU splits input in half
    if isinstance(x, np.ndarray):
        split_size = x.shape[split_dim] // 2
        a, b = np.split(x, 2, axis=split_dim)
        return a * sigmoid(b)


def glu_derivative(x, split_dim=-1):
    # This is a simplified version that returns gradients w.r.t inputs
    split_size = x.shape[split_dim] // 2
    a, b = np.split(x, 2, axis=split_dim)
    s = sigmoid(b)
    da = s  # gradient w.r.t a
    db = a * s * (1 - s)  # gradient w.r.t b
    return da, db


def swiglu(x, split_dim=-1):
    # SwiGLU splits input in half
    if isinstance(x, np.ndarray):
        a, b = np.split(x, 2, axis=split_dim)
        return a * swish(b)


def swiglu_derivative(x, split_dim=-1):
    # Gradients w.r.t inputs
    a, b = np.split(x, 2, axis=split_dim)
    sw = swish(b)
    swd = swish_derivative(b)
    da = sw  # gradient w.r.t a
    db = a * swd  # gradient w.r.t b
    return da, db

def gelu_erf(x):
    """GELU activation function using the error function (erf)."""
    return 0.5 * x * (1 + erf(x / np.sqrt(2.0)))

def gelu_erf_derivative(x):
    """Derivative of the GELU activation function using erf."""
    phi = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2) # PDF of standard normal
    Phi = 0.5 * (1 + erf(x / np.sqrt(2.0))) # CDF of standard normal
    return Phi + x * phi

# PyTorch implementations for validation
def pytorch_relu(x):
    return torch.relu(torch.tensor(x)).numpy()

def pytorch_sigmoid(x):
    return torch.sigmoid(torch.tensor(x)).numpy()

def pytorch_leaky_relu(x, alpha=0.01):
    return torch.nn.functional.leaky_relu(torch.tensor(x), alpha).numpy()

def pytorch_elu(x, alpha=1.0):
    return torch.nn.functional.elu(torch.tensor(x), alpha).numpy()

# def pytorch_gelu(x):
#     return torch.nn.functional.gelu(torch.tensor(x)).numpy()

def pytorch_gelu(x):
    # PyTorch's default GELU uses the erf implementation
    return torch.nn.functional.gelu(torch.tensor(x, dtype=torch.float64)).numpy() # Use float64 for precision


def pytorch_swish(x):
    # PyTorch calls this SiLU
    return torch.nn.functional.silu(torch.tensor(x)).numpy()

def pytorch_tanh(x):
    return torch.tanh(torch.tensor(x)).numpy()

def pytorch_glu(x, dim=-1):
    return torch.nn.functional.glu(torch.tensor(x), dim).numpy()

# PyTorch doesn't have a direct SwiGLU implementation, so we'll implement it
def pytorch_swiglu(x, dim=-1):
    a, b = torch.chunk(torch.tensor(x), 2, dim=dim)
    return (a * torch.nn.functional.silu(b)).numpy()


def validate_activations():
    # Create input data
    x = np.linspace(-5, 5, 1000)

    # Dictionary to hold functions and their PyTorch equivalents
    functions = {
        "ReLU": (relu, pytorch_relu),
        "Sigmoid": (sigmoid, pytorch_sigmoid),
        "Leaky ReLU": (lambda x: leaky_relu(x, 0.01), lambda x: pytorch_leaky_relu(x, 0.01)),
        "ELU": (lambda x: elu(x, 1.0), lambda x: pytorch_elu(x, 1.0)),
        "GELU": (gelu, pytorch_gelu),
        "GELU_erf": (gelu_erf, pytorch_gelu),
        "Swish": (swish, pytorch_swish),
        "TanH": (tanh_custom, pytorch_tanh)
    }

    # Validate functions
    for name, (custom_fn, pytorch_fn) in functions.items():
        custom_result = custom_fn(x)
        pytorch_result = pytorch_fn(x)
        max_diff = np.max(np.abs(custom_result - pytorch_result))
        print(f"{name} max difference: {max_diff}")

    # Special handling for GLU and SwiGLU which need 2D input
    x_2d = np.random.randn(100, 10)  # Random 2D array

    # Double the last dimension for GLU and SwiGLU
    x_doubled = np.concatenate([x_2d, x_2d], axis=-1)

    # GLU validation
    custom_glu = glu(x_doubled)
    pytorch_glu_res = pytorch_glu(x_doubled)
    glu_diff = np.max(np.abs(custom_glu - pytorch_glu_res))
    print(f"GLU max difference: {glu_diff}")

    # SwiGLU validation
    custom_swiglu = swiglu(x_doubled)
    pytorch_swiglu_res = pytorch_swiglu(x_doubled)
    swiglu_diff = np.max(np.abs(custom_swiglu - pytorch_swiglu_res))
    print(f"SwiGLU max difference: {swiglu_diff}")

    # Plot the activation functions
    plt.figure(figsize=(20, 15))
    for i, (name, (custom_fn, _)) in enumerate(functions.items(), 1):
        plt.subplot(3, 4, i)
        y = custom_fn(x)

        # For the derivative, we need to handle each function separately
        if name == "ReLU":
            y_derivative = relu_derivative(x)
        elif name == "Sigmoid":
            y_derivative = sigmoid_derivative(x)
        elif name == "Leaky ReLU":
            y_derivative = leaky_relu_derivative(x, 0.01)
        elif name == "ELU":
            y_derivative = elu_derivative(x, 1.0)
        elif name == "GELU":
            y_derivative = gelu_derivative(x)
        elif name == "GELU_erf":
            y_derivative = gelu_erf_derivative(x)
        elif name == "Swish":
            y_derivative = swish_derivative(x)
        elif name == "TanH":
            y_derivative = tanh_derivative(x)

        plt.plot(x, y, label=name)
        plt.plot(x, y_derivative, label=f"{name} derivative", linestyle='--')
        plt.grid(True)
        plt.legend()
        plt.title(name)

    # Add GLU and SwiGLU plots (simplified for 1D visualization)
    plt.subplot(3, 4, 9)
    a = x
    b = x
    plt.plot(x, a * sigmoid(b), label="GLU (simplified)")
    plt.plot(x, sigmoid(b), label="GLU derivative w.r.t a (simplified)", linestyle='--')
    plt.plot(x, a * sigmoid(b) * (1 - sigmoid(b)), label="GLU derivative w.r.t b (simplified)", linestyle=':')
    plt.grid(True)
    plt.legend()
    plt.title("GLU (simplified)")

    plt.subplot(3, 4, 10)
    plt.plot(x, a * swish(b), label="SwiGLU (simplified)")
    plt.plot(x, swish(b), label="SwiGLU derivative w.r.t a (simplified)", linestyle='--')
    plt.plot(x, a * swish_derivative(b), label="SwiGLU derivative w.r.t b (simplified)", linestyle=':')
    plt.grid(True)
    plt.legend()
    plt.title("SwiGLU (simplified)")

    plt.tight_layout()
    plt.savefig("activation_functions.png")
    plt.show()


if __name__ == "__main__":
    validate_activations()