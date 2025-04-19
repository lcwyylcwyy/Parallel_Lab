import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import copy
import time
from numpy_optimizers import SGD, SGDM, Adagrad, RMSProp, Adadelta, Adam, AdamW, Adafactor
from optimizer_validation import load_mnist, create_vgg_model, get_numpy_params_and_grads, create_simple_cnn_model

def train_one_epoch(optimizer_name, data, target):
    """Train for one epoch with both NumPy and PyTorch optimizers"""
    # Create a fresh model
    model = create_vgg_model()
    # model = create_simple_cnn_model()
    criterion = nn.CrossEntropyLoss()
    
    # Create PyTorch optimizer
    if optimizer_name == 'SGD':
        torch_optimizer = optim.SGD(model.parameters(), lr=0.01)
    elif optimizer_name == 'SGDM':
        torch_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_name == 'Adagrad':
        torch_optimizer = optim.Adagrad(model.parameters(), lr=0.01)
    elif optimizer_name == 'RMSProp':
        torch_optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)
    elif optimizer_name == 'Adadelta':
        torch_optimizer = optim.Adadelta(model.parameters(), rho=0.9)
    elif optimizer_name == 'Adam':
        torch_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    elif optimizer_name == 'AdamW':
        torch_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    elif optimizer_name == 'Adafactor':
        try:
            from transformers.optimization import Adafactor as TransformersAdafactor
            torch_optimizer = TransformersAdafactor(model.parameters(), lr=0.001, relative_step=False)
        except ImportError:
            print("Adafactor not available in PyTorch, skipping PyTorch comparison")
            return None
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Train with PyTorch
    print(f"\nTraining with PyTorch {optimizer_name}...")
    start_time = time.time()
    model.train()
    torch_optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    torch_optimizer.step()
    pytorch_loss = loss.item()
    pytorch_time = time.time() - start_time
    
    # Get predictions
    _, torch_preds = torch.max(output, 1)
    torch_accuracy = (torch_preds == target).sum().item() / len(target)
    
    # Clone model for NumPy optimizer
    numpy_model = copy.deepcopy(model)
    numpy_model.zero_grad()
    
    # Get parameters and create NumPy optimizer
    numpy_params = get_numpy_params_and_grads(numpy_model)
    
    if optimizer_name == 'SGD':
        numpy_optimizer = SGD(copy.deepcopy(numpy_params), lr=0.01)
    elif optimizer_name == 'SGDM':
        numpy_optimizer = SGDM(copy.deepcopy(numpy_params), lr=0.01, momentum=0.9)
    elif optimizer_name == 'Adagrad':
        numpy_optimizer = Adagrad(copy.deepcopy(numpy_params), lr=0.01)
    elif optimizer_name == 'RMSProp':
        numpy_optimizer = RMSProp(copy.deepcopy(numpy_params), lr=0.01, alpha=0.99)
    elif optimizer_name == 'Adadelta':
        numpy_optimizer = Adadelta(copy.deepcopy(numpy_params), rho=0.9)
    elif optimizer_name == 'Adam':
        numpy_optimizer = Adam(copy.deepcopy(numpy_params), lr=0.001, betas=(0.9, 0.999))
    elif optimizer_name == 'AdamW':
        numpy_optimizer = AdamW(copy.deepcopy(numpy_params), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    elif optimizer_name == 'Adafactor':
        numpy_optimizer = Adafactor(copy.deepcopy(numpy_params), lr=0.001, relative_step=False)
    
    # Train with NumPy optimizer
    print(f"Training with NumPy {optimizer_name}...")
    start_time = time.time()
    
    # Forward pass
    numpy_model.train()
    output = numpy_model(data)
    loss = criterion(output, target)
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    numpy_grads = []
    for param in numpy_model.parameters():
        if param.requires_grad:
            numpy_grads.append(param.grad.detach().cpu().numpy().copy())
    
    # Apply optimizer
    numpy_optimizer.step(numpy_grads)
    numpy_loss = loss.item()
    numpy_time = time.time() - start_time
    
    # Get predictions
    _, numpy_preds = torch.max(output, 1)
    numpy_accuracy = (numpy_preds == target).sum().item() / len(target)
    
    # Calculate parameter differences
    torch_params = []
    for param in model.parameters():
        if param.requires_grad:
            torch_params.append(param.detach().cpu().numpy())
    
    numpy_model_params = []
    param_idx = 0
    for param in numpy_model.parameters():
        if param.requires_grad:
            numpy_model_params.append(numpy_optimizer.params[param_idx])
            param_idx += 1
    
    diffs = []
    for i in range(len(torch_params)):
        diff = np.abs(torch_params[i] - numpy_model_params[i])
        rel_diff = np.mean(diff) / (np.mean(np.abs(torch_params[i])) + 1e-8)
        diffs.append(rel_diff)
    
    avg_diff = np.mean(diffs)
    
    # Print results
    print(f"PyTorch Loss: {pytorch_loss:.4f}, Accuracy: {torch_accuracy:.4f}, Time: {pytorch_time:.4f}s")
    print(f"NumPy Loss: {numpy_loss:.4f}, Accuracy: {numpy_accuracy:.4f}, Time: {numpy_time:.4f}s")
    print(f"Average parameter difference: {avg_diff:.8f}")
    
    return {
        'optimizer': optimizer_name,
        'pytorch_loss': pytorch_loss,
        'pytorch_acc': torch_accuracy,
        'pytorch_time': pytorch_time,
        'numpy_loss': numpy_loss,
        'numpy_acc': numpy_accuracy,
        'numpy_time': numpy_time,
        'param_diff': avg_diff
    }

def main():
    # Load data
    print("Loading MNIST data...")
    data, target = load_mnist()
    
    # List of optimizers to test
    optimizers = ['SGD', 'SGDM', 'Adagrad', 'RMSProp', 'Adadelta', 'Adam', 'AdamW', ] # 'Adafactor'
    
    results = []
    for opt_name in optimizers:
        print(f"\n==== Testing {opt_name} ====")
        result = train_one_epoch(opt_name, data, target)
        if result:
            results.append(result)
    
    # Print summary
    print("\n==== Summary ====")
    print(f"{'Optimizer':<10} {'PyTorch Loss':<12} {'PyTorch Acc':<12} {'NumPy Loss':<12} {'NumPy Acc':<12} {'Param Diff':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['optimizer']:<10} {result['pytorch_loss']:<12.4f} {result['pytorch_acc']:<12.4f} {result['numpy_loss']:<12.4f} {result['numpy_acc']:<12.4f} {result['param_diff']:<12.8f}")

if __name__ == "__main__":
    main()
