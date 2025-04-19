import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import copy
from numpy_optimizers import SGD, SGDM, Adagrad, RMSProp, Adadelta, Adam, AdamW, Adafactor

def load_mnist():
    """Load a batch of MNIST data"""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match VGG input
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Get a single batch
    sample_data, sample_target = next(iter(train_loader))
    return sample_data, sample_target

def create_simple_cnn_model():
    """Create a simple CNN model for MNIST"""
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    return model

def create_vgg_model():
    """Create a VGG model for MNIST"""
    # Get VGG16 with batch normalization
    model = models.vgg11_bn(pretrained=False)
    
    # Modify first layer to accept grayscale images (1 channel)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    
    # Modify classifier for MNIST (10 classes)
    model.classifier[-1] = nn.Linear(4096, 10)
    
    return model


def get_numpy_params_and_grads(model):
    """Extract parameters from PyTorch model and convert to NumPy arrays"""
    numpy_params = []
    for param in model.parameters():
        if param.requires_grad:
            numpy_params.append(param.detach().cpu().numpy().copy())
    return numpy_params

def create_numpy_optimizers(numpy_params):
    """Create NumPy optimizers with the same parameters as PyTorch optimizers"""
    numpy_optimizers = {
        'SGD': SGD(copy.deepcopy(numpy_params), lr=0.01),
        'SGDM': SGDM(copy.deepcopy(numpy_params), lr=0.01, momentum=0.9),
        'Adagrad': Adagrad(copy.deepcopy(numpy_params), lr=0.01),
        'RMSProp': RMSProp(copy.deepcopy(numpy_params), lr=0.01, alpha=0.99),
        'Adadelta': Adadelta(copy.deepcopy(numpy_params), rho=0.9),
        'Adam': Adam(copy.deepcopy(numpy_params), lr=0.001, betas=(0.9, 0.999)),
        'AdamW': AdamW(copy.deepcopy(numpy_params), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01),
        'Adafactor': Adafactor(copy.deepcopy(numpy_params), relative_step=False, lr=0.001)
    }
    return numpy_optimizers

def create_pytorch_optimizers(model):
    """Create PyTorch optimizers for comparison"""
    pytorch_optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=0.01),
        'SGDM': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        'Adagrad': optim.Adagrad(model.parameters(), lr=0.01),
        'RMSProp': optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99),
        'Adadelta': optim.Adadelta(model.parameters(), rho=0.9),
        'Adam': optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999)),
        'AdamW': optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    }
    
    try:
        # Import Adafactor if available from transformers
        from transformers.optimization import Adafactor as TransformersAdafactor
        pytorch_optimizers['Adafactor'] = TransformersAdafactor(
            model.parameters(), relative_step=False, lr=0.001
        )
    except ImportError:
        print("Transformers library not available, skipping Adafactor PyTorch comparison")
    
    return pytorch_optimizers

def validate_optimizers():
    """Validate NumPy optimizers against PyTorch implementations for 2 epochs"""
    print("Loading MNIST data...")
    data, target = load_mnist()
    
    # Results dictionary: optimizer -> list of avg diffs per epoch
    results = {}
    
    optimizer_names = ['SGD', 'SGDM', 'Adagrad', 'RMSProp', 'Adadelta', 'Adam', 'AdamW', ] # 'Adafactor'
    
    for optimizer_name in optimizer_names:
        print(f"\nValidating {optimizer_name}...")
        
        # Create a fresh model for each optimizer
        model = create_simple_cnn_model()
        criterion = nn.CrossEntropyLoss()
        
        # Forward and backward pass with PyTorch
        torch_model = copy.deepcopy(model)
        pytorch_optimizers = create_pytorch_optimizers(torch_model)
        
        if optimizer_name not in pytorch_optimizers:
            print(f"Skipping {optimizer_name} - not available in PyTorch")
            continue
            
        pytorch_opt = pytorch_optimizers[optimizer_name]
        
        # Get initial parameters
        numpy_params = get_numpy_params_and_grads(model)
        numpy_optimizers = create_numpy_optimizers(numpy_params)
        numpy_opt = numpy_optimizers[optimizer_name]
        
        epoch_diffs = []
        for epoch in range(10):
            # Train for one epoch on a single batch
            torch_model.train()
            pytorch_opt.zero_grad()
            output = torch_model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Get gradients from PyTorch model
            numpy_grads = []
            for param in torch_model.parameters():
                if param.requires_grad:
                    numpy_grads.append(param.grad.detach().cpu().numpy().copy())
            
            # Apply both optimizers
            pytorch_opt.step()
            numpy_opt.step(numpy_grads)
            
            # Compare parameters after optimization
            torch_params_after = []
            for param in torch_model.parameters():
                if param.requires_grad:
                    torch_params_after.append(param.detach().cpu().numpy())
            
            # Calculate differences
            differences = []
            for i in range(len(torch_params_after)):
                diff = np.abs(torch_params_after[i] - numpy_optimizers[optimizer_name].params[i])
                rel_diff = np.mean(diff) / (np.mean(np.abs(torch_params_after[i])) + 1e-8)
                differences.append(rel_diff)
            
            avg_diff = np.mean(differences)
            epoch_diffs.append(avg_diff)
            print(f"Epoch {epoch+1} average relative difference: {avg_diff:.8f}")
        
        results[optimizer_name] = epoch_diffs
    
    print("\nSummary of validation results (per epoch):")
    for name, diffs in results.items():
        for epoch, diff in enumerate(diffs):
            status = "✓ PASS" if diff < 1e-5 else "✗ FAIL"
            print(f"{name} - Epoch {epoch+1}: {diff:.8f} - {status}")

if __name__ == "__main__":
    validate_optimizers()
