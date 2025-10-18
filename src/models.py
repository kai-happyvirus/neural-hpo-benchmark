"""
Simple but accurate neural network models for MNIST and CIFAR-10
Optimized for M1 Pro with hyperparameter optimization focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import math


class MNISTNet(nn.Module):
    """Simple but effective neural network for MNIST classification"""
    
    def __init__(self, hyperparams: Dict[str, Any]):
        super(MNISTNet, self).__init__()
        self.hyperparams = hyperparams
        
        # Extract hyperparameters with defaults
        hidden_units = hyperparams.get('hidden_units', 128)
        dropout_rate = hyperparams.get('dropout_rate', 0.2)
        
        # Simple 3-layer architecture
        self.fc1 = nn.Linear(784, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
        self.fc3 = nn.Linear(hidden_units // 2, 10)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.batch_norm1 = nn.BatchNorm1d(hidden_units)
        self.batch_norm2 = nn.BatchNorm1d(hidden_units // 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # First layer
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc3(x)
        return x


class CIFAR10Net(nn.Module):
    """Simple but effective CNN for CIFAR-10 classification"""
    
    def __init__(self, hyperparams: Dict[str, Any]):
        super(CIFAR10Net, self).__init__()
        self.hyperparams = hyperparams
        
        # Extract hyperparameters with defaults
        hidden_units = hyperparams.get('hidden_units', 128)
        dropout_rate = hyperparams.get('dropout_rate', 0.3)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        # After 3 conv+pool layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(64 * 4 * 4, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(dataset: str, hyperparams: Dict[str, Any], device: str = 'auto') -> nn.Module:
    """Factory function to create appropriate model for dataset"""
    
    # Auto-detect device if not specified
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    # Create model
    if dataset.lower() == 'mnist':
        model = MNISTNet(hyperparams)
    elif dataset.lower() == 'cifar10':
        model = CIFAR10Net(hyperparams)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    # Move model to device
    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, hyperparams: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on hyperparameters"""
    optimizer_name = hyperparams.get('optimizer', 'adam').lower()
    learning_rate = hyperparams.get('learning_rate', 0.001)
    weight_decay = hyperparams.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = hyperparams.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=learning_rate, 
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get model summary information"""
    total_params = count_parameters(model)
    
    summary = {
        'model_class': model.__class__.__name__,
        'total_parameters': total_params,
        'hyperparameters': getattr(model, 'hyperparams', {}),
        'device': next(model.parameters()).device.type if len(list(model.parameters())) > 0 else 'cpu'
    }
    
    return summary


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test MNIST model
    mnist_hyperparams = {
        'hidden_units': 128,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'optimizer': 'adam'
    }
    
    mnist_model = create_model('mnist', mnist_hyperparams)
    print(f"MNIST Model: {get_model_summary(mnist_model)}")
    
    # Test CIFAR-10 model
    cifar_hyperparams = {
        'hidden_units': 256,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'optimizer': 'adam'
    }
    
    cifar_model = create_model('cifar10', cifar_hyperparams)
    print(f"CIFAR-10 Model: {get_model_summary(cifar_model)}")
    
    # Test forward pass
    with torch.no_grad():
        # MNIST test
        mnist_input = torch.randn(1, 1, 28, 28)
        mnist_output = mnist_model(mnist_input)
        print(f"MNIST output shape: {mnist_output.shape}")
        
        # CIFAR-10 test
        cifar_input = torch.randn(1, 3, 32, 32)
        cifar_output = cifar_model(cifar_input)
        print(f"CIFAR-10 output shape: {cifar_output.shape}")
    
    print("Model tests completed successfully!")