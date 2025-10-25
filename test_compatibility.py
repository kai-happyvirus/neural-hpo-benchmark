#!/usr/bin/env python3
"""
Simple compatibility test for neural HPO benchmark
"""

import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_environment():
    """Test PyTorch and device availability"""
    print("Testing Environment")
    print("=" * 30)
    
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # Test devices
    print("\nDevice Availability:")
    print(f"CPU: Available")
    
    if torch.cuda.is_available():
        print(f"CUDA: Available")
    else:
        print(f"CUDA: Not available")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"MPS: Available")
    else:
        print(f"MPS: Not available")


def test_models():
    """Test model creation"""
    print("\nTesting Models")
    print("=" * 30)
    
    try:
        from models import create_model
        from trainer import ModelTrainer
        
        hyperparams = {
            'hidden_units': 64,
            'dropout_rate': 0.2,
            'learning_rate': 0.01,
            'optimizer': 'adam'
        }
        
        # Test MNIST model
        model = create_model('mnist', hyperparams)
        print("MNIST model: OK")
        
        # Test CIFAR10 model
        model = create_model('cifar10', hyperparams)
        print("CIFAR10 model: OK")
        
        # Test trainer
        trainer = ModelTrainer()
        print("Trainer: OK")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Neural HPO Benchmark - Compatibility Test")
    print("=" * 50)
    
    test_environment()
    test_models()
    
    print("\nTest completed!")