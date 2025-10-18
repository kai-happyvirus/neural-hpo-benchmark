"""
Data loading and preprocessing for MNIST and CIFAR-10
Optimized for M1 Pro with efficient data loading
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Dict, Any
import os
import numpy as np


class DataManager:
    """Manages data loading and preprocessing for experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_dir = config.get('data_dir', './data')
        self.num_workers = 0  # Always use 0 for single-threaded reliability
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Define transforms
        self.transforms = self._create_transforms()
    
    def _create_transforms(self) -> Dict[str, Any]:
        """Create data transforms for each dataset"""
        transforms_dict = {
            'mnist': {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
            },
            'cifar10': {
                'train': transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                       (0.2023, 0.1994, 0.2010))  # CIFAR-10 mean and std
                ]),
                'test': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                       (0.2023, 0.1994, 0.2010))
                ])
            }
        }
        return transforms_dict
    
    def get_dataloaders(self, dataset: str, batch_size: int, 
                       validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, validation, and test dataloaders
        
        Args:
            dataset: 'mnist' or 'cifar10'
            batch_size: Batch size for data loading
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        dataset = dataset.lower()
        
        if dataset == 'mnist':
            return self._get_mnist_dataloaders(batch_size, validation_split)
        elif dataset == 'cifar10':
            return self._get_cifar10_dataloaders(batch_size, validation_split)
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")
    
    def _get_mnist_dataloaders(self, batch_size: int, 
                              validation_split: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get MNIST dataloaders"""
        
        # Download and load training data
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transforms['mnist']['train']
        )
        
        # Download and load test data
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transforms['mnist']['test']
        )
        
        # Split training data into train and validation
        train_size = int((1 - validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _get_cifar10_dataloaders(self, batch_size: int, 
                                validation_split: float) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get CIFAR-10 dataloaders"""
        
        # Download and load training data
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transforms['cifar10']['train']
        )
        
        # Download and load test data
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transforms['cifar10']['test']
        )
        
        # Split training data into train and validation
        train_size = int((1 - validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_dataset_info(self, dataset: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        dataset = dataset.lower()
        
        # Get sample dataloaders to extract info
        train_loader, val_loader, test_loader = self.get_dataloaders(dataset, batch_size=1)
        
        # Get sample batch
        sample_batch = next(iter(train_loader))
        input_shape = sample_batch[0].shape[1:]  # Exclude batch dimension
        
        info = {
            'name': dataset.upper(),
            'input_shape': input_shape,
            'num_classes': 10,  # Both MNIST and CIFAR-10 have 10 classes
            'train_size': len(train_loader.dataset),
            'val_size': len(val_loader.dataset),
            'test_size': len(test_loader.dataset),
            'total_samples': len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
        }
        
        return info


def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    # Test configuration
    config = {
        'data_dir': './data',
        'hardware': {'num_workers': 2}
    }
    
    data_manager = DataManager(config)
    
    # Test MNIST
    print("\nTesting MNIST:")
    mnist_info = data_manager.get_dataset_info('mnist')
    print(f"MNIST Info: {mnist_info}")
    
    train_loader, val_loader, test_loader = data_manager.get_dataloaders('mnist', batch_size=32)
    
    # Test one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"MNIST batch shape: {data.shape}, target shape: {target.shape}")
        if batch_idx == 0:  # Only test first batch
            break
    
    # Test CIFAR-10
    print("\nTesting CIFAR-10:")
    cifar_info = data_manager.get_dataset_info('cifar10')
    print(f"CIFAR-10 Info: {cifar_info}")
    
    train_loader, val_loader, test_loader = data_manager.get_dataloaders('cifar10', batch_size=32)
    
    # Test one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"CIFAR-10 batch shape: {data.shape}, target shape: {target.shape}")
        if batch_idx == 0:  # Only test first batch
            break
    
    print("Data loading tests completed successfully!")


if __name__ == "__main__":
    test_data_loading()