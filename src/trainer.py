"""
Training utilities for neural network models
Includes early stopping, device handling, and evaluation metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, List, Optional
import time
import copy
import numpy as np
from collections import defaultdict


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
        
        return self.early_stop


class ModelTrainer:
    """Handles model training with early stopping and metric tracking"""
    
    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        print(f"Using device: {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device for training"""
        if device == 'auto':
            if torch.backends.mps.is_available():
                return 'mps'  # Apple Metal Performance Shaders
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, optimizer: torch.optim.Optimizer,
                   hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a model with the given hyperparameters
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            hyperparams: Dictionary containing training hyperparameters
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Extract training parameters
        max_epochs = hyperparams.get('max_epochs', 50)
        early_stopping_patience = hyperparams.get('early_stopping_patience', 10)
        
        # Move model to device
        model = model.to(self.device)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_acc = self._evaluate_model(model, val_loader, criterion)
            
            # Record metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            history['epoch_times'].append(epoch_time)
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        total_time = time.time() - start_time
        
        # Final evaluation on validation set
        final_val_loss, final_val_acc = self._evaluate_model(model, val_loader, criterion)
        
        # Prepare results
        results = {
            'final_val_accuracy': final_val_acc,
            'final_val_loss': final_val_loss,
            'best_val_accuracy': max(history['val_acc']),
            'best_val_loss': min(history['val_loss']),
            'total_epochs': len(history['train_loss']),
            'total_training_time': total_time,
            'average_epoch_time': np.mean(history['epoch_times']),
            'converged': early_stopping.early_stop,
            'history': history,
            'hyperparameters': hyperparams.copy()
        }
        
        return results
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        """Train model for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def _evaluate_model(self, model: nn.Module, data_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model on given data loader"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = running_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def evaluate_hyperparameters(self, hyperparams: Dict[str, Any], 
                                train_loader: DataLoader, val_loader: DataLoader,
                                test_loader: DataLoader, dataset: str) -> float:
        """
        Evaluate a set of hyperparameters and return fitness score
        This is the main function used by optimization algorithms
        
        Args:
            hyperparams: Dictionary of hyperparameters to evaluate
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            dataset: Dataset name ('mnist' or 'cifar10')
            
        Returns:
            Fitness score (validation accuracy)
        """
        try:
            # Import here to avoid circular imports - use absolute import for multiprocessing
            import sys
            from pathlib import Path
            
            # Add src directory to path for absolute imports
            src_path = Path(__file__).parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            
            from models import create_model, create_optimizer
            
            # Create model and optimizer with device
            model = create_model(dataset, hyperparams, device=str(self.device))
            optimizer = create_optimizer(model, hyperparams)
            
            # Train model
            results = self.train_model(model, train_loader, val_loader, optimizer, hyperparams)
            
            # Return best validation accuracy as fitness
            fitness = results['best_val_accuracy']
            
            # Store additional info in hyperparams for later retrieval
            hyperparams['_training_results'] = results
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating hyperparameters: {e}")
            return 0.0  # Return poor fitness for failed evaluations


def test_training():
    """Test training functionality"""
    print("Testing training functionality...")
    
    # Import required modules
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from models import create_model, create_optimizer
    from data_loader import DataManager
    
    # Test configuration
    config = {
        'data_dir': './data',
        'hardware': {'num_workers': 2}
    }
    
    # Create data manager and get small dataset for testing
    data_manager = DataManager(config)
    train_loader, val_loader, test_loader = data_manager.get_dataloaders('mnist', batch_size=64)
    
    # Limit to small subset for testing
    train_samples = []
    val_samples = []
    
    for i, (data, target) in enumerate(train_loader):
        train_samples.append((data, target))
        if i >= 2:  # Only use first 3 batches
            break
            
    for i, (data, target) in enumerate(val_loader):
        val_samples.append((data, target))
        if i >= 1:  # Only use first 2 batches
            break
    
    # Create small dataloaders
    from torch.utils.data import TensorDataset, DataLoader
    
    train_data = torch.cat([x[0] for x in train_samples])
    train_targets = torch.cat([x[1] for x in train_samples])
    val_data = torch.cat([x[0] for x in val_samples])
    val_targets = torch.cat([x[1] for x in val_samples])
    
    small_train_loader = DataLoader(TensorDataset(train_data, train_targets), batch_size=32, shuffle=True)
    small_val_loader = DataLoader(TensorDataset(val_data, val_targets), batch_size=32, shuffle=False)
    
    # Test hyperparameters
    hyperparams = {
        'hidden_units': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'max_epochs': 5,
        'early_stopping_patience': 3
    }
    
    # Create model and optimizer
    model = create_model('mnist', hyperparams)
    optimizer = create_optimizer(model, hyperparams)
    
    # Create trainer
    trainer = ModelTrainer()
    
    # Train model
    results = trainer.train_model(model, small_train_loader, small_val_loader, optimizer, hyperparams)
    
    print(f"Training completed!")
    print(f"Final validation accuracy: {results['final_val_accuracy']:.2f}%")
    print(f"Total epochs: {results['total_epochs']}")
    print(f"Total training time: {results['total_training_time']:.2f}s")
    
    print("Training test completed successfully!")


if __name__ == "__main__":
    test_training()