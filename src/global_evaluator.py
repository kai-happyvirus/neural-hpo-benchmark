"""
Global evaluation functions for multiprocessing support
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from trainer import ModelTrainer


# Global variables to be set by the main process
_trainer = None
_train_loader = None
_val_loader = None 
_test_loader = None
_dataset = None
_training_config = None


def setup_global_evaluator(trainer, train_loader, val_loader, test_loader, dataset, training_config):
    """Setup global evaluation variables for multiprocessing"""
    global _trainer, _train_loader, _val_loader, _test_loader, _dataset, _training_config
    _trainer = trainer
    _train_loader = train_loader
    _val_loader = val_loader
    _test_loader = test_loader
    _dataset = dataset
    _training_config = training_config


def evaluate_hyperparams_global(hyperparams: Dict[str, Any]) -> float:
    """Global evaluation function that can be pickled for multiprocessing"""
    global _trainer, _train_loader, _val_loader, _test_loader, _dataset, _training_config
    
    try:
        # Merge with training config
        full_hyperparams = {**_training_config, **hyperparams}
        
        return _trainer.evaluate_hyperparameters(
            full_hyperparams, _train_loader, _val_loader, _test_loader, _dataset
        )
    except Exception as e:
        print(f"Global evaluation error: {e}")
        return 0.0