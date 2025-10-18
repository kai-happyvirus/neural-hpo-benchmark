#!/usr/bin/env python3
"""
Multiprocessing-safe evaluation function for evolutionary algorithms
This module provides a standalone evaluation function that can be safely pickled
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np

# Global variables for multiprocessing
_GLOBAL_CONFIG = None
_GLOBAL_DATASET = None
_GLOBAL_BATCH_SIZE = None

def initialize_worker(config_dict, dataset, batch_size):
    """Initialize worker process with shared data - Best Practice: Ensure config is properly serialized"""
    global worker_config, worker_data_manager, worker_batch_size
    
    # Best Practice: Validate config is properly passed
    if config_dict is None:
        print("❌ Critical Error: Config is None in worker initialization")
        return
    
    worker_config = config_dict
    worker_batch_size = batch_size
    
    try:
        # Best Practice: Initialize data manager with error handling
        worker_data_manager = DataManager(config_dict)
        print(f"✅ Worker initialized successfully with device: {config_dict.get('hardware', {}).get('device', 'cpu')}")
    except Exception as e:
        print(f"❌ Worker initialization failed: {e}")
        # Create minimal fallback data manager
        fallback_config = {
            'hardware': {'device': 'cpu'},
            'data_dir': './data',
            'datasets': {'mnist': {'name': 'MNIST'}},
        }
        worker_data_manager = DataManager(fallback_config)
        worker_config = fallback_config

def safe_evaluate_individual(individual):
    """
Multiprocessing-safe evaluation functions with cross-platform compatibility
Implements best practices for reliable worker pool execution
"""

import multiprocessing as mp
import platform
import os
import sys
import traceback

# Import required modules for worker processes
try:
    from .data_loader import DataManager
    from .model_trainer import ModelTrainer
except ImportError:
    # Handle relative import issues
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent
    sys.path.insert(0, str(src_path))
    
    from data_loader import DataManager
    from model_trainer import ModelTrainer
    try:
        # Import modules after path is set (critical for multiprocessing)
        from trainer import ModelTrainer
        from data_loader import DataManager
        from evolutionary_algorithms import HyperparameterSpace
        
        # Use global configuration
        config = _GLOBAL_CONFIG
        dataset = _GLOBAL_DATASET
        batch_size = _GLOBAL_BATCH_SIZE
        
        # Create fresh instances for this process
        trainer = ModelTrainer()
        data_manager = DataManager(config)
        
        # Get data loaders
        train_loader, val_loader, test_loader = data_manager.get_dataloaders(
            dataset, batch_size, config['training']['validation_split']
        )
        
        # Decode hyperparameters
        space = HyperparameterSpace(config)
        hyperparams = space.decode_individual(individual)
        
        # Merge with training config
        full_hyperparams = {**config['training'], **hyperparams}
        
        # Evaluate
        fitness = trainer.evaluate_hyperparameters(
            full_hyperparams, train_loader, val_loader, test_loader, dataset
        )
        
        return fitness
        
    except Exception as e:
        print(f"❌ Worker evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0  # Return poor fitness for failed evaluations

def create_safe_evaluation_function(config, dataset, batch_size):
    """
    Create a multiprocessing-safe evaluation function following best practices
    """
    
    # Best Practice: Check system compatibility first
    import platform
    
    # Check if multiprocessing is enabled AND safe on this platform
    use_multiprocessing = config.get('hardware', {}).get('use_multiprocessing', False)
    
    # Best Practice: Platform-specific safety checks
    if platform.system() == "Windows":
        use_multiprocessing = False  # Force disable on Windows
        print("🔧 Windows detected - multiprocessing disabled for compatibility")
    elif not use_multiprocessing:
        print("🔧 Multiprocessing disabled in config - using single-threaded evaluation")
    
    if not use_multiprocessing:
        # Best Practice: Graceful fallback with clear messaging
        print("🔧 Using single-threaded evaluation (safer, works everywhere)")
        
        try:
            # Create standard evaluation function with error handling
            from data_loader import DataManager
            from trainer import ModelTrainer
            
            # Best Practice: Get device from config with fallback
            device = config.get('hardware', {}).get('device', 'cpu')
            
            data_manager = DataManager(config)
            trainer = ModelTrainer(device=device)
            
            train_loader, val_loader, test_loader = data_manager.get_dataloaders(
                dataset, batch_size, config['training']['validation_split']
            )
            
            def single_thread_evaluate(hyperparams):
                """Single-threaded evaluation function with error handling"""
                try:
                    full_hyperparams = {**config['training'], **hyperparams}
                    return trainer.evaluate_hyperparameters(
                        full_hyperparams, train_loader, val_loader, test_loader, dataset
                    )
                except Exception as e:
                    print(f"⚠️  Evaluation failed: {e}")
                    return 0.0  # Best Practice: Return poor fitness instead of crashing
            
            print(f"   ✅ Single-threaded evaluation ready (device: {device})")
            return single_thread_evaluate
            
        except Exception as e:
            print(f"❌ Failed to create single-threaded evaluator: {e}")
            
            # Best Practice: Ultra-safe fallback
            def emergency_evaluate(hyperparams):
                """Emergency fallback evaluator"""
                print(f"⚠️  Using emergency fallback evaluator")
                return 0.1  # Return minimal non-zero fitness
            
            return emergency_evaluate
    
    else:
        print("🚀 Using multiprocessing evaluation (with safety fixes)")
        
        # Set up multiprocessing
        import multiprocessing as mp
        
        # Configure multiprocessing method
        start_method = config.get('hardware', {}).get('multiprocessing_start_method', 'spawn')
        try:
            mp.set_start_method(start_method, force=True)
            print(f"   ✅ Set multiprocessing start method: {start_method}")
        except RuntimeError as e:
            print(f"   ⚠️  Could not set start method: {e}")
        
        # Best Practice: Validate and limit worker count
        max_workers = config.get('hardware', {}).get('max_parallel_processes', 2)
        
        # Best Practice: Ensure worker count is reasonable (use standard library)
        system_cpu_count = os.cpu_count() or 2  # Fallback to 2 if None
        max_safe_workers = min(max_workers, system_cpu_count, 4)  # Never exceed 4
        
        if max_safe_workers <= 0:
            print("⚠️  No workers available - falling back to single-threaded")
            return create_safe_evaluation_function(
                {**config, 'hardware': {**config.get('hardware', {}), 'use_multiprocessing': False}},
                dataset, batch_size
            )
        
        print(f"   🚀 Creating worker pool with {max_safe_workers} processes")
        
        try:
            # Best Practice: Create pool with error handling
            pool = mp.Pool(
                processes=max_safe_workers,
                initializer=initialize_worker,
                initargs=(config, dataset, batch_size)
            )
            print(f"   ✅ Multiprocessing pool created successfully")
            
        except Exception as e:
            print(f"❌ Failed to create multiprocessing pool: {e}")
            print("   🔄 Falling back to single-threaded mode")
            return create_safe_evaluation_function(
                {**config, 'hardware': {**config.get('hardware', {}), 'use_multiprocessing': False}},
                dataset, batch_size
            )
        
        def multiprocess_evaluate_batch(individuals):
            """Evaluate a batch of individuals using multiprocessing"""
            try:
                results = pool.map(safe_evaluate_individual, individuals)
                return results
            except Exception as e:
                print(f"❌ Multiprocessing batch evaluation failed: {e}")
                # Fallback to single-threaded
                return [safe_evaluate_individual(ind) for ind in individuals]
        
        def multiprocess_evaluate_single(hyperparams):
            """Single evaluation wrapper for multiprocessing"""
            # For single evaluations, we need to encode hyperparams to individual format
            from evolutionary_algorithms import HyperparameterSpace
            space = HyperparameterSpace(config)
            individual = space.encode_individual(hyperparams)
            
            try:
                result = safe_evaluate_individual(individual)
                return result
            except Exception as e:
                print(f"❌ Single evaluation failed: {e}")
                return 0.0
        
        # Store pool reference for cleanup
        multiprocess_evaluate_single._pool = pool
        multiprocess_evaluate_batch._pool = pool
        
        return multiprocess_evaluate_single

def cleanup_multiprocessing():
    """Clean up multiprocessing resources"""
    try:
        import multiprocessing as mp
        # Close any active pools
        for obj in list(globals().values()):
            if hasattr(obj, '_pool') and obj._pool:
                obj._pool.close()
                obj._pool.join()
        print("✅ Multiprocessing resources cleaned up")
    except Exception as e:
        print(f"⚠️  Error cleaning up multiprocessing: {e}")

if __name__ == "__main__":
    print("🧪 Testing Multiprocessing-Safe Evaluation")
    
    # Test configuration
    config = {
        'hardware': {
            'use_multiprocessing': True,
            'max_parallel_processes': 2,
            'multiprocessing_start_method': 'spawn'
        },
        'hyperparameters': {
            'learning_rate': {'min': 0.001, 'max': 0.1, 'log_scale': True},
            'batch_size': [64, 128],
            'dropout_rate': {'min': 0.0, 'max': 0.5},
            'hidden_units': [128, 256],
            'optimizer': ['adam', 'sgd'],
            'weight_decay': {'min': 0.0, 'max': 0.01}
        },
        'training': {
            'max_epochs': 3,
            'early_stopping_patience': 2,
            'validation_split': 0.2
        }
    }
    
    print("Creating safe evaluation function...")
    eval_func = create_safe_evaluation_function(config, 'mnist', 64)
    
    # Test evaluation
    test_hyperparams = {
        'learning_rate': 0.01,
        'dropout_rate': 0.2,
        'weight_decay': 0.001,
        'hidden_units': 128,
        'optimizer': 'adam'
    }
    
    print("Testing evaluation...")
    result = eval_func(test_hyperparams)
    print(f"✅ Test result: {result}")
    
    cleanup_multiprocessing()
    print("🎉 Test completed!")