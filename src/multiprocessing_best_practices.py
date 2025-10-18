"""
Multiprocessing-safe evaluation functions with cross-platform compatibility
Implements industry best practices for reliable worker pool execution
"""

import multiprocessing as mp
import platform
import os
import sys
import traceback
from pathlib import Path

# Global worker state (best practice for multiprocessing)
worker_config = None
worker_data_manager = None  
worker_batch_size = None

def initialize_worker_process(config_dict, dataset_name, batch_size_val):
    """
    Initialize worker process with shared data
    Best Practice: Robust initialization with fallback handling
    """
    global worker_config, worker_data_manager, worker_batch_size
    
    try:
        # Best Practice: Validate config is properly passed
        if config_dict is None:
            print("❌ Critical Error: Config is None in worker initialization")
            return False
        
        # Set global state
        worker_config = config_dict
        worker_batch_size = batch_size_val
        
        # Import required modules (handle both relative and absolute imports)
        try:
            # First try direct imports
            from data_loader import DataManager
        except ImportError:
            # Add src path and try again
            src_path = Path(__file__).parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from data_loader import DataManager
        
        # Initialize data manager with error handling
        worker_data_manager = DataManager(config_dict)
        print(f"✅ Worker process initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Worker initialization failed: {e}")
        # Create minimal fallback
        worker_config = {
            'hardware': {'device': 'cpu'},
            'data_dir': './data',
            'datasets': {dataset_name: {'name': dataset_name.upper()}},
            'training': {'validation_split': 0.2}
        }
        worker_data_manager = None
        return False

def evaluate_individual_safely(individual, dataset_name):
    """
    Safely evaluate an individual using pre-initialized worker data
    Best Practice: Comprehensive error handling and validation
    """
    try:
        global worker_config, worker_data_manager, worker_batch_size
        
        # Best Practice: Validate worker state
        if worker_config is None:
            print("❌ Critical: Worker config is None")
            return 0.0
            
        # Import required modules in worker context
        try:
            from model_trainer import ModelTrainer
            from hyperparameter_space import HyperparameterSpace
        except ImportError:
            try:
                # Try alternative imports
                from trainer import ModelTrainer
                from evolutionary_algorithms import HyperparameterSpace
            except ImportError:
                print("❌ Critical: Could not import required modules")
                return 0.0
        
        # Initialize or use existing data manager
        if worker_data_manager is None:
            try:
                from data_loader import DataManager
                worker_data_manager = DataManager(worker_config)
            except Exception as e:
                print(f"❌ Failed to create data manager: {e}")
                return 0.0
        
        # Create trainer instance
        trainer = ModelTrainer()
        
        # Get data loaders
        validation_split = worker_config.get('training', {}).get('validation_split', 0.2)
        train_loader, val_loader, test_loader = worker_data_manager.get_dataloaders(
            dataset_name, worker_batch_size, validation_split
        )
        
        # Decode hyperparameters
        space = HyperparameterSpace(worker_config)
        hyperparams = space.decode_individual(individual)
        
        # Merge with training config
        training_config = worker_config.get('training', {})
        full_hyperparams = {**training_config, **hyperparams}
        
        # Evaluate with timeout protection
        fitness = trainer.evaluate_hyperparameters(
            full_hyperparams, train_loader, val_loader, test_loader, dataset_name
        )
        
        return fitness if fitness is not None else 0.0
        
    except Exception as e:
        print(f"❌ Worker evaluation failed: {e}")
        traceback.print_exc()
        return 0.0

def create_evaluation_function(config, dataset, batch_size):
    """
    Create platform-safe evaluation function with best practices
    """
    
    # Best Practice: System compatibility check
    system = platform.system()
    use_multiprocessing = config.get('hardware', {}).get('use_multiprocessing', False)
    
    # Best Practice: Conservative approach - disable on problematic systems
    if system == 'Darwin' and use_multiprocessing:
        print("🔧 macOS detected - using spawn method for safety")
        
    if not use_multiprocessing:
        print("   📱 Using single-threaded evaluation (safer)")
        return create_single_threaded_evaluator(config, dataset, batch_size)
    
    # Best Practice: Validate multiprocessing capability
    try:
        # Test multiprocessing availability
        with mp.Pool(processes=1) as test_pool:
            pass
        print("   ✅ Multiprocessing capability confirmed")
    except Exception as e:
        print(f"   ❌ Multiprocessing test failed: {e}")
        print("   🔄 Falling back to single-threaded mode")
        return create_single_threaded_evaluator(config, dataset, batch_size)
    
    return create_multiprocessing_evaluator(config, dataset, batch_size)

def create_single_threaded_evaluator(config, dataset, batch_size):
    """Create single-threaded evaluator as fallback"""
    
    # Import required modules
    try:
        from model_trainer import ModelTrainer
        from data_loader import DataManager
        from hyperparameter_space import HyperparameterSpace
    except ImportError:
        from trainer import ModelTrainer
        from data_loader import DataManager  
        from evolutionary_algorithms import HyperparameterSpace
    
    # Initialize components
    trainer = ModelTrainer()
    data_manager = DataManager(config)
    space = HyperparameterSpace(config)
    
    def evaluate_single(individual):
        """Single-threaded evaluation function"""
        try:
            # Get data loaders
            validation_split = config.get('training', {}).get('validation_split', 0.2)
            train_loader, val_loader, test_loader = data_manager.get_dataloaders(
                dataset, batch_size, validation_split
            )
            
            # Decode and evaluate
            hyperparams = space.decode_individual(individual)
            full_hyperparams = {**config.get('training', {}), **hyperparams}
            
            fitness = trainer.evaluate_hyperparameters(
                full_hyperparams, train_loader, val_loader, test_loader, dataset
            )
            
            return fitness if fitness is not None else 0.0
            
        except Exception as e:
            print(f"❌ Single-threaded evaluation failed: {e}")
            return 0.0
    
    return evaluate_single

def create_multiprocessing_evaluator(config, dataset, batch_size):
    """Create multiprocessing evaluator with best practices"""
    
    # Best Practice: Validate and limit worker count
    max_workers = config.get('hardware', {}).get('max_parallel_processes', 2)
    system_cpu_count = os.cpu_count() or 2
    max_safe_workers = min(max_workers, system_cpu_count, 4)  # Never exceed 4
    
    if max_safe_workers <= 0:
        print("⚠️  No workers available - falling back to single-threaded")
        return create_single_threaded_evaluator(config, dataset, batch_size)
    
    print(f"   🚀 Creating worker pool with {max_safe_workers} processes")
    
    try:
        # Best Practice: Create pool with error handling
        pool = mp.Pool(
            processes=max_safe_workers,
            initializer=initialize_worker_process,
            initargs=(config, dataset, batch_size)
        )
        print(f"   ✅ Multiprocessing pool created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create multiprocessing pool: {e}")
        print("   🔄 Falling back to single-threaded mode")
        return create_single_threaded_evaluator(config, dataset, batch_size)
    
    def evaluate_population(population):
        """Evaluate population using worker pool"""
        try:
            # Use starmap to pass both individual and dataset
            args = [(individual, dataset) for individual in population]
            results = pool.starmap(evaluate_individual_safely, args)
            return results
        except Exception as e:
            print(f"❌ Population evaluation failed: {e}")
            # Fallback to single evaluation
            fallback_evaluator = create_single_threaded_evaluator(config, dataset, batch_size)
            return [fallback_evaluator(ind) for ind in population]
    
    # Store pool reference for cleanup
    evaluate_population.pool = pool
    
    return evaluate_population

def cleanup_multiprocessing():
    """Best Practice: Clean shutdown of multiprocessing resources"""
    try:
        # Force cleanup of any remaining processes
        for process in mp.active_children():
            process.terminate()
            process.join(timeout=1.0)
        print("   ✅ Multiprocessing cleanup completed")
    except Exception as e:
        print(f"   ⚠️ Cleanup warning: {e}")

# Best Practice: Setup multiprocessing method on import
def setup_multiprocessing_method():
    """Configure multiprocessing method based on platform"""
    try:
        system = platform.system()
        
        if system in ['Darwin', 'Windows']:
            # Force spawn on macOS and Windows for safety
            if hasattr(mp, 'set_start_method'):
                try:
                    mp.set_start_method('spawn', force=True)
                    print(f"   ✅ Set multiprocessing method to 'spawn' for {system}")
                except RuntimeError:
                    pass  # Already set
        
    except Exception as e:
        print(f"   ⚠️ Multiprocessing method setup warning: {e}")

# Setup on module import
setup_multiprocessing_method()