#!/usr/bin/env python3
"""
Portable System Detection - Best Practice Implementation
Automatically detects and configures optimal settings for any system
"""

import torch
import psutil
import platform
import multiprocessing as mp
import yaml
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SystemDetector:
    """Detects system capabilities and creates optimal configuration"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.machine()
        self.python_version = platform.python_version()
        
    def detect_device(self) -> Tuple[str, Dict[str, Any]]:
        """Detect optimal computation device with fallback chain"""
        device_info = {
            'available_devices': [],
            'cuda_available': False,
            'mps_available': False,
            'cuda_device_count': 0,
            'total_memory_gb': 0
        }
        
        # Test CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            try:
                # Actually test CUDA works
                test_tensor = torch.tensor([1.0], device='cuda')
                device_count = torch.cuda.device_count()
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                device_info.update({
                    'cuda_available': True,
                    'cuda_device_count': device_count,
                    'total_memory_gb': memory_gb,
                    'cuda_device_name': torch.cuda.get_device_properties(0).name
                })
                device_info['available_devices'].append('cuda')
                
                logger.info(f"✅ CUDA detected: {device_info['cuda_device_name']}")
                logger.info(f"   Memory: {memory_gb:.1f}GB, Devices: {device_count}")
                
                del test_tensor
                return 'cuda', device_info
                
            except Exception as e:
                logger.warning(f"⚠️  CUDA available but failed test: {e}")
        
        # Test MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Actually test MPS works
                test_tensor = torch.tensor([1.0], device='mps')
                
                device_info.update({
                    'mps_available': True,
                    'total_memory_gb': psutil.virtual_memory().total / (1024**3)
                })
                device_info['available_devices'].append('mps')
                
                logger.info(f"✅ MPS (Apple Silicon) detected")
                logger.info(f"   System Memory: {device_info['total_memory_gb']:.1f}GB")
                
                del test_tensor
                return 'mps', device_info
                
            except Exception as e:
                logger.warning(f"⚠️  MPS available but failed test: {e}")
        
        # Fallback to CPU
        device_info.update({
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        })
        device_info['available_devices'].append('cpu')
        
        logger.info(f"✅ Using CPU fallback")
        logger.info(f"   System Memory: {device_info['total_memory_gb']:.1f}GB")
        
        return 'cpu', device_info
    
    def detect_multiprocessing_capability(self) -> Tuple[bool, int, str]:
        """Detect safe multiprocessing configuration"""
        
        cpu_count = psutil.cpu_count(logical=True)
        physical_cores = psutil.cpu_count(logical=False)
        
        # Platform-specific multiprocessing behavior
        if self.platform == "Windows":
            # Windows has multiprocessing issues with complex imports
            logger.info("⚠️  Windows detected - multiprocessing disabled for safety")
            return False, 0, "spawn"
        
        elif self.platform == "Darwin":  # macOS
            # macOS requires spawn method
            max_workers = min(4, physical_cores)  # Conservative on macOS
            logger.info(f"✅ macOS detected - using spawn method, {max_workers} workers")
            return True, max_workers, "spawn"
        
        elif self.platform == "Linux":
            # Linux supports fork (fastest)
            max_workers = min(8, physical_cores)  # Can be more aggressive on Linux
            logger.info(f"✅ Linux detected - using fork method, {max_workers} workers")
            return True, max_workers, "fork"
        
        else:
            # Unknown platform - be very conservative
            logger.warning(f"⚠️  Unknown platform '{self.platform}' - disabling multiprocessing")
            return False, 0, "spawn"
    
    def detect_optimal_batch_sizes(self, memory_gb: float, device: str) -> Dict[str, int]:
        """Calculate optimal batch sizes based on available memory"""
        
        if device == 'cuda':
            # GPU memory is precious - be conservative
            if memory_gb >= 24:  # High-end GPU
                return {'mnist': 256, 'cifar10': 128}
            elif memory_gb >= 12:  # Mid-range GPU
                return {'mnist': 128, 'cifar10': 64}
            elif memory_gb >= 6:   # Entry-level GPU
                return {'mnist': 64, 'cifar10': 32}
            else:  # Low VRAM
                return {'mnist': 32, 'cifar10': 16}
        
        elif device == 'mps':
            # MPS uses system memory - can be more generous
            if memory_gb >= 32:
                return {'mnist': 256, 'cifar10': 128}
            elif memory_gb >= 16:
                return {'mnist': 128, 'cifar10': 64}
            else:
                return {'mnist': 64, 'cifar10': 32}
        
        else:  # CPU
            # CPU training - very conservative with batch sizes
            if memory_gb >= 16:
                return {'mnist': 64, 'cifar10': 32}
            elif memory_gb >= 8:
                return {'mnist': 32, 'cifar10': 16}
            else:
                return {'mnist': 16, 'cifar10': 8}
    
    def detect_optimal_epochs(self, device: str) -> Dict[str, int]:
        """Calculate reasonable epoch limits based on device"""
        
        if device == 'cuda':
            return {'max_epochs': 50, 'light_epochs': 10}
        elif device == 'mps':
            return {'max_epochs': 30, 'light_epochs': 8}
        else:  # CPU
            return {'max_epochs': 20, 'light_epochs': 5}
    
    def create_portable_config(self) -> Dict[str, Any]:
        """Create a fully portable configuration for current system"""
        
        logger.info("🔍 Detecting system capabilities...")
        
        # Detect all system properties
        device, device_info = self.detect_device()
        use_mp, max_workers, mp_method = self.detect_multiprocessing_capability()
        batch_sizes = self.detect_optimal_batch_sizes(device_info['total_memory_gb'], device)
        epochs = self.detect_optimal_epochs(device)
        
        # Create comprehensive configuration
        config = {
            'system_info': {
                'platform': self.platform,
                'architecture': self.architecture,
                'python_version': self.python_version,
                'pytorch_version': torch.__version__,
                'detected_device': device,
                'total_memory_gb': device_info['total_memory_gb'],
                'cpu_count': psutil.cpu_count(),
                'generation_timestamp': self.get_timestamp()
            },
            
            'hardware': {
                'device': device,
                'use_multiprocessing': use_mp,
                'max_parallel_processes': max_workers,
                'multiprocessing_start_method': mp_method,
                'num_workers': min(4, max_workers) if use_mp else 0,
                'memory_limit_gb': int(device_info['total_memory_gb'] * 0.8),  # Use 80%
                'pin_memory': device == 'cuda'  # Only beneficial for CUDA
            },
            
            'datasets': {
                'mnist': {
                    'name': 'MNIST',
                    'input_size': 784,
                    'num_classes': 10,
                    'batch_sizes': [batch_sizes['mnist'] // 4, batch_sizes['mnist'] // 2, batch_sizes['mnist']]
                },
                'cifar10': {
                    'name': 'CIFAR-10', 
                    'input_size': [3, 32, 32],
                    'num_classes': 10,
                    'batch_sizes': [batch_sizes['cifar10'] // 2, batch_sizes['cifar10']]
                }
            },
            
            'training': {
                'max_epochs': epochs['max_epochs'],
                'early_stopping_patience': min(10, epochs['max_epochs'] // 3),
                'validation_split': 0.2,
                'test_split': 0.1
            },
            
            'hyperparameters': {
                'learning_rate': {'min': 0.0001, 'max': 0.1, 'log_scale': True},
                'batch_size': [32, 64, 128, 256],  # Will be filtered by dataset limits
                'dropout_rate': {'min': 0.0, 'max': 0.5},
                'hidden_units': [64, 128, 256, 512],
                'optimizer': ['adam', 'sgd', 'rmsprop'],
                'weight_decay': {'min': 0.0, 'max': 0.01}
            },
            
            'algorithms': {
                'genetic_algorithm': {
                    'population_size': 20 if use_mp else 10,
                    'generations': 50 if device != 'cpu' else 30,
                    'mutation_rate': 0.1,
                    'crossover_rate': 0.8,
                    'tournament_size': 3
                },
                'differential_evolution': {
                    'population_size': 20 if use_mp else 10,
                    'generations': 50 if device != 'cpu' else 30,
                    'mutation_factor': 0.8,
                    'crossover_rate': 0.7
                },
                'particle_swarm': {
                    'population_size': 20 if use_mp else 10,
                    'generations': 50 if device != 'cpu' else 30,
                    'inertia_weight': 0.7,
                    'cognitive_factor': 1.5,
                    'social_factor': 1.5
                },
                'grid_search': {
                    'max_evaluations': 200 if device != 'cpu' else 50
                },
                'random_search': {
                    'max_evaluations': 200 if device != 'cpu' else 50
                }
            },
            
            'execution_modes': {
                'full_run': {
                    'algorithms': ['grid', 'random', 'ga', 'de', 'pso'],
                    'datasets': ['mnist', 'cifar10'] if device != 'cpu' else ['mnist'],
                    'runs_per_algorithm': 3 if use_mp else 2
                },
                'light_run': {
                    'algorithms': ['random', 'ga'] if not use_mp else ['grid', 'random', 'ga'],
                    'datasets': ['mnist'],
                    'runs_per_algorithm': 1,
                    'max_generations': 5 if device == 'cpu' else 10,
                    'population_size': 5 if device == 'cpu' else 10,
                    'max_epochs': epochs['light_epochs']
                }
            },
            
            'output': {
                'save_checkpoints': True,
                'checkpoint_frequency': 25,  # More frequent for safety
                'save_convergence_plots': True,
                'save_best_models': True,
                'export_results_csv': True,
                'export_results_json': True
            },
            
            'random_seed': 42,
            
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_file': 'experiment.log'
            }
        }
        
        # Log final configuration
        logger.info("✅ Configuration generated successfully:")
        logger.info(f"   Device: {device}")
        logger.info(f"   Multiprocessing: {use_mp} ({'enabled' if use_mp else 'disabled'})")
        logger.info(f"   Max workers: {max_workers}")
        logger.info(f"   Batch sizes: MNIST={batch_sizes['mnist']}, CIFAR-10={batch_sizes['cifar10']}")
        logger.info(f"   Max epochs: {epochs['max_epochs']}")
        
        return config
    
    def get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save_config(self, config: Dict[str, Any], filename: str = "portable_config.yaml") -> Path:
        """Save configuration to file"""
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        config_path = config_dir / filename
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"💾 Configuration saved to: {config_path}")
        return config_path
    
    def verify_config(self, config: Dict[str, Any]) -> bool:
        """Verify the generated configuration works"""
        logger.info("🧪 Verifying configuration...")
        
        try:
            # Test device initialization
            device = config['hardware']['device']
            test_tensor = torch.tensor([1.0], device=device)
            del test_tensor
            logger.info(f"   ✅ Device '{device}' verified")
            
            # Test multiprocessing if enabled
            if config['hardware']['use_multiprocessing']:
                mp_method = config['hardware']['multiprocessing_start_method']
                try:
                    mp.set_start_method(mp_method, force=True)
                    logger.info(f"   ✅ Multiprocessing method '{mp_method}' verified")
                except RuntimeError:
                    pass  # Already set
            
            # Test batch sizes make sense
            for dataset, dataset_config in config['datasets'].items():
                batch_sizes = dataset_config['batch_sizes']
                if all(b > 0 and b <= 512 for b in batch_sizes):
                    logger.info(f"   ✅ {dataset} batch sizes verified: {batch_sizes}")
                else:
                    logger.warning(f"   ⚠️  Unusual batch sizes for {dataset}: {batch_sizes}")
            
            logger.info("✅ Configuration verification passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration verification failed: {e}")
            return False


def create_portable_setup():
    """Main function to create portable configuration"""
    print("🚀 PORTABLE SYSTEM SETUP - Best Practices Implementation")
    print("=" * 60)
    
    detector = SystemDetector()
    
    try:
        # Generate configuration
        config = detector.create_portable_config()
        
        # Save configuration
        config_path = detector.save_config(config)
        
        # Verify configuration
        if detector.verify_config(config):
            print(f"\n✅ SUCCESS! Portable configuration created:")
            print(f"   📄 Config file: {config_path}")
            print(f"   🖥️  Platform: {config['system_info']['platform']}")
            print(f"   🔧 Device: {config['hardware']['device']}")
            print(f"   ⚡ Multiprocessing: {'Enabled' if config['hardware']['use_multiprocessing'] else 'Disabled'}")
            print(f"\n🎯 Usage:")
            print(f"   python run_experiment.py light --config {config_path}")
            print(f"   python run_experiment.py full --config {config_path}")
            
            return config_path
        else:
            print(f"\n⚠️  Configuration created but verification failed.")
            print(f"   The system will use safe fallback settings.")
            return config_path
            
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        print(f"\n❌ Portable setup failed: {e}")
        print(f"   Please check your Python environment and try again.")
        return None


if __name__ == "__main__":
    create_portable_setup()