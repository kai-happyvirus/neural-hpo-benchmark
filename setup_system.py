#!/usr/bin/env python3
"""
System Setup and Configuration Generator
Implements best practices for cross-platform hyperparameter optimization deployment

Usage:
    python setup_system.py           # Quick setup with detection
    python setup_system.py --full    # Full system analysis and setup
    python setup_system.py --test    # Test current configuration
"""

import os
import sys
import json
import yaml
import platform
import subprocess
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

class SystemAnalyzer:
    """Comprehensive system analysis and optimization for academic deployment"""
    
    def __init__(self):
        self.platform = platform.system()
        self.architecture = platform.machine()
        self.python_version = sys.version_info
        self.results = {}
        
    def analyze_system(self) -> Dict[str, Any]:
        """Complete system analysis following best practices"""
        print("🔍 Analyzing system configuration...")
        
        # Core system information
        self.results['system'] = {
            'platform': self.platform,
            'architecture': self.architecture,
            'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            'is_64bit': sys.maxsize > 2**32
        }
        
        # Hardware capabilities
        self.results['hardware'] = self._analyze_hardware()
        
        # Python environment
        self.results['environment'] = self._analyze_environment()
        
        # Multiprocessing capabilities
        self.results['multiprocessing'] = self._analyze_multiprocessing()
        
        # Device acceleration
        self.results['acceleration'] = self._analyze_acceleration()
        
        return self.results
    
    def _analyze_hardware(self) -> Dict[str, Any]:
        """Analyze hardware capabilities with fallbacks"""
        hardware = {}
        
        # CPU Analysis
        try:
            cpu_count = os.cpu_count() or 1
            hardware['cpu_count'] = cpu_count
            hardware['recommended_workers'] = min(cpu_count, 4)  # Conservative limit
            
            # Memory estimation (basic)
            if hasattr(os, 'sysconf') and 'SC_PAGE_SIZE' in os.sysconf_names:
                try:
                    page_size = os.sysconf('SC_PAGE_SIZE')
                    page_count = os.sysconf('SC_PHYS_PAGES')
                    memory_mb = (page_size * page_count) // (1024 * 1024)
                    hardware['memory_mb'] = memory_mb
                except:
                    hardware['memory_mb'] = None
            else:
                hardware['memory_mb'] = None
                
        except Exception as e:
            print(f"   ⚠️ CPU analysis warning: {e}")
            hardware['cpu_count'] = 2
            hardware['recommended_workers'] = 2
            hardware['memory_mb'] = None
        
        return hardware
    
    def _analyze_environment(self) -> Dict[str, Any]:
        """Analyze Python environment and dependencies"""
        env = {
            'python_executable': sys.executable,
            'virtual_env': os.environ.get('VIRTUAL_ENV'),
            'conda_env': os.environ.get('CONDA_DEFAULT_ENV'),
        }
        
        # Check critical dependencies
        critical_packages = ['torch', 'numpy', 'scipy', 'pyyaml']
        env['packages'] = {}
        
        for package in critical_packages:
            try:
                __import__(package)
                env['packages'][package] = '✅ Available'
            except ImportError:
                env['packages'][package] = '❌ Missing'
        
        return env
    
    def _analyze_multiprocessing(self) -> Dict[str, Any]:
        """Analyze multiprocessing capabilities and safety"""
        mp_info = {
            'start_method': None,
            'available_methods': [],
            'recommended_method': None,
            'is_safe': False
        }
        
        try:
            # Get available start methods
            mp_info['available_methods'] = mp.get_all_start_methods()
            
            # Current method
            try:
                mp_info['start_method'] = mp.get_start_method()
            except RuntimeError:
                mp_info['start_method'] = 'not_set'
            
            # Platform-specific recommendations
            if self.platform == 'Darwin':  # macOS
                mp_info['recommended_method'] = 'spawn'
                mp_info['is_safe'] = 'spawn' in mp_info['available_methods']
            elif self.platform == 'Windows':
                mp_info['recommended_method'] = 'spawn'
                mp_info['is_safe'] = True  # Windows defaults to spawn
            else:  # Linux and others
                mp_info['recommended_method'] = 'fork'
                mp_info['is_safe'] = 'fork' in mp_info['available_methods']
                
        except Exception as e:
            print(f"   ⚠️ Multiprocessing analysis warning: {e}")
            mp_info['is_safe'] = False
        
        return mp_info
    
    def _analyze_acceleration(self) -> Dict[str, Any]:
        """Analyze available hardware acceleration"""
        acceleration = {
            'cuda_available': False,
            'mps_available': False,
            'recommended_device': 'cpu'
        }
        
        try:
            import torch
            
            # CUDA check
            if torch.cuda.is_available():
                acceleration['cuda_available'] = True
                acceleration['cuda_device_count'] = torch.cuda.device_count()
                acceleration['cuda_device_name'] = torch.cuda.get_device_name(0)
                acceleration['recommended_device'] = 'cuda'
            
            # MPS check (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                acceleration['mps_available'] = True
                if not acceleration['cuda_available']:  # Prefer CUDA if both available
                    acceleration['recommended_device'] = 'mps'
                    
        except ImportError:
            print("   ⚠️ PyTorch not available - GPU detection skipped")
        
        return acceleration
    
    def generate_optimal_config(self) -> Dict[str, Any]:
        """Generate optimal configuration based on system analysis"""
        if not self.results:
            self.analyze_system()
        
        config = {
            'hardware': {
                'device': self.results['acceleration']['recommended_device'],
                'use_multiprocessing': self.results['multiprocessing']['is_safe'],
                'max_parallel_processes': self.results['hardware']['recommended_workers'],
                'batch_size': self._calculate_optimal_batch_size(),
            },
            'multiprocessing': {
                'start_method': self.results['multiprocessing']['recommended_method'],
                'force_spawn': self.platform in ['Darwin', 'Windows'],
            },
            'algorithms': {
                'execution_order': ['grid_search', 'random_search', 'genetic_algorithm', 'differential_evolution', 'particle_swarm'],
                'default_populations': self._get_safe_population_sizes(),
            }
        }
        
        return config
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory and device"""
        device = self.results['acceleration']['recommended_device']
        memory_mb = self.results['hardware']['memory_mb']
        
        if device == 'cuda':
            return 128  # Conservative for GPU
        elif device == 'mps':
            return 64   # Conservative for Apple Silicon
        elif memory_mb and memory_mb > 8000:  # >8GB RAM
            return 64
        else:
            return 32   # Conservative default
    
    def _get_safe_population_sizes(self) -> Dict[str, int]:
        """Get safe population sizes based on system capabilities"""
        workers = self.results['hardware']['recommended_workers']
        
        if workers >= 4:
            return {
                'genetic_algorithm': 50,
                'differential_evolution': 40,
                'particle_swarm': 30,
            }
        else:
            return {
                'genetic_algorithm': 20,
                'differential_evolution': 15,
                'particle_swarm': 15,
            }

def setup_multiprocessing_safety():
    """Apply multiprocessing safety measures"""
    print("🔧 Configuring multiprocessing safety...")
    
    try:
        # Set start method if not already set
        if hasattr(mp, 'get_start_method'):
            try:
                current_method = mp.get_start_method()
                print(f"   Current multiprocessing method: {current_method}")
            except RuntimeError:
                # Set appropriate method
                system = platform.system()
                if system in ['Darwin', 'Windows']:
                    mp.set_start_method('spawn', force=True)
                    print("   ✅ Set multiprocessing method to 'spawn'")
                else:
                    print("   ✅ Using default multiprocessing method")
    except Exception as e:
        print(f"   ⚠️ Multiprocessing setup warning: {e}")

def install_missing_dependencies():
    """Install missing critical dependencies"""
    print("📦 Checking dependencies...")
    
    critical_packages = {
        'torch': 'torch torchvision',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'pyyaml': 'pyyaml',
    }
    
    missing_packages = []
    for package, install_name in critical_packages.items():
        try:
            __import__(package)
            print(f"   ✅ {package} available")
        except ImportError:
            print(f"   ❌ {package} missing")
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("   ✅ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install dependencies: {e}")
            return False
    
    return True

def save_configuration(config: Dict[str, Any], config_path: str):
    """Save optimized configuration to file"""
    print(f"💾 Saving configuration to {config_path}")
    
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"   ✅ Configuration saved successfully")
        
    except Exception as e:
        print(f"   ❌ Failed to save configuration: {e}")

def test_configuration(config_path: str = None):
    """Test the current configuration"""
    print("🧪 Testing system configuration...")
    
    try:
        # Test imports
        import torch
        import numpy as np
        import scipy
        print("   ✅ Critical imports successful")
        
        # Test device
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("   ✅ MPS available (Apple Silicon)")
        else:
            print("   ✅ CPU computation available")
        
        # Test multiprocessing
        try:
            with mp.Pool(processes=2) as pool:
                result = pool.map(lambda x: x*2, [1, 2, 3])
            print("   ✅ Multiprocessing test successful")
        except Exception as e:
            print(f"   ⚠️ Multiprocessing test failed: {e}")
        
        print("   🎉 System configuration test completed")
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")

def main():
    """Main setup routine"""
    print("🚀 System Setup and Configuration Generator")
    print("=" * 50)
    
    # Parse arguments
    args = sys.argv[1:]
    full_analysis = '--full' in args
    test_only = '--test' in args
    
    if test_only:
        test_configuration()
        return
    
    # Initialize analyzer
    analyzer = SystemAnalyzer()
    
    # Quick or full analysis
    if full_analysis:
        print("\n📊 Performing full system analysis...")
        results = analyzer.analyze_system()
        
        # Display results
        print("\n" + "=" * 30 + " SYSTEM ANALYSIS " + "=" * 30)
        print(f"Platform: {results['system']['platform']} ({results['system']['architecture']})")
        print(f"Python: {results['system']['python_version']}")
        print(f"CPU Cores: {results['hardware']['cpu_count']}")
        print(f"Recommended Workers: {results['hardware']['recommended_workers']}")
        print(f"Multiprocessing Safe: {results['multiprocessing']['is_safe']}")
        print(f"Recommended Device: {results['acceleration']['recommended_device']}")
        print("=" * 76)
    else:
        print("\n⚡ Performing quick system detection...")
        analyzer.analyze_system()
    
    # Setup multiprocessing
    setup_multiprocessing_safety()
    
    # Install dependencies
    if not install_missing_dependencies():
        print("\n❌ Dependency installation failed. Please install manually.")
        return
    
    # Generate configuration
    print("\n⚙️ Generating optimal configuration...")
    config = analyzer.generate_optimal_config()
    
    # Save configuration
    config_path = "config/system_config.yaml"
    save_configuration(config, config_path)
    
    # Test configuration
    test_configuration(config_path)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run: python run_experiment.py --help")
    print("2. Start with: python run_experiment.py light")
    print("3. For specific algorithm: python run_experiment.py full --model ga")

if __name__ == "__main__":
    main()