#!/usr/bin/env python3
"""
Quick checkpoint test script
"""

import os
import sys
sys.path.append('./src')

from experiment_manager import ExperimentManager
from evolutionary_algorithms import create_optimizer
import yaml

def test_checkpoints():
    """Test checkpoint functionality"""
    print("🧪 Testing checkpoint functionality...")
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment manager
    exp_manager = ExperimentManager("checkpoint_test")
    
    # Mock evaluation function
    def mock_evaluate(hyperparams):
        return 0.85  # Mock fitness
    
    # Create GA optimizer with experiment manager
    optimizer = create_optimizer('ga', config, mock_evaluate, exp_manager)
    
    # Run a few generations to test checkpoints
    algorithm_params = {
        'population_size': 4,
        'generations': 3,
        'mutation_rate': 0.1
    }
    
    print("🚀 Running GA with checkpoint test...")
    results = optimizer.optimize(algorithm_params)
    
    # Check if checkpoints were created
    checkpoint_dir = exp_manager.experiment_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.rglob("*.pkl"))
        print(f"✅ Found {len(checkpoints)} checkpoint files:")
        for cp in checkpoints:
            print(f"   📁 {cp}")
    else:
        print("❌ No checkpoint directory found")
    
    print(f"📊 Results: {results.get('best_fitness', 0):.3f}")
    print(f"📂 Experiment directory: {exp_manager.experiment_dir}")

if __name__ == "__main__":
    test_checkpoints()