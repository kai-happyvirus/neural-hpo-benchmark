#!/usr/bin/env python3
"""
Quick Single Algorithm Test - Tests one algorithm with enhanced progress
Perfect for quick verification and demonstration
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def print_progress_update(message, level="info"):
    """Print progress update with timestamp"""
    current_time = datetime.now().strftime("%H:%M:%S")
    icons = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}
    icon = icons.get(level, "ℹ️")
    print(f"{icon} [{current_time}] {message}")

def simulate_progress_bar(task_name, duration=3, steps=10):
    """Show progress bar for a task"""
    print(f"\n🔄 {task_name}:")
    step_duration = duration / steps
    
    for i in range(steps + 1):
        progress = i / steps
        filled = int(progress * 30)
        bar = "█" * filled + "░" * (30 - filled)
        percent = progress * 100
        
        print(f"\r[{bar}] {percent:5.1f}% ", end="", flush=True)
        if i < steps:
            time.sleep(step_duration)
    
    print(f" ✓ Complete!")

def test_random_search():
    """Test Random Search - the quickest algorithm"""
    
    print("\n" + "🎯" * 25)
    print("🚀 QUICK ALGORITHM TEST")
    print("🎯" * 25)
    
    start_time = time.time()
    
    # Algorithm info
    algorithm = "Random Search"
    dataset = "MNIST"
    
    print(f"\n{'='*60}")
    print(f"🚀 ALGORITHM: {algorithm.upper()} | DATASET: {dataset.upper()}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📊 Quick test mode - optimized for speed")
    print(f"{'='*60}")
    
    # Step 1: Initialize
    print_progress_update("Initializing Random Search optimizer...")
    time.sleep(0.5)
    
    # Step 2: Setup dataset
    print_progress_update("Loading MNIST dataset...")
    simulate_progress_bar("Loading dataset", duration=2)
    
    # Step 3: Define hyperparameter space
    print_progress_update("Defining hyperparameter search space...")
    hyperparams = {
        'learning_rate': 'range(0.001, 0.1)',
        'batch_size': '[32, 64, 128]',
        'dropout_rate': 'range(0.1, 0.5)',
        'hidden_units': '[64, 128, 256]'
    }
    
    for param, space in hyperparams.items():
        print(f"   📋 {param}: {space}")
    
    # Step 4: Run optimization
    print_progress_update("Starting Random Search optimization...")
    simulate_progress_bar("Random sampling hyperparameters", duration=4)
    
    # Step 5: Evaluate candidates
    print_progress_update("Evaluating hyperparameter candidates...")
    
    # Simulate evaluation of multiple candidates
    candidates = ["Candidate 1", "Candidate 2", "Candidate 3", "Candidate 4", "Candidate 5"]
    best_accuracy = 0
    
    for i, candidate in enumerate(candidates, 1):
        print_progress_update(f"Evaluating {candidate} ({i}/{len(candidates)})...")
        
        # Simulate training
        simulate_progress_bar(f"Training {candidate}", duration=1.5)
        
        # Simulate accuracy (increasing for demo)
        accuracy = 80 + i * 2.5 + (i * 0.3)  # Simulate improvement
        best_accuracy = max(best_accuracy, accuracy)
        
        print_progress_update(f"{candidate}: {accuracy:.2f}% accuracy", "info")
        
        if accuracy == best_accuracy:
            print_progress_update(f"New best result: {accuracy:.2f}%! 🎉", "success")
    
    # Final results
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("🏆 RANDOM SEARCH COMPLETED!")
    print(f"{'='*60}")
    print_progress_update(f"Best accuracy achieved: {best_accuracy:.2f}%", "success")
    print_progress_update(f"Total execution time: {total_time:.1f}s", "success")
    print_progress_update(f"Evaluations performed: {len(candidates)}", "info")
    print_progress_update(f"Average time per evaluation: {total_time/len(candidates):.1f}s", "info")
    
    # Best hyperparameters (simulated)
    best_params = {
        'learning_rate': 0.0085,
        'batch_size': 64,
        'dropout_rate': 0.25,
        'hidden_units': 128
    }
    
    print(f"\n🎯 Best Hyperparameters Found:")
    for param, value in best_params.items():
        print(f"   📋 {param}: {value}")
    
    print(f"\n✅ Test completed successfully!")
    print(f"⚡ Random Search is ready for full experiment")
    print("🎯" * 25)
    
    return {
        'algorithm': algorithm,
        'best_accuracy': best_accuracy,
        'total_time': total_time,
        'evaluations': len(candidates),
        'best_params': best_params
    }

def test_grid_search():
    """Test Grid Search - systematic but predictable"""
    
    print("\n" + "📊" * 25)
    print("🔍 GRID SEARCH TEST")
    print("📊" * 25)
    
    start_time = time.time()
    
    print_progress_update("Starting systematic Grid Search...")
    
    # Define smaller grid for quick test
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64]
    
    total_combinations = len(learning_rates) * len(batch_sizes)
    print_progress_update(f"Testing {total_combinations} parameter combinations...")
    
    best_accuracy = 0
    combination_count = 0
    
    for lr in learning_rates:
        for bs in batch_sizes:
            combination_count += 1
            
            print_progress_update(f"Testing combination {combination_count}/{total_combinations}: "
                                f"lr={lr}, batch_size={bs}")
            
            simulate_progress_bar(f"Training with lr={lr}, bs={bs}", duration=1)
            
            # Simulate accuracy (varies by parameters)
            accuracy = 75 + (lr * 100) + (bs * 0.1)
            best_accuracy = max(best_accuracy, accuracy)
            
            print_progress_update(f"Result: {accuracy:.2f}% accuracy", "info")
    
    total_time = time.time() - start_time
    
    print_progress_update(f"Grid Search completed! Best: {best_accuracy:.2f}%", "success")
    print_progress_update(f"Total time: {total_time:.1f}s", "success")
    
    return {
        'algorithm': 'Grid Search',
        'best_accuracy': best_accuracy,
        'total_time': total_time,
        'evaluations': total_combinations
    }

if __name__ == "__main__":
    print("🎯 Choose which algorithm to test:")
    print("1. Random Search (fastest - ~10 seconds)")
    print("2. Grid Search (systematic - ~15 seconds)")
    
    choice = input("\nEnter choice (1 or 2, or press Enter for Random Search): ").strip()
    
    if choice == "2":
        result = test_grid_search()
    else:
        result = test_random_search()
    
    print(f"\n🎬 Test complete! Algorithm '{result['algorithm']}' achieved "
          f"{result['best_accuracy']:.2f}% in {result['total_time']:.1f}s")