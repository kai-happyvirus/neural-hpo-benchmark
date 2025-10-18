#!/usr/bin/env python3
"""Quick progress demo script for video recording"""

import time
import sys
from datetime import datetime

def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{int(mins)}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h {int(mins)}m"

def print_status_update(message, level="info"):
    """Print status update with timestamp."""
    current_time = datetime.now().strftime("%H:%M:%S")
    icons = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}
    icon = icons.get(level, "ℹ️")
    print(f"{icon} [{current_time}] {message}")

def simulate_progress_bar(task_name, duration, steps=20):
    """Simulate a progress bar for long-running tasks."""
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
    return True

def main():
    """Demo the enhanced progress features"""
    start_time = time.time()
    
    print(f"🎬" * 25)
    print("🎥 ENHANCED PROGRESS DEMO")
    print(f"🎬" * 25)
    
    algorithms = ['GA', 'DE', 'PSO', 'Grid', 'Random']
    datasets = ['MNIST']
    
    total_experiments = len(algorithms) * len(datasets)
    completed = 0
    
    print(f"\n📊 Total experiments: {total_experiments}")
    print(f"🧬 Algorithms: {', '.join(algorithms)}")
    print(f"📁 Datasets: {', '.join(datasets)}")
    
    for i, algorithm in enumerate(algorithms):
        for j, dataset in enumerate(datasets):
            current_experiment = i * len(datasets) + j + 1
            
            print_status_update(
                f"Starting experiment {current_experiment}/{total_experiments}: "
                f"{algorithm} on {dataset}", "info"
            )
            
            # Simulate algorithm execution
            simulate_progress_bar(f"{algorithm} Optimization", 2.0)
            
            completed += 1
            progress_percent = (completed / total_experiments) * 100
            elapsed = time.time() - start_time
            
            print_status_update(
                f"Experiment {completed}/{total_experiments} complete "
                f"({progress_percent:.0f}%): 85.{i}% accuracy "
                f"in {elapsed:.1f}s total", "success"
            )
    
    total_time = time.time() - start_time
    
    print(f"\n🎊" * 20)
    print("🎬 DEMO COMPLETE!")
    print(f"🎊" * 20)
    print(f"⏱️  Total time: {format_time(total_time)}")
    print(f"📊 All {total_experiments} experiments completed successfully!")

if __name__ == "__main__":
    main()