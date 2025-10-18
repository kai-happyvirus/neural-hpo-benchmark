#!/usr/bin/env python3
"""
Master Analysis Generator
Run this after all experiments complete to generate comprehensive comparison figures
"""

import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.append('src')

from result_analyzer import ResultAnalyzer
from experiment_manager import ExperimentManager

def generate_master_analysis():
    """Generate comprehensive analysis comparing all algorithms"""
    
    # Find all experiment directories
    results_dir = Path("results")
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('hpo_experiment')]
    
    print(f"🔍 Found {len(experiment_dirs)} experiments:")
    for exp_dir in experiment_dirs:
        print(f"   📁 {exp_dir.name}")
    
    if len(experiment_dirs) < 2:
        print("❌ Need at least 2 experiments for comparison")
        return
    
    # Create master comparison directory
    master_dir = results_dir / "master_analysis"
    master_dir.mkdir(exist_ok=True)
    
    print(f"\n🎨 Generating master analysis in {master_dir}")
    
    # Generate comparison figures
    try:
        # Load all experiment managers
        experiments = []
        for exp_dir in experiment_dirs:
            try:
                exp_manager = ExperimentManager(experiment_name=exp_dir.name)
                experiments.append(exp_manager)
            except Exception as e:
                print(f"⚠️  Could not load {exp_dir.name}: {e}")
        
        if experiments:
            # Generate comprehensive analysis
            analyzer = ResultAnalyzer()
            
            # Algorithm performance comparison
            print("📊 Generating algorithm comparison plots...")
            analyzer.compare_algorithms(experiments, save_dir=str(master_dir))
            
            # Hyperparameter analysis
            print("🔥 Generating hyperparameter heatmaps...")  
            analyzer.analyze_hyperparameter_trends(experiments, save_dir=str(master_dir))
            
            # Statistical summary
            print("📉 Generating statistical summary...")
            analyzer.generate_statistical_summary(experiments, save_dir=str(master_dir))
            
            print(f"✅ Master analysis complete! Check {master_dir}")
            
        else:
            print("❌ No valid experiments found")
            
    except Exception as e:
        print(f"❌ Error generating analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    generate_master_analysis()