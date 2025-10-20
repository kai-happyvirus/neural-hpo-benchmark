#!/usr/bin/env python3
"""Results Plotter"""

import json
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


class SimplePlotter:
    """Plot experiment results"""
    
    def __init__(self, results_dir='results'):
        self.results_dir = Path(results_dir)
        self.figures_dir = Path('figures')
        self.figures_dir.mkdir(exist_ok=True)
    
    def plot_single(self, algorithm, dataset):
        """Plot results for single algorithm"""
        
        pattern = f"{algorithm}_{dataset}_*.json"
        files = list(self.results_dir.glob(pattern))
        
        if not files:
            print(f"No results found for {algorithm} on {dataset}")
            return
        
        result_file = max(files, key=lambda p: p.stat().st_mtime)
        
        with open(result_file) as f:
            results = json.load(f)
        
        print(f"Plotting {algorithm.upper()} on {dataset.upper()}")
        print(f"   File: {result_file.name}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Convergence curves for all runs
        ax1 = axes[0]
        for run in results['runs']:
            history = run['evaluation_history']
            evaluations = [h['evaluation'] for h in history]
            fitness = [h['fitness'] for h in history]
            ax1.plot(evaluations, fitness, 'o-', alpha=0.7, label=f"Run {run['run']}")
        
        ax1.set_xlabel('Evaluation')
        ax1.set_ylabel('Validation Accuracy (%)')
        ax1.set_title(f'{algorithm.upper()} Convergence on {dataset.upper()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Best fitness per run (bar chart)
        ax2 = axes[1]
        runs = [f"Run {run['run']}" for run in results['runs']]
        best_fitness = [run['best_fitness'] for run in results['runs']]
        
        bars = ax2.bar(runs, best_fitness, color='steelblue', alpha=0.7)
        ax2.axhline(np.mean(best_fitness), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(best_fitness):.2f}%')
        ax2.set_ylabel('Best Validation Accuracy (%)')
        ax2.set_title(f'{algorithm.upper()} Performance on {dataset.upper()}')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        fig_name = f"{algorithm}_{dataset}.png"
        fig_path = self.figures_dir / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {fig_path}")
        
        plt.show()
    
    def plot_comparison(self, algorithms, dataset):
        """Compare multiple algorithms"""
        
        all_results = {}
        
        for algo in algorithms:
            pattern = f"{algo}_{dataset}_*.json"
            files = list(self.results_dir.glob(pattern))
            
            if files:
                result_file = max(files, key=lambda p: p.stat().st_mtime)
                with open(result_file) as f:
                    all_results[algo] = json.load(f)
        
        if len(all_results) < 2:
            print(f"Need at least 2 algorithms with results on {dataset}")
            return
        
        print(f"Comparing {len(all_results)} algorithms on {dataset.upper()}")
        
        # Create comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Best fitness comparison (box plot)
        ax1 = axes[0]
        data_for_boxplot = []
        labels = []
        
        for algo, results in all_results.items():
            best_fitness = [run['best_fitness'] for run in results['runs']]
            data_for_boxplot.append(best_fitness)
            labels.append(algo.upper())
        
        bp = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax1.set_ylabel('Best Validation Accuracy (%)')
        ax1.set_title(f'Algorithm Comparison on {dataset.upper()}')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Mean performance with error bars
        ax2 = axes[1]
        means = []
        stds = []
        
        for algo, results in all_results.items():
            best_fitness = [run['best_fitness'] for run in results['runs']]
            means.append(np.mean(best_fitness))
            stds.append(np.std(best_fitness))
        
        x_pos = np.arange(len(labels))
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                      color='steelblue', alpha=0.7, error_kw={'linewidth': 2})
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Mean Best Accuracy (%)')
        ax2.set_title(f'Mean Performance with Std Dev on {dataset.upper()}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax2.text(i, mean + std + 0.2, f'{mean:.2f}%\n±{std:.2f}%',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        fig_name = f"compare_{'_'.join(algorithms)}_{dataset}.png"
        fig_path = self.figures_dir / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {fig_path}")
        
        plt.show()
        
        self._print_comparison_stats(all_results, dataset)
    
    def plot_all(self):
        """Generate comprehensive comparison"""
        
        all_files = list(self.results_dir.glob("*.json"))
        
        if not all_files:
            print("No result files found")
            return
        
        organized = defaultdict(lambda: defaultdict(list))
        
        for file in all_files:
            with open(file) as f:
                results = json.load(f)
                algo = results['algorithm']
                dataset = results['dataset']
                organized[dataset][algo] = results
        
        print(f"Generating comprehensive analysis")
        print(f"   Found {len(all_files)} result files")
        print(f"   Datasets: {list(organized.keys())}")
        
        # Create master comparison figure
        datasets = list(organized.keys())
        fig, axes = plt.subplots(len(datasets), 2, figsize=(14, 5 * len(datasets)))
        
        if len(datasets) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, dataset in enumerate(datasets):
            algorithms = organized[dataset]
            
            # Box plot comparison
            ax1 = axes[idx, 0]
            data_for_boxplot = []
            labels = []
            
            for algo, results in algorithms.items():
                best_fitness = [run['best_fitness'] for run in results['runs']]
                data_for_boxplot.append(best_fitness)
                labels.append(algo.upper())
            
            bp = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax1.set_ylabel('Best Validation Accuracy (%)')
            ax1.set_title(f'Algorithm Comparison on {dataset.upper()}')
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Bar chart with error bars
            ax2 = axes[idx, 1]
            means = []
            stds = []
            
            for algo, results in algorithms.items():
                best_fitness = [run['best_fitness'] for run in results['runs']]
                means.append(np.mean(best_fitness))
                stds.append(np.std(best_fitness))
            
            x_pos = np.arange(len(labels))
            bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                          color=colors, alpha=0.7, error_kw={'linewidth': 2})
            
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(labels)
            ax2.set_ylabel('Mean Best Accuracy (%)')
            ax2.set_title(f'Mean Performance on {dataset.upper()}')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        fig_path = self.figures_dir / "master_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Master figure saved: {fig_path}")
        
        plt.show()
        
        self._generate_summary_table(organized)
    
    def _print_comparison_stats(self, all_results, dataset):
        """Print statistical comparison"""
        print(f"\n{'='*60}")
        print(f"Statistical Comparison on {dataset.upper()}")
        print(f"{'='*60}")
        
        for algo, results in all_results.items():
            best_fitness = [run['best_fitness'] for run in results['runs']]
            print(f"\n{algo.upper()}:")
            print(f"  Best: {max(best_fitness):.2f}%")
            print(f"  Mean: {np.mean(best_fitness):.2f}% ± {np.std(best_fitness):.2f}%")
            print(f"  Worst: {min(best_fitness):.2f}%")
            print(f"  Runs: {len(best_fitness)}")
        
        # Determine winner
        means = {algo: np.mean([run['best_fitness'] for run in results['runs']]) 
                for algo, results in all_results.items()}
        winner = max(means, key=means.get)
        print(f"\n🏆 Best Algorithm: {winner.upper()} ({means[winner]:.2f}%)")
        print(f"{'='*60}\n")
    
    def _generate_summary_table(self, organized):
        """Generate summary table of all results"""
        print(f"\n{'='*80}")
        print(f"📋 COMPREHENSIVE RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for dataset, algorithms in organized.items():
            print(f"\n{dataset.upper()}:")
            print(f"{'Algorithm':<15} {'Best':<12} {'Mean±Std':<20} {'Runs':<8}")
            print("-" * 60)
            
            for algo, results in algorithms.items():
                best_fitness = [run['best_fitness'] for run in results['runs']]
                best = max(best_fitness)
                mean = np.mean(best_fitness)
                std = np.std(best_fitness)
                runs = len(best_fitness)
                
                print(f"{algo.upper():<15} {best:>6.2f}%      {mean:>6.2f}% ± {std:<6.2f}%  {runs:<8}")
        
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Plot experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single algorithm results
  python plot_results.py --algorithm grid --dataset mnist
  
  # Compare two algorithms
  python plot_results.py --compare grid random --dataset mnist
  
  # Generate all comparison figures
  python plot_results.py --all
        """
    )
    
    parser.add_argument('--algorithm', '-a',
                       choices=['grid', 'random', 'ga', 'de', 'pso'],
                       help='Plot specific algorithm')
    
    parser.add_argument('--dataset', '-d',
                       choices=['mnist', 'cifar10'],
                       help='Dataset to plot')
    
    parser.add_argument('--compare', nargs='+',
                       choices=['grid', 'random', 'ga', 'de', 'pso'],
                       help='Compare multiple algorithms')
    
    parser.add_argument('--all', action='store_true',
                       help='Generate comprehensive comparison of all results')
    
    args = parser.parse_args()
    
    plotter = SimplePlotter()
    
    if args.all:
        plotter.plot_all()
    elif args.compare:
        if not args.dataset:
            parser.error("--compare requires --dataset")
        plotter.plot_comparison(args.compare, args.dataset)
    elif args.algorithm:
        if not args.dataset:
            parser.error("--algorithm requires --dataset")
        plotter.plot_single(args.algorithm, args.dataset)
    else:
        parser.error("Specify --algorithm, --compare, or --all")


if __name__ == "__main__":
    main()
