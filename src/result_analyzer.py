"""
Result generation and visualization system
Creates figures, tables, and analysis for experiment results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import os
from pathlib import Path
import json


class ResultAnalyzer:
    """Analyzes and visualizes experiment results"""
    
    def __init__(self, experiment_manager=None):
        self.experiment_manager = experiment_manager
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
    
    def create_convergence_plot(self, results: Dict[str, Any], title: str = "Convergence Plot") -> plt.Figure:
        """Create convergence plot for a single algorithm run"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Get fitness history
        if 'fitness_history' in results:
            history = results['fitness_history']
            generations = [h['generation'] for h in history]
            max_fitness = [h['max'] for h in history]
            avg_fitness = [h['avg'] for h in history]
            
            # Plot convergence
            ax1.plot(generations, max_fitness, label='Best Fitness', linewidth=2)
            ax1.plot(generations, avg_fitness, label='Average Fitness', linewidth=2, alpha=0.7)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness (Validation Accuracy %)')
            ax1.set_title(f'{title} - Convergence')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot fitness distribution (last generation)
            if 'std' in history[-1]:
                last_gen = history[-1]
                std = last_gen['std']
                mean = last_gen['avg']
                
                # Create histogram simulation
                x = np.linspace(mean - 3*std, mean + 3*std, 100)
                y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                ax2.plot(x, y, linewidth=2)
                ax2.axvline(last_gen['max'], color='red', linestyle='--', label=f"Best: {last_gen['max']:.2f}")
                ax2.axvline(mean, color='orange', linestyle='--', label=f"Mean: {mean:.2f}")
                ax2.set_xlabel('Fitness (Validation Accuracy %)')
                ax2.set_ylabel('Density')
                ax2.set_title('Final Generation Fitness Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        elif 'evaluation_history' in results:
            # For baseline methods
            history = results['evaluation_history']
            evaluations = [h['evaluation'] for h in history]
            fitness_values = [h['fitness'] for h in history]
            
            # Running best
            running_best = []
            best_so_far = 0
            for fitness in fitness_values:
                if fitness > best_so_far:
                    best_so_far = fitness
                running_best.append(best_so_far)
            
            ax1.plot(evaluations, running_best, label='Best Fitness Found', linewidth=2)
            ax1.scatter(evaluations, fitness_values, alpha=0.3, s=20, label='Individual Evaluations')
            ax1.set_xlabel('Evaluation')
            ax1.set_ylabel('Fitness (Validation Accuracy %)')
            ax1.set_title(f'{title} - Search Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Fitness histogram
            ax2.hist(fitness_values, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(fitness_values), color='red', linestyle='--', 
                       label=f"Mean: {np.mean(fitness_values):.2f}")
            ax2.axvline(np.max(fitness_values), color='orange', linestyle='--', 
                       label=f"Best: {np.max(fitness_values):.2f}")
            ax2.set_xlabel('Fitness (Validation Accuracy %)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Fitness Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, all_results: Dict[str, Dict[str, List[Dict[str, Any]]]], 
                              dataset: str) -> plt.Figure:
        """Create comparison plot for all algorithms on a dataset"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Collect data for comparison
        algorithm_data = {}
        
        for algorithm, datasets in all_results.items():
            if dataset in datasets:
                runs = datasets[dataset]
                best_fitnesses = [run.get('best_fitness', 0) for run in runs]
                total_times = [run.get('total_time', 0) for run in runs]
                evaluations = [run.get('total_evaluations', 0) for run in runs]
                
                algorithm_data[algorithm] = {
                    'best_fitness': best_fitnesses,
                    'total_time': total_times,
                    'evaluations': evaluations,
                    'mean_fitness': np.mean(best_fitnesses),
                    'std_fitness': np.std(best_fitnesses),
                    'mean_time': np.mean(total_times)
                }
        
        # Box plot of fitness values
        fitness_data = []
        labels = []
        for alg, data in algorithm_data.items():
            fitness_data.extend(data['best_fitness'])
            labels.extend([alg] * len(data['best_fitness']))
        
        df_fitness = pd.DataFrame({'Algorithm': labels, 'Best Fitness': fitness_data})
        sns.boxplot(data=df_fitness, x='Algorithm', y='Best Fitness', ax=ax1)
        ax1.set_title(f'Best Fitness Comparison - {dataset.upper()}')
        ax1.set_ylabel('Best Validation Accuracy (%)')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Bar plot of mean fitness with error bars
        algorithms = list(algorithm_data.keys())
        mean_fitness = [algorithm_data[alg]['mean_fitness'] for alg in algorithms]
        std_fitness = [algorithm_data[alg]['std_fitness'] for alg in algorithms]
        
        bars = ax2.bar(algorithms, mean_fitness, yerr=std_fitness, capsize=5, alpha=0.7)
        ax2.set_title(f'Mean Best Fitness ± Std - {dataset.upper()}')
        ax2.set_ylabel('Mean Best Validation Accuracy (%)')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Add value labels on bars
        for bar, mean_val in zip(bars, mean_fitness):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{mean_val:.1f}%', ha='center', va='bottom')
        
        # Scatter plot: Time vs Fitness
        for alg, data in algorithm_data.items():
            ax3.scatter(data['total_time'], data['best_fitness'], 
                       label=alg, s=60, alpha=0.7)
        
        ax3.set_xlabel('Total Time (seconds)')
        ax3.set_ylabel('Best Validation Accuracy (%)')
        ax3.set_title(f'Time vs Performance - {dataset.upper()}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Efficiency plot: Fitness per evaluation
        for alg, data in algorithm_data.items():
            efficiency = [f/e for f, e in zip(data['best_fitness'], data['evaluations']) if e > 0]
            ax4.scatter([alg] * len(efficiency), efficiency, s=60, alpha=0.7)
        
        ax4.set_ylabel('Fitness per Evaluation')
        ax4.set_title(f'Search Efficiency - {dataset.upper()}')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_hyperparameter_analysis(self, results: Dict[str, Any], 
                                     title: str = "Hyperparameter Analysis") -> plt.Figure:
        """Analyze best hyperparameters found"""
        best_hyperparams = results.get('best_hyperparameters', {})
        
        if not best_hyperparams:
            # Create empty figure
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, 'No hyperparameter data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Create subplots based on number of hyperparameters
        n_params = len(best_hyperparams)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        param_names = list(best_hyperparams.keys())
        
        for i, param_name in enumerate(param_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            param_value = best_hyperparams[param_name]
            
            if isinstance(param_value, (int, float)):
                # Numerical parameter - show as bar
                ax.bar([param_name], [param_value], alpha=0.7)
                ax.set_ylabel('Value')
                ax.text(0, param_value + param_value*0.05, f'{param_value:.4f}', 
                       ha='center', va='bottom')
            else:
                # Categorical parameter - show as text
                ax.text(0.5, 0.5, str(param_value), ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xticks([])
                ax.set_yticks([])
            
            ax.set_title(f'Best {param_name.replace("_", " ").title()}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            if n_rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig
    
    def create_summary_table(self, all_results: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> pd.DataFrame:
        """Create summary table of all results"""
        summary_data = []
        
        for algorithm, datasets in all_results.items():
            for dataset, runs in datasets.items():
                if not runs:
                    continue
                
                # Calculate statistics across runs
                best_fitnesses = [run.get('best_fitness', 0) for run in runs]
                total_times = [run.get('total_time', 0) for run in runs]
                evaluations = [run.get('total_evaluations', 0) for run in runs]
                
                row = {
                    'Algorithm': algorithm.upper(),
                    'Dataset': dataset.upper(),
                    'Runs': len(runs),
                    'Best Fitness (%)': f"{np.max(best_fitnesses):.2f}",
                    'Mean Fitness (%)': f"{np.mean(best_fitnesses):.2f} ± {np.std(best_fitnesses):.2f}",
                    'Mean Time (s)': f"{np.mean(total_times):.1f}",
                    'Mean Evaluations': f"{np.mean(evaluations):.0f}",
                    'Success Rate (%)': f"{np.mean([f > 90 for f in best_fitnesses]) * 100:.0f}"
                }
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def generate_all_figures(self, save_dir: Optional[str] = None):
        """Generate all analysis figures"""
        if self.experiment_manager is None:
            print("No experiment manager provided")
            return
        
        all_results = self.experiment_manager.get_all_results()
        
        if save_dir is None:
            save_dir = self.experiment_manager.experiment_dir / "figures"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Individual convergence plots
        for algorithm, datasets in all_results.items():
            for dataset, runs in datasets.items():
                for i, run in enumerate(runs):
                    title = f"{algorithm.upper()} - {dataset.upper()} (Run {i+1})"
                    fig = self.create_convergence_plot(run, title)
                    
                    filename = f"{algorithm}_{dataset}_run{i+1}_convergence"
                    fig.savefig(save_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
                    fig.savefig(save_dir / f"{filename}.pdf", bbox_inches='tight')
                    plt.close(fig)
        
        # Comparison plots for each dataset
        datasets = set(dataset for datasets in all_results.values() for dataset in datasets.keys())
        for dataset in datasets:
            fig = self.create_comparison_plot(all_results, dataset)
            
            filename = f"comparison_{dataset}"
            fig.savefig(save_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
            fig.savefig(save_dir / f"{filename}.pdf", bbox_inches='tight')
            plt.close(fig)
        
        # Hyperparameter analysis for best runs
        for algorithm, datasets in all_results.items():
            for dataset, runs in datasets.items():
                if runs:
                    best_run = max(runs, key=lambda r: r.get('best_fitness', 0))
                    title = f"Best Hyperparameters - {algorithm.upper()} - {dataset.upper()}"
                    fig = self.create_hyperparameter_analysis(best_run, title)
                    
                    filename = f"{algorithm}_{dataset}_best_hyperparams"
                    fig.savefig(save_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
                    fig.savefig(save_dir / f"{filename}.pdf", bbox_inches='tight')
                    plt.close(fig)
        
        # Summary table
        summary_df = self.create_summary_table(all_results)
        summary_df.to_csv(save_dir / "summary_table.csv", index=False)
        
        # Create summary table figure
        fig, ax = plt.subplots(figsize=(12, len(summary_df) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(summary_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.title('Experiment Results Summary', fontsize=16, fontweight='bold', pad=20)
        fig.savefig(save_dir / "summary_table.png", dpi=300, bbox_inches='tight')
        fig.savefig(save_dir / "summary_table.pdf", bbox_inches='tight')
        plt.close(fig)
        
        print(f"All figures generated and saved to: {save_dir}")


def create_light_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create light analysis for quick demonstration"""
    analysis = {
        'algorithm': results.get('algorithm', 'Unknown'),
        'best_fitness': results.get('best_fitness', 0),
        'total_time': results.get('total_time', 0),
        'converged': results.get('converged', False),
        'generations': len(results.get('fitness_history', [])),
        'improvement': 0
    }
    
    # Calculate improvement from first to last generation
    if 'fitness_history' in results and len(results['fitness_history']) > 1:
        first_gen = results['fitness_history'][0]['max']
        last_gen = results['fitness_history'][-1]['max']
        analysis['improvement'] = last_gen - first_gen
    
    return analysis


if __name__ == "__main__":
    # Test result analyzer
    print("Testing ResultAnalyzer...")
    
    # Create mock results
    mock_results = {
        'ga': {
            'mnist': [{
                'best_fitness': 94.5,
                'total_time': 120.5,
                'total_evaluations': 1000,
                'best_hyperparameters': {
                    'learning_rate': 0.001,
                    'batch_size': 64,
                    'dropout_rate': 0.2,
                    'optimizer': 'adam'
                },
                'fitness_history': [
                    {'generation': 0, 'max': 80.0, 'avg': 70.0, 'std': 5.0},
                    {'generation': 1, 'max': 85.0, 'avg': 75.0, 'std': 4.0},
                    {'generation': 2, 'max': 90.0, 'avg': 80.0, 'std': 3.0},
                    {'generation': 3, 'max': 94.5, 'avg': 85.0, 'std': 2.5}
                ]
            }]
        },
        'random': {
            'mnist': [{
                'best_fitness': 92.1,
                'total_time': 100.2,
                'total_evaluations': 1000,
                'evaluation_history': [
                    {'evaluation': i, 'fitness': 70 + np.random.random() * 25} 
                    for i in range(1, 101)
                ]
            }]
        }
    }
    
    analyzer = ResultAnalyzer()
    
    # Test convergence plot
    fig = analyzer.create_convergence_plot(mock_results['ga']['mnist'][0], "Test GA")
    fig.savefig("test_convergence.png")
    plt.close(fig)
    
    # Test comparison plot
    fig = analyzer.create_comparison_plot(mock_results, 'mnist')
    fig.savefig("test_comparison.png")
    plt.close(fig)
    
    # Test summary table
    summary = analyzer.create_summary_table(mock_results)
    print(summary)
    
    print("ResultAnalyzer test completed!")