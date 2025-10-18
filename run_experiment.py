"""
Main execution script for hyperparameter optimization experiments
Supports three modes: full run, specific algorithm, and light run for video demonstration
"""

import sys
import os
import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import our modules
from data_loader import DataManager
from trainer import ModelTrainer
from evolutionary_algorithms import create_optimizer
from baseline_methods import create_baseline_optimizer
from experiment_manager import ExperimentManager
from result_analyzer import ResultAnalyzer, create_light_analysis
import yaml
import multiprocessing as mp


class ExperimentRunner:
    """Main experiment runner class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Initialize components
        self.data_manager = DataManager(self.config)
        self.trainer = ModelTrainer(device=self.config.get('hardware', {}).get('device', 'auto'))
        
        # Set random seeds for reproducibility
        self.set_random_seeds(self.config.get('random_seed', 42))
        
        print(f"Experiment runner initialized with config: {config_path}")
        
        # Initialize progress tracking
        self.start_time = None
        self.total_experiments = 0
        self.completed_experiments = 0
    
    def _format_time(self, seconds: float) -> str:
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
    
    def _print_progress_header(self, algorithm: str, dataset: str, run_id: int):
        """Print detailed progress header with timing info."""
        current_time = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print(f"\n{'='*70}")
        print(f"🚀 ALGORITHM: {algorithm.upper()} | DATASET: {dataset.upper()} | RUN: {run_id}")
        print(f"⏰ Time: {current_time} | Elapsed: {self._format_time(elapsed)}")
        print(f"📊 Progress: {self.completed_experiments}/{self.total_experiments} experiments")
        print(f"{'='*70}")
    
    def _print_status_update(self, message: str, level: str = "info"):
        """Print status update with timestamp."""
        current_time = datetime.now().strftime("%H:%M:%S")
        icons = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}
        icon = icons.get(level, "ℹ️")
        print(f"{icon} [{current_time}] {message}")
    
    def _simulate_progress_bar(self, task_name: str, duration: float, steps: int = 20):
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
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            print("Using default configuration...")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config file is not available"""
        return {
            'hardware': {'device': 'auto', 'num_workers': 4},
            'datasets': {
                'mnist': {'name': 'MNIST', 'batch_sizes': [32, 64, 128]},
                'cifar10': {'name': 'CIFAR-10', 'batch_sizes': [32, 64, 128]}
            },
            'hyperparameters': {
                'learning_rate': {'min': 0.0001, 'max': 0.1, 'log_scale': True},
                'batch_size': [32, 64, 128],
                'dropout_rate': {'min': 0.0, 'max': 0.5},
                'hidden_units': [64, 128, 256],
                'optimizer': ['adam', 'sgd'],
                'weight_decay': {'min': 0.0, 'max': 0.01}
            },
            'algorithms': {
                'genetic_algorithm': {'population_size': 20, 'generations': 50},
                'differential_evolution': {'population_size': 20, 'generations': 50},
                'particle_swarm': {'population_size': 20, 'generations': 50},
                'grid_search': {'max_evaluations': 1000},
                'random_search': {'max_evaluations': 1000}
            },
            'training': {'max_epochs': 50, 'early_stopping_patience': 10, 'validation_split': 0.2},
            'execution_modes': {
                'full_run': {'algorithms': ['ga', 'de', 'pso', 'grid', 'random'], 
                           'datasets': ['mnist', 'cifar10'], 'runs_per_algorithm': 3},
                'light_run': {'algorithms': ['ga', 'de', 'pso'], 'datasets': ['mnist'], 
                            'runs_per_algorithm': 1, 'max_generations': 10, 'population_size': 10}
            },
            'random_seed': 42
        }
    
    def set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    
    def create_evaluation_function(self, dataset: str, batch_size: int):
        """Create evaluation function for hyperparameter optimization"""
        train_loader, val_loader, test_loader = self.data_manager.get_dataloaders(
            dataset, batch_size, self.config['training']['validation_split']
        )
        
        def evaluate_hyperparams(hyperparams: Dict[str, Any]) -> float:
            """Evaluate hyperparameters and return fitness (validation accuracy)"""
            # Merge with training config
            full_hyperparams = {**self.config['training'], **hyperparams}
            
            return self.trainer.evaluate_hyperparameters(
                full_hyperparams, train_loader, val_loader, test_loader, dataset
            )
        
        return evaluate_hyperparams
    
    def run_single_algorithm(self, algorithm: str, dataset: str, run_id: int,
                           experiment_manager: ExperimentManager,
                           algorithm_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run a single algorithm on a dataset"""
        self._print_progress_header(algorithm, dataset, run_id)
        
        # Get dataset configuration
        dataset_config = self.config['datasets'][dataset]
        batch_size = dataset_config['batch_sizes'][0]  # Use first batch size
        
        # Create evaluation function
        evaluate_func = self.create_evaluation_function(dataset, batch_size)
        
        # Get algorithm parameters
        if algorithm_params is None:
            algorithm_params = self.config['algorithms'].get(
                f"{algorithm}_algorithm" if algorithm in ['genetic', 'differential', 'particle'] 
                else f"{algorithm}_search", {}
            )
        
        try:
            start_time = time.time()
            
            # Create optimizer with status updates
            self._print_status_update(f"Initializing {algorithm.upper()} optimizer...")
            
            if algorithm in ['ga', 'genetic']:
                optimizer = create_optimizer('ga', self.config, evaluate_func)
            elif algorithm in ['de', 'differential']:
                optimizer = create_optimizer('de', self.config, evaluate_func)
            elif algorithm in ['pso', 'particle']:
                optimizer = create_optimizer('pso', self.config, evaluate_func)
            elif algorithm in ['grid', 'grid_search']:
                optimizer = create_baseline_optimizer('grid', self.config, evaluate_func)
            elif algorithm in ['random', 'random_search']:
                optimizer = create_baseline_optimizer('random', self.config, evaluate_func)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            self._print_status_update(f"Starting optimization process...")
            
            # Run optimization
            results = optimizer.optimize(algorithm_params)
            
            # Add runtime information
            results['total_time'] = time.time() - start_time
            results['dataset'] = dataset
            results['algorithm'] = algorithm
            results['run_id'] = run_id
            
            # Save results
            run_name = f"run_{run_id:02d}"
            experiment_manager.save_results(algorithm, dataset, results, run_name)
            
            # Update progress tracking
            self.completed_experiments += 1
            
            # Print completion status
            self._print_status_update(
                f"{algorithm.upper()} completed! Best: {results.get('best_fitness', 0):.2f}% "
                f"in {results['total_time']:.1f}s", "success"
            )
            
            return results
            
        except Exception as e:
            print(f"✗ Error running {algorithm}: {e}")
            traceback.print_exc()
            return {'error': str(e), 'algorithm': algorithm, 'dataset': dataset, 'run_id': run_id}
    
    def run_full_experiment(self, experiment_name: Optional[str] = None) -> str:
        """Run full experiment with all algorithms and datasets"""
        # Create experiment manager
        experiment_manager = ExperimentManager(experiment_name=experiment_name)
        experiment_manager.save_config(self.config)
        
        # Get execution configuration
        exec_config = self.config['execution_modes']['full_run']
        algorithms = exec_config['algorithms']
        datasets = exec_config['datasets']
        runs_per_algorithm = exec_config['runs_per_algorithm']
        
        print(f"Starting full experiment: {experiment_manager.experiment_name}")
        print(f"Algorithms: {algorithms}")
        print(f"Datasets: {datasets}")
        print(f"Runs per algorithm: {runs_per_algorithm}")
        
        total_runs = len(algorithms) * len(datasets) * runs_per_algorithm
        current_run = 0
        
        experiment_start_time = time.time()
        
        # Run all combinations
        for algorithm in algorithms:
            for dataset in datasets:
                for run_id in range(1, runs_per_algorithm + 1):
                    current_run += 1
                    print(f"\\nProgress: {current_run}/{total_runs}")
                    
                    result = self.run_single_algorithm(
                        algorithm, dataset, run_id, experiment_manager
                    )
                    
                    # Update progress
                    experiment_manager.update_metadata(
                        status='running',
                        current_run=current_run,
                        total_runs=total_runs,
                        progress_percent=(current_run / total_runs) * 100
                    )
        
        # Generate analysis
        print(f"\\n{'='*60}")
        print("Generating analysis and figures...")
        print(f"{'='*60}")
        
        analyzer = ResultAnalyzer(experiment_manager)
        analyzer.generate_all_figures()
        
        # Export results
        experiment_manager.export_results_csv()
        
        # Final summary
        total_time = time.time() - experiment_start_time
        experiment_manager.update_metadata(
            status='completed',
            total_experiment_time=total_time,
            completion_time=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        print(f"\\n{'='*60}")
        print("EXPERIMENT COMPLETED!")
        print(f"{'='*60}")
        print(f"Experiment directory: {experiment_manager.experiment_dir}")
        print(f"Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
        print(f"Total runs: {total_runs}")
        
        return str(experiment_manager.experiment_dir)
    
    def run_light_experiment(self, experiment_name: Optional[str] = None) -> str:
        """Run light experiment for video demonstration"""
        # Initialize progress tracking
        self.start_time = time.time()
        
        # Create experiment manager
        if experiment_name is None:
            experiment_name = "light_demo"
        experiment_manager = ExperimentManager(experiment_name=experiment_name)
        
        # Get light execution configuration
        exec_config = self.config['execution_modes']['light_run']
        algorithms = exec_config['algorithms']
        datasets = exec_config['datasets']
        
        # Calculate total experiments for progress tracking
        self.total_experiments = len(algorithms) * len(datasets)
        self.completed_experiments = 0
        
        # Override algorithm parameters for quick execution
        light_params = {
            'population_size': exec_config.get('population_size', 10),
            'generations': exec_config.get('max_generations', 10),
            'max_evaluations': 100
        }
        
        # Override training parameters
        self.config['training']['max_epochs'] = exec_config.get('max_epochs', 10)
        
        print(f"\n{'🎬'*20}")
        print(f"🎥 STARTING VIDEO DEMO EXPERIMENT")
        print(f"{'🎬'*20}")
        print(f"📊 Total experiments: {self.total_experiments}")
        print(f"🧬 Algorithms: {', '.join(algorithms)}")
        print(f"📁 Datasets: {', '.join(datasets)}")
        print(f"⚡ Quick parameters: {light_params}")
        print(f"{'🎬'*20}\n")
        
        experiment_start_time = time.time()
        all_analyses = {}
        
        # Run each algorithm once with detailed progress
        for i, algorithm in enumerate(algorithms):
            for j, dataset in enumerate(datasets):
                current_experiment = i * len(datasets) + j + 1
                
                self._print_status_update(
                    f"Starting experiment {current_experiment}/{self.total_experiments}: "
                    f"{algorithm.upper()} on {dataset.upper()}", "info"
                )
                
                result = self.run_single_algorithm(
                    algorithm, dataset, 1, experiment_manager, light_params
                )
                
                # Create quick analysis
                analysis = create_light_analysis(result)
                all_analyses[f"{algorithm}_{dataset}"] = analysis
                
                # Show experiment summary
                progress_percent = (current_experiment / self.total_experiments) * 100
                self._print_status_update(
                    f"Experiment {current_experiment}/{self.total_experiments} complete "
                    f"({progress_percent:.0f}%): {analysis['best_fitness']:.1f}% accuracy "
                    f"in {analysis['total_time']:.1f}s", "success"
                )
        
        total_time = time.time() - experiment_start_time
        
        print(f"\\n{'='*60}")
        print("LIGHT EXPERIMENT COMPLETED!")
        print(f"{'='*60}")
        print(f"Total demonstration time: {total_time:.1f}s")
        
        # Print summary
        for name, analysis in all_analyses.items():
            print(f"{name}: {analysis['best_fitness']:.1f}% in {analysis['total_time']:.1f}s")
        
        return str(experiment_manager.experiment_dir)
    
    def run_specific_algorithm(self, algorithm: str, dataset: str = 'mnist', 
                             runs: int = 1, experiment_name: Optional[str] = None) -> str:
        """Run specific algorithm"""
        # Create experiment manager
        if experiment_name is None:
            experiment_name = f"{algorithm}_{dataset}_specific"
        experiment_manager = ExperimentManager(experiment_name=experiment_name)
        experiment_manager.save_config(self.config)
        
        print(f"Running {algorithm.upper()} on {dataset.upper()} ({runs} runs)")
        
        experiment_start_time = time.time()
        
        # Run multiple times if requested
        for run_id in range(1, runs + 1):
            result = self.run_single_algorithm(algorithm, dataset, run_id, experiment_manager)
        
        # Generate analysis
        analyzer = ResultAnalyzer(experiment_manager)
        analyzer.generate_all_figures()
        experiment_manager.export_results_csv()
        
        total_time = time.time() - experiment_start_time
        
        print(f"\\nSpecific algorithm run completed in {total_time:.1f}s")
        return str(experiment_manager.experiment_dir)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization Experiments')
    parser.add_argument('mode', choices=['full', 'light', 'specific'], 
                       help='Execution mode')
    parser.add_argument('--algorithm', type=str, 
                       choices=['ga', 'de', 'pso', 'grid', 'random'],
                       help='Specific algorithm to run (for specific mode)')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10'], 
                       default='mnist', help='Dataset to use')
    parser.add_argument('--runs', type=int, default=1, 
                       help='Number of runs (for specific mode)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--name', type=str, 
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Create experiment runner
    try:
        runner = ExperimentRunner(args.config)
    except Exception as e:
        print(f"Error initializing experiment runner: {e}")
        print("Make sure you have installed all dependencies:")
        print("pip install -r requirements.txt")
        return
    
    # Run based on mode
    try:
        if args.mode == 'full':
            result_dir = runner.run_full_experiment(args.name)
        elif args.mode == 'light':
            result_dir = runner.run_light_experiment(args.name)
        elif args.mode == 'specific':
            if args.algorithm is None:
                print("Error: --algorithm is required for specific mode")
                return
            result_dir = runner.run_specific_algorithm(
                args.algorithm, args.dataset, args.runs, args.name
            )
        
        print(f"\\nResults saved to: {result_dir}")
        
    except KeyboardInterrupt:
        print("\\nExperiment interrupted by user")
    except Exception as e:
        print(f"\\nError running experiment: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()