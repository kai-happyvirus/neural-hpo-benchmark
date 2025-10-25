#!/usr/bin/env python3
"""Simple Hyperparameter Optimization Runner"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append('src')

from data_loader import DataManager
from trainer import ModelTrainer
from algorithms import GridSearch, RandomSearch, GeneticAlgorithm, DifferentialEvolution, ParticleSwarmOptimization


class HPOExperiment:
    """Hyperparameter Optimization Experiment"""
    
    def __init__(self):
        self.data_manager = DataManager({'data_dir': './data', 'hardware': {'num_workers': 2}})
        self.trainer = ModelTrainer()
        
    def run_experiment(self, algorithm, dataset, runs=3, max_evaluations=20):
        """Run experiment: one algorithm on one dataset"""
        print(f"\n{'='*60}")
        print(f"Running {algorithm.upper()} on {dataset.upper()}")
        print(f"{'='*60}\n")
        
        results = {
            'algorithm': algorithm,
            'dataset': dataset,
            'config': {
                'runs': runs,
                'max_evaluations': max_evaluations,
                'max_epochs': 50,
                'early_stopping': 10
            },
            'runs': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for run_num in range(1, runs + 1):
            print(f"\nRun {run_num}/{runs}")
            print("-" * 40)
            
            run_start = time.time()
            
            # Create evaluation function
            eval_func = self._create_evaluator(dataset)
            
            # Create optimizer
            optimizer = self._create_optimizer(algorithm, max_evaluations)
            
            # Run optimization
            best_params, best_fitness, history = optimizer.optimize(eval_func)
            
            run_time = time.time() - run_start
            
            # Store results
            run_results = {
                'run': run_num,
                'best_fitness': float(best_fitness),
                'best_hyperparameters': best_params,
                'evaluation_history': history,
                'total_evaluations': len(history),
                'time_seconds': run_time
            }
            
            results['runs'].append(run_results)
            
            print(f"    Run {run_num} complete!")
            print(f"    Best accuracy: {best_fitness:.2f}%")
            print(f"    Time: {run_time/60:.1f} minutes")
        
        # Save results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _create_evaluator(self, dataset):
        """Create evaluation function for the dataset"""
        
        def evaluate(hyperparams):
            """Evaluate hyperparameters and return validation accuracy"""
            batch_size = hyperparams.get('batch_size', 128)
            
            # Get data loaders
            train_loader, val_loader, test_loader = self.data_manager.get_dataloaders(
                dataset, batch_size
            )
            
            # Evaluate
            fitness = self.trainer.evaluate_hyperparameters(
                hyperparams, train_loader, val_loader, test_loader, dataset
            )
            
            return fitness
        
        return evaluate
    
    def _create_optimizer(self, algorithm, max_evaluations):
        """Create optimizer instance"""
        
        # Hyperparameter search space for evolutionary algorithms
        search_space = {
            'learning_rate': (0.0001, 0.01, 'log'),
            'batch_size': [64, 128, 256, 512],
            'dropout_rate': (0.0, 0.5, 'linear'),
            'hidden_units': [64, 128, 256, 512],
            'optimizer': ['adam', 'sgd', 'rmsprop'],
            'weight_decay': (0.0, 0.01, 'linear')
        }
        
        if algorithm == 'grid':
            # Simplified grid for faster execution
            grid_space = {
                'learning_rate': [0.0001, 0.001, 0.01],
                'batch_size': [128, 256],
                'dropout_rate': [0.0, 0.3],
                'hidden_units': [256, 512],
                'optimizer': ['adam', 'rmsprop'],
                'weight_decay': [0.0, 0.001]
            }
            return GridSearch(grid_space)
            
        elif algorithm == 'random':
            return RandomSearch(search_space, max_evaluations)
            
        elif algorithm == 'ga':
            return GeneticAlgorithm(
                search_space,
                population_size=6,
                generations=10,
                mutation_rate=0.1,
                crossover_rate=0.8
            )
            
        elif algorithm == 'de':
            return DifferentialEvolution(
                search_space,
                population_size=6,
                generations=10,
                F=0.8,
                CR=0.9
            )
            
        elif algorithm == 'pso':
            return ParticleSwarmOptimization(
                search_space,
                swarm_size=6,
                iterations=10,
                w=0.7,
                c1=1.5,
                c2=1.5
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _save_results(self, results):
        """Save results to simple JSON file"""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Simple filename: algorithm_dataset_timestamp.json
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{results['algorithm']}_{results['dataset']}_{timestamp}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def _print_summary(self, results):
        """Print experiment summary"""
        all_best = [run['best_fitness'] for run in results['runs']]
        all_times = [run['time_seconds'] for run in results['runs']]
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Algorithm: {results['algorithm'].upper()}")
        print(f"Dataset: {results['dataset'].upper()}")
        print(f"Runs: {len(results['runs'])}")
        print(f"    Performance:")
        print(f"    Best accuracy: {max(all_best):.2f}%")
        print(f"    Mean accuracy: {np.mean(all_best):.2f}% ± {np.std(all_best):.2f}%")
        print(f"    Worst accuracy: {min(all_best):.2f}%")
        print(f"    Time:")
        print(f"    Total: {sum(all_times)/60:.1f} minutes")
        print(f"    Per run: {np.mean(all_times)/60:.1f} minutes")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization Experiments')
    
    parser.add_argument('--algorithm', '-a', required=True,
                       choices=['grid', 'random', 'ga', 'de', 'pso'],
                       help='Optimization algorithm')
    
    parser.add_argument('--dataset', '-d', required=True,
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    
    parser.add_argument('--runs', '-r', type=int, default=3,
                       help='Number of independent runs')
    
    parser.add_argument('--evaluations', '-e', type=int, default=20,
                       help='Maximum evaluations per run')
    
    args = parser.parse_args()
    
    # Run experiment
    experiment = HPOExperiment()
    experiment.run_experiment(
        algorithm=args.algorithm,
        dataset=args.dataset,
        runs=args.runs,
        max_evaluations=args.evaluations
    )


if __name__ == "__main__":
    main()
