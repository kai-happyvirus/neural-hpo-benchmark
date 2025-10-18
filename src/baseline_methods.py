"""
Baseline hyperparameter optimization methods: Grid Search and Random Search
"""

import itertools
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Generator
import time
import copy
import pickle
import os


class BaselineOptimizer:
    """Base class for baseline optimization methods"""
    
    def __init__(self, config: Dict[str, Any], evaluation_function: Callable):
        self.config = config
        self.evaluation_function = evaluation_function
        self.hyperparams_config = config.get('hyperparameters', {})
        
        # Results storage
        self.results = {
            'best_hyperparameters': None,
            'best_fitness': 0.0,
            'evaluation_history': [],
            'total_evaluations': 0,
            'total_time': 0.0
        }
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimization (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def save_checkpoint(self, filepath: str):
        """Save optimization checkpoint"""
        checkpoint_data = {
            'results': self.results,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, filepath: str):
        """Load optimization checkpoint"""
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.results = checkpoint_data['results']


class GridSearch(BaselineOptimizer):
    """Grid Search implementation"""
    
    def _generate_grid_points(self, max_evaluations: int) -> Generator[Dict[str, Any], None, None]:
        """Generate grid points for hyperparameters"""
        # Create parameter grids
        param_grids = {}
        
        for param_name, param_config in self.hyperparams_config.items():
            if param_name == 'learning_rate':
                # Log-spaced learning rates
                min_lr = param_config.get('min', 0.0001)
                max_lr = param_config.get('max', 0.1)
                n_points = min(10, int(max_evaluations ** (1/len(self.hyperparams_config))))
                param_grids[param_name] = np.logspace(np.log10(min_lr), np.log10(max_lr), n_points)
                
            elif param_name == 'dropout_rate':
                # Linear-spaced dropout rates
                min_dr = param_config.get('min', 0.0)
                max_dr = param_config.get('max', 0.5)
                n_points = min(6, int(max_evaluations ** (1/len(self.hyperparams_config))))
                param_grids[param_name] = np.linspace(min_dr, max_dr, n_points)
                
            elif param_name == 'weight_decay':
                # Linear-spaced weight decay
                min_wd = param_config.get('min', 0.0)
                max_wd = param_config.get('max', 0.01)
                n_points = min(5, int(max_evaluations ** (1/len(self.hyperparams_config))))
                param_grids[param_name] = np.linspace(min_wd, max_wd, n_points)
                
            elif isinstance(param_config, list):
                # Discrete choices
                param_grids[param_name] = param_config
            else:
                # Default: treat as discrete choices
                param_grids[param_name] = [param_config]
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        count = 0
        for combination in itertools.product(*param_values):
            if count >= max_evaluations:
                break
                
            hyperparams = dict(zip(param_names, combination))
            yield hyperparams
            count += 1
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Grid Search optimization"""
        max_evaluations = algorithm_params.get('max_evaluations', 1000)
        
        print(f"Starting Grid Search with up to {max_evaluations} evaluations...")
        
        start_time = time.time()
        evaluation_count = 0
        
        for hyperparams in self._generate_grid_points(max_evaluations):
            # Evaluate hyperparameters
            fitness = self.evaluation_function(hyperparams)
            evaluation_count += 1
            
            # Record evaluation
            evaluation_record = {
                'evaluation': evaluation_count,
                'hyperparameters': copy.deepcopy(hyperparams),
                'fitness': fitness,
                'timestamp': time.time() - start_time
            }
            self.results['evaluation_history'].append(evaluation_record)
            
            # Update best if necessary
            if fitness > self.results['best_fitness']:
                self.results['best_fitness'] = fitness
                self.results['best_hyperparameters'] = copy.deepcopy(hyperparams)
            
            # Progress reporting
            if evaluation_count % 50 == 0:
                print(f"Evaluated {evaluation_count}/{max_evaluations} configurations. "
                      f"Best fitness: {self.results['best_fitness']:.4f}")
            
            # Save checkpoint every 100 evaluations
            if evaluation_count % 100 == 0:
                checkpoint_path = f"checkpoints/grid_search_checkpoint_{evaluation_count}.pkl"
                self.save_checkpoint(checkpoint_path)
        
        self.results['total_evaluations'] = evaluation_count
        self.results['total_time'] = time.time() - start_time
        self.results['algorithm'] = 'Grid Search'
        
        print(f"Grid Search completed. Total evaluations: {evaluation_count}, "
              f"Best fitness: {self.results['best_fitness']:.4f}")
        
        return self.results


class RandomSearch(BaselineOptimizer):
    """Random Search implementation"""
    
    def _sample_random_hyperparams(self) -> Dict[str, Any]:
        """Sample random hyperparameters"""
        hyperparams = {}
        
        for param_name, param_config in self.hyperparams_config.items():
            if param_name == 'learning_rate':
                # Log-uniform sampling
                min_lr = param_config.get('min', 0.0001)
                max_lr = param_config.get('max', 0.1)
                log_min, log_max = np.log10(min_lr), np.log10(max_lr)
                log_lr = random.uniform(log_min, log_max)
                hyperparams[param_name] = 10 ** log_lr
                
            elif param_name in ['dropout_rate', 'weight_decay']:
                # Uniform sampling
                min_val = param_config.get('min', 0.0)
                max_val = param_config.get('max', 0.5 if param_name == 'dropout_rate' else 0.01)
                hyperparams[param_name] = random.uniform(min_val, max_val)
                
            elif isinstance(param_config, list):
                # Random choice from discrete options
                hyperparams[param_name] = random.choice(param_config)
            else:
                # Default: use as-is
                hyperparams[param_name] = param_config
        
        return hyperparams
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Random Search optimization"""
        max_evaluations = algorithm_params.get('max_evaluations', 1000)
        
        print(f"Starting Random Search with {max_evaluations} evaluations...")
        
        start_time = time.time()
        
        for evaluation_count in range(1, max_evaluations + 1):
            # Sample random hyperparameters
            hyperparams = self._sample_random_hyperparams()
            
            # Evaluate hyperparameters
            fitness = self.evaluation_function(hyperparams)
            
            # Record evaluation
            evaluation_record = {
                'evaluation': evaluation_count,
                'hyperparameters': copy.deepcopy(hyperparams),
                'fitness': fitness,
                'timestamp': time.time() - start_time
            }
            self.results['evaluation_history'].append(evaluation_record)
            
            # Update best if necessary
            if fitness > self.results['best_fitness']:
                self.results['best_fitness'] = fitness
                self.results['best_hyperparameters'] = copy.deepcopy(hyperparams)
            
            # Progress reporting
            if evaluation_count % 50 == 0:
                print(f"Evaluated {evaluation_count}/{max_evaluations} configurations. "
                      f"Best fitness: {self.results['best_fitness']:.4f}")
            
            # Save checkpoint every 100 evaluations
            if evaluation_count % 100 == 0:
                checkpoint_path = f"checkpoints/random_search_checkpoint_{evaluation_count}.pkl"
                self.save_checkpoint(checkpoint_path)
        
        self.results['total_evaluations'] = max_evaluations
        self.results['total_time'] = time.time() - start_time
        self.results['algorithm'] = 'Random Search'
        
        print(f"Random Search completed. Total evaluations: {max_evaluations}, "
              f"Best fitness: {self.results['best_fitness']:.4f}")
        
        return self.results


class AdaptiveRandomSearch(BaselineOptimizer):
    """Adaptive Random Search that focuses on promising regions"""
    
    def __init__(self, config: Dict[str, Any], evaluation_function: Callable):
        super().__init__(config, evaluation_function)
        self.elite_hyperparams = []
        self.elite_fitness = []
        self.adaptation_interval = 50  # Adapt every 50 evaluations
        
    def _adaptive_sample(self) -> Dict[str, Any]:
        """Sample hyperparameters with bias towards good regions"""
        if len(self.elite_hyperparams) < 5:
            # Not enough data, use pure random sampling
            return self._sample_random_hyperparams()
        
        # Choose whether to sample near elite or randomly
        if random.random() < 0.7:  # 70% chance to sample near elite
            # Select random elite hyperparameter
            elite_idx = random.randint(0, len(self.elite_hyperparams) - 1)
            base_hyperparams = self.elite_hyperparams[elite_idx]
            
            # Add noise to create variation
            noisy_hyperparams = {}
            for param_name, value in base_hyperparams.items():
                param_config = self.hyperparams_config[param_name]
                
                if param_name == 'learning_rate':
                    # Log-space noise
                    log_value = np.log10(value)
                    noise = np.random.normal(0, 0.2)  # Small noise in log space
                    new_log_value = log_value + noise
                    
                    min_lr = param_config.get('min', 0.0001)
                    max_lr = param_config.get('max', 0.1)
                    new_log_value = np.clip(new_log_value, np.log10(min_lr), np.log10(max_lr))
                    noisy_hyperparams[param_name] = 10 ** new_log_value
                    
                elif param_name in ['dropout_rate', 'weight_decay']:
                    # Linear space noise
                    noise = np.random.normal(0, 0.05)  # Small noise
                    new_value = value + noise
                    
                    min_val = param_config.get('min', 0.0)
                    max_val = param_config.get('max', 0.5 if param_name == 'dropout_rate' else 0.01)
                    noisy_hyperparams[param_name] = np.clip(new_value, min_val, max_val)
                    
                elif isinstance(param_config, list):
                    # For discrete parameters, sometimes keep same, sometimes random
                    if random.random() < 0.8:
                        noisy_hyperparams[param_name] = value
                    else:
                        noisy_hyperparams[param_name] = random.choice(param_config)
                else:
                    noisy_hyperparams[param_name] = value
                    
            return noisy_hyperparams
        else:
            # Pure random sampling
            return self._sample_random_hyperparams()
    
    def _sample_random_hyperparams(self) -> Dict[str, Any]:
        """Sample random hyperparameters (same as RandomSearch)"""
        hyperparams = {}
        
        for param_name, param_config in self.hyperparams_config.items():
            if param_name == 'learning_rate':
                min_lr = param_config.get('min', 0.0001)
                max_lr = param_config.get('max', 0.1)
                log_min, log_max = np.log10(min_lr), np.log10(max_lr)
                log_lr = random.uniform(log_min, log_max)
                hyperparams[param_name] = 10 ** log_lr
                
            elif param_name in ['dropout_rate', 'weight_decay']:
                min_val = param_config.get('min', 0.0)
                max_val = param_config.get('max', 0.5 if param_name == 'dropout_rate' else 0.01)
                hyperparams[param_name] = random.uniform(min_val, max_val)
                
            elif isinstance(param_config, list):
                hyperparams[param_name] = random.choice(param_config)
            else:
                hyperparams[param_name] = param_config
        
        return hyperparams
    
    def _update_elite(self, hyperparams: Dict[str, Any], fitness: float):
        """Update elite hyperparameters"""
        self.elite_hyperparams.append(copy.deepcopy(hyperparams))
        self.elite_fitness.append(fitness)
        
        # Keep only top 20 elite configurations
        if len(self.elite_hyperparams) > 20:
            # Sort by fitness and keep top 20
            sorted_indices = sorted(range(len(self.elite_fitness)), 
                                  key=lambda i: self.elite_fitness[i], reverse=True)
            
            self.elite_hyperparams = [self.elite_hyperparams[i] for i in sorted_indices[:20]]
            self.elite_fitness = [self.elite_fitness[i] for i in sorted_indices[:20]]
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Adaptive Random Search optimization"""
        max_evaluations = algorithm_params.get('max_evaluations', 1000)
        
        print(f"Starting Adaptive Random Search with {max_evaluations} evaluations...")
        
        start_time = time.time()
        
        for evaluation_count in range(1, max_evaluations + 1):
            # Sample hyperparameters (adaptively)
            hyperparams = self._adaptive_sample()
            
            # Evaluate hyperparameters
            fitness = self.evaluation_function(hyperparams)
            
            # Record evaluation
            evaluation_record = {
                'evaluation': evaluation_count,
                'hyperparameters': copy.deepcopy(hyperparams),
                'fitness': fitness,
                'timestamp': time.time() - start_time
            }
            self.results['evaluation_history'].append(evaluation_record)
            
            # Update best if necessary
            if fitness > self.results['best_fitness']:
                self.results['best_fitness'] = fitness
                self.results['best_hyperparameters'] = copy.deepcopy(hyperparams)
            
            # Update elite configurations (if performance is decent)
            if fitness > np.percentile([r['fitness'] for r in self.results['evaluation_history']], 75):
                self._update_elite(hyperparams, fitness)
            
            # Progress reporting
            if evaluation_count % 50 == 0:
                print(f"Evaluated {evaluation_count}/{max_evaluations} configurations. "
                      f"Best fitness: {self.results['best_fitness']:.4f}, "
                      f"Elite size: {len(self.elite_hyperparams)}")
            
            # Save checkpoint every 100 evaluations
            if evaluation_count % 100 == 0:
                checkpoint_path = f"checkpoints/adaptive_random_search_checkpoint_{evaluation_count}.pkl"
                self.save_checkpoint(checkpoint_path)
        
        self.results['total_evaluations'] = max_evaluations
        self.results['total_time'] = time.time() - start_time
        self.results['algorithm'] = 'Adaptive Random Search'
        
        print(f"Adaptive Random Search completed. Total evaluations: {max_evaluations}, "
              f"Best fitness: {self.results['best_fitness']:.4f}")
        
        return self.results


def create_baseline_optimizer(algorithm: str, config: Dict[str, Any], 
                             evaluation_function: Callable) -> BaselineOptimizer:
    """Factory function to create baseline optimizers"""
    algorithm = algorithm.lower()
    
    if algorithm in ['grid', 'grid_search']:
        return GridSearch(config, evaluation_function)
    elif algorithm in ['random', 'random_search']:
        return RandomSearch(config, evaluation_function)
    elif algorithm in ['adaptive_random', 'adaptive']:
        return AdaptiveRandomSearch(config, evaluation_function)
    else:
        raise ValueError(f"Unsupported baseline algorithm: {algorithm}")


if __name__ == "__main__":
    # Test baseline methods
    print("Testing baseline optimization methods...")
    
    # Mock configuration
    config = {
        'hyperparameters': {
            'learning_rate': {'min': 0.001, 'max': 0.1, 'log_scale': True},
            'batch_size': [32, 64, 128],
            'dropout_rate': {'min': 0.0, 'max': 0.5},
            'hidden_units': [64, 128, 256],
            'optimizer': ['adam', 'sgd'],
            'weight_decay': {'min': 0.0, 'max': 0.01}
        }
    }
    
    # Mock evaluation function
    def mock_evaluation(hyperparams):
        # Mock function that prefers certain hyperparameters
        score = 50 + random.random() * 30  # Base score 50-80
        
        # Bonus for certain choices
        if hyperparams.get('optimizer') == 'adam':
            score += 10
        if 0.001 <= hyperparams.get('learning_rate', 0) <= 0.01:
            score += 5
        if hyperparams.get('batch_size') == 64:
            score += 3
            
        return score + random.random() * 5  # Add some noise
    
    # Test Grid Search
    print("\nTesting Grid Search...")
    grid_search = GridSearch(config, mock_evaluation)
    grid_results = grid_search.optimize({'max_evaluations': 50})
    print(f"Grid Search best fitness: {grid_results['best_fitness']:.2f}")
    
    # Test Random Search
    print("\nTesting Random Search...")
    random_search = RandomSearch(config, mock_evaluation)
    random_results = random_search.optimize({'max_evaluations': 50})
    print(f"Random Search best fitness: {random_results['best_fitness']:.2f}")
    
    print("Baseline methods test completed!")