"""Random Search implementation"""

import random
import numpy as np
from typing import Dict, Any, List, Union, Tuple


class RandomSearch:
    """Random Search optimizer"""
    
    def __init__(self, search_space: Dict[str, Any], max_evaluations: int = 20):
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        
    def optimize(self, eval_func):
        """Run random search optimization"""
        best_params = None
        best_fitness = 0.0
        history = []
        
        print(f"Random Search: Sampling {self.max_evaluations} random configurations")
        
        for eval_count in range(self.max_evaluations):
            hyperparams = self._sample_hyperparameters()
            
            print(f"   Evaluation {eval_count + 1}/{self.max_evaluations}...", end=" ")
            fitness = eval_func(hyperparams)
            print(f"{fitness:.2f}%")
            
            history.append({
                'evaluation': eval_count + 1,
                'hyperparameters': hyperparams.copy(),
                'fitness': fitness,
                'timestamp': eval_count
            })
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = hyperparams.copy()
                print(f"   New best: {best_fitness:.2f}%")
        
        print(f"Random Search complete: Best = {best_fitness:.2f}%")
        return best_params, best_fitness, history
    
    def _sample_hyperparameters(self) -> Dict[str, Any]:
        """Sample random configuration from search space"""
        hyperparams = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, list):
                hyperparams[param_name] = random.choice(param_config)
                
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                min_val, max_val, scale = param_config
                
                if scale == 'log':
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    hyperparams[param_name] = 10 ** random.uniform(log_min, log_max)
                else:
                    hyperparams[param_name] = random.uniform(min_val, max_val)
            else:
                hyperparams[param_name] = param_config
        
        return hyperparams
