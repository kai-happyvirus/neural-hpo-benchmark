"""Grid Search implementation"""

import itertools
import numpy as np
from typing import Dict, Any, List


class GridSearch:
    """Grid Search optimizer"""
    
    def __init__(self, search_space: Dict[str, List], max_evaluations: int = 20):
        self.search_space = search_space
        self.max_evaluations = max_evaluations
        
    def optimize(self, eval_func):
        """Run grid search optimization"""
        best_params = None
        best_fitness = 0.0
        history = []
        
        param_names = list(self.search_space.keys())
        param_values = [self.search_space[name] for name in param_names]
        
        eval_count = 0
        print(f"Grid Search: Testing up to {self.max_evaluations} combinations")
        
        for combination in itertools.product(*param_values):
            if eval_count >= self.max_evaluations:
                break
                
            hyperparams = dict(zip(param_names, combination))
            
            print(f"Eval {eval_count + 1}: ", end="")
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
            
            eval_count += 1
        
        print(f"Grid Search complete: Best = {best_fitness:.2f}%")
        return best_params, best_fitness, history
