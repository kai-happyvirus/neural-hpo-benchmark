"""Differential Evolution implementation

References:
    Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic 
    for global optimization over continuous spaces. Journal of Global Optimization, 11(4), 341-359.
    https://doi.org/10.1023/A:1008202821328
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple, Callable


class DifferentialEvolution:
    """Differential Evolution optimizer
    
    Implementation Guide:
        Brownlee, J. (2021). Differential Evolution from Scratch in Python.
        Machine Learning Mastery.
        https://machinelearningmastery.com/differential-evolution-from-scratch-in-python/
    """
    
    def __init__(self, search_space: Dict[str, Any],
                 population_size: int = 10,
                 generations: int = 20,
                 F: float = 0.8,
                 CR: float = 0.9):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        
    def optimize(self, eval_func: Callable) -> Tuple[Dict, float, List]:
        """Run differential evolution optimization"""
        population = [self._random_individual() for _ in range(self.population_size)]
        fitnesses = [eval_func(ind) for ind in population]
        
        best_params = None
        best_fitness = 0.0
        history = []
        
        print(f"DE: {self.population_size} pop x {self.generations} gen")
        
        for gen in range(self.generations):
            new_population = []
            new_fitnesses = []
            
            for i in range(self.population_size):
                mutant = self._mutate(population, i)
                trial = self._crossover(population[i], mutant)
                trial_fitness = eval_func(trial)
                
                if trial_fitness > fitnesses[i]:
                    new_population.append(trial)
                    new_fitnesses.append(trial_fitness)
                else:
                    new_population.append(population[i])
                    new_fitnesses.append(fitnesses[i])
            
            population = new_population
            fitnesses = new_fitnesses
            
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_params = population[gen_best_idx].copy()
                print(f"Gen {gen+1}: New best = {best_fitness:.2f}%")
            
            history.append({
                'generation': gen + 1,
                'best_fitness': gen_best_fitness,
                'avg_fitness': np.mean(fitnesses)
            })
        
        print(f"DE complete: Best = {best_fitness:.2f}%")
        
        # Build evaluation history with per-generation best fitness
        eval_history = [{
            'evaluation': i + 1,
            'hyperparameters': best_params if i == len(history) - 1 else None,
            'fitness': history[i]['best_fitness'],
            'timestamp': i
        } for i in range(len(history))]
        
        return best_params, best_fitness, eval_history
    
    def _random_individual(self) -> Dict[str, Any]:
        """Generate random hyperparameters"""
        individual = {}
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, list):
                individual[param_name] = random.choice(param_config)
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                min_val, max_val, scale = param_config
                if scale == 'log':
                    individual[param_name] = 10 ** random.uniform(np.log10(min_val), np.log10(max_val))
                else:
                    individual[param_name] = random.uniform(min_val, max_val)
        return individual
    
    def _mutate(self, population: List[Dict], current_idx: int) -> Dict:
        """Create mutant vector using DE/rand/1 strategy
        
        Implements mutation: v_i = x_r1 + F * (x_r2 - x_r3)
        where r1, r2, r3 are random distinct indices ≠ i
        
        Reference: Storn & Price (1997), Equation 4
        """
        indices = [i for i in range(len(population)) if i != current_idx]
        a, b, c = random.sample(indices, 3)
        
        mutant = {}
        for param in population[0].keys():
            if param.startswith('_'):
                continue
            param_config = self.search_space[param]
            
            if isinstance(param_config, list):
                mutant[param] = random.choice([population[a][param], 
                                              population[b][param], 
                                              population[c][param]])
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                min_val, max_val, scale = param_config
                
                val_a = population[a][param]
                val_b = population[b][param]
                val_c = population[c][param]
                
                if scale == 'log':
                    log_a = np.log10(val_a)
                    log_b = np.log10(val_b)
                    log_c = np.log10(val_c)
                    log_mutant = log_a + self.F * (log_b - log_c)
                    log_mutant = np.clip(log_mutant, np.log10(min_val), np.log10(max_val))
                    mutant[param] = 10 ** log_mutant
                else:
                    mutant[param] = val_a + self.F * (val_b - val_c)
                    mutant[param] = np.clip(mutant[param], min_val, max_val)
        
        return mutant
    
    def _crossover(self, target: Dict, mutant: Dict) -> Dict:
        """Binomial crossover between target and mutant
        
        Implements crossover: u_i,j = v_i,j if rand() < CR or j = j_rand, else x_i,j
        
        Reference: Storn & Price (1997), Equation 5
        """
        trial = {}
        params = list(target.keys())
        j_rand = random.randint(0, len(params) - 1)
        
        for j, param in enumerate(params):
            if random.random() < self.CR or j == j_rand:
                trial[param] = mutant[param]
            else:
                trial[param] = target[param]
        
        return trial
