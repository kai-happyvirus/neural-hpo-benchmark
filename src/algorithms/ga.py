"""Genetic Algorithm implementation"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple, Callable


class GeneticAlgorithm:
    """Genetic Algorithm optimizer"""
    
    def __init__(self, search_space: Dict[str, Any], 
                 population_size: int = 10,
                 generations: int = 20,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8):
        self.search_space = search_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
    def optimize(self, eval_func: Callable) -> Tuple[Dict, float, List]:
        """Run genetic algorithm optimization"""
        population = [self._random_individual() for _ in range(self.population_size)]
        fitnesses = [eval_func(ind) for ind in population]
        
        best_params = None
        best_fitness = 0.0
        history = []
        
        print(f"Genetic Algorithm: {self.population_size} pop × {self.generations} gen")
        
        for gen in range(self.generations):
            offspring = []
            for _ in range(self.population_size):
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                offspring.append(child)
            
            offspring_fitnesses = [eval_func(ind) for ind in offspring]
            
            population = offspring
            fitnesses = offspring_fitnesses
            
            gen_best_idx = np.argmax(fitnesses)
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_params = population[gen_best_idx].copy()
                print(f"   Gen {gen+1}/{self.generations}: New best = {best_fitness:.2f}%")
            else:
                print(f"   Gen {gen+1}/{self.generations}: Best = {best_fitness:.2f}%")
            
            history.append({
                'generation': gen + 1,
                'best_fitness': gen_best_fitness,
                'avg_fitness': np.mean(fitnesses)
            })
        
        print(f"GA complete: Best = {best_fitness:.2f}%")
        
        eval_history = [{
            'evaluation': i + 1,
            'hyperparameters': best_params,
            'fitness': best_fitness,
            'timestamp': i
        } for i in range(len(history))]
        
        return best_params, best_fitness, eval_history
    
    def _random_individual(self) -> Dict[str, Any]:
        """Generate random individual from search space"""
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
    
    def _tournament_select(self, population: List[Dict], fitnesses: List[float], k: int = 3) -> Dict:
        """Select parent using tournament selection"""
        tournament_idx = random.sample(range(len(population)), k)
        tournament_fitnesses = [fitnesses[i] for i in tournament_idx]
        winner_idx = tournament_idx[np.argmax(tournament_fitnesses)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Single-point crossover between two parents"""
        child = {}
        params = list(parent1.keys())
        crossover_point = random.randint(1, len(params) - 1)
        
        for i, param in enumerate(params):
            child[param] = parent1[param] if i < crossover_point else parent2[param]
        
        return child
    
    def _mutate(self, individual: Dict) -> Dict:
        """Apply mutation to individual"""
        mutated = individual.copy()
        param_to_mutate = random.choice(list(mutated.keys()))
        param_config = self.search_space[param_to_mutate]
        
        if isinstance(param_config, list):
            mutated[param_to_mutate] = random.choice(param_config)
        elif isinstance(param_config, tuple) and len(param_config) == 3:
            min_val, max_val, scale = param_config
            current = mutated[param_to_mutate]
            
            if scale == 'log':
                log_current = np.log10(current)
                log_min = np.log10(min_val)
                log_max = np.log10(max_val)
                log_range = log_max - log_min
                noise = np.random.normal(0, log_range * 0.1)
                new_log = np.clip(log_current + noise, log_min, log_max)
                mutated[param_to_mutate] = 10 ** new_log
            else:
                range_val = max_val - min_val
                noise = np.random.normal(0, range_val * 0.1)
                mutated[param_to_mutate] = np.clip(current + noise, min_val, max_val)
        
        return mutated
