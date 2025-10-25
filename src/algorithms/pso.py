"""Particle Swarm Optimization implementation"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple, Callable


class ParticleSwarmOptimization:
    """Particle Swarm Optimization optimizer"""
    
    def __init__(self, search_space: Dict[str, Any],
                 swarm_size: int = 10,
                 iterations: int = 20,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5):
        self.search_space = search_space
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
    def optimize(self, eval_func: Callable) -> Tuple[Dict, float, List]:
        """Run PSO optimization"""
        particles = [self._random_individual() for _ in range(self.swarm_size)]
        velocities = [self._random_velocity() for _ in range(self.swarm_size)]
        fitnesses = [eval_func(p) for p in particles]
        
        p_best = [p.copy() for p in particles]
        p_best_fitness = fitnesses.copy()
        
        g_best_idx = np.argmax(fitnesses)
        g_best = particles[g_best_idx].copy()
        g_best_fitness = fitnesses[g_best_idx]
        
        history = []
        
        print(f"PSO: {self.swarm_size} particles x {self.iterations} iter")
        
        for iteration in range(self.iterations):
            for i in range(self.swarm_size):
                velocities[i] = self._update_velocity(
                    velocities[i], particles[i], p_best[i], g_best
                )
                
                particles[i] = self._update_position(particles[i], velocities[i])
                
                fitnesses[i] = eval_func(particles[i])
                
                if fitnesses[i] > p_best_fitness[i]:
                    p_best[i] = particles[i].copy()
                    p_best_fitness[i] = fitnesses[i]
                
                if fitnesses[i] > g_best_fitness:
                    g_best = particles[i].copy()
                    g_best_fitness = fitnesses[i]
                    print(f"Iter {iteration+1}: New best = {g_best_fitness:.2f}%")
            
            history.append({
                'iteration': iteration + 1,
                'best_fitness': g_best_fitness,
                'avg_fitness': np.mean(fitnesses)
            })
        
        print(f"PSO complete: Best = {g_best_fitness:.2f}%")
        
        eval_history = [{
            'evaluation': i + 1,
            'hyperparameters': g_best,
            'fitness': g_best_fitness,
            'timestamp': i
        } for i in range(len(history))]
        
        return g_best, g_best_fitness, eval_history
    
    def _random_individual(self) -> Dict[str, Any]:
        """Generate random particle from search space"""
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
    
    def _random_velocity(self) -> Dict[str, float]:
        """Initialize random velocity for each particle"""
        velocity = {}
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, list):
                velocity[param_name] = 0
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                min_val, max_val, scale = param_config
                if scale == 'log':
                    log_range = np.log10(max_val) - np.log10(min_val)
                    velocity[param_name] = random.uniform(-log_range * 0.1, log_range * 0.1)
                else:
                    range_val = max_val - min_val
                    velocity[param_name] = random.uniform(-range_val * 0.1, range_val * 0.1)
        return velocity
    
    def _update_velocity(self, velocity: Dict, position: Dict, 
                        p_best: Dict, g_best: Dict) -> Dict:
        """Update particle velocity"""
        new_velocity = {}
        for param in velocity.keys():
            if param.startswith('_'):
                continue
            param_config = self.search_space[param]
            
            if isinstance(param_config, list):
                new_velocity[param] = 0
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                min_val, max_val, scale = param_config
                
                r1 = random.random()
                r2 = random.random()
                
                if scale == 'log':
                    log_pos = np.log10(position[param])
                    log_pbest = np.log10(p_best[param])
                    log_gbest = np.log10(g_best[param])
                    
                    new_v = (self.w * velocity[param] +
                            self.c1 * r1 * (log_pbest - log_pos) +
                            self.c2 * r2 * (log_gbest - log_pos))
                    
                    log_range = np.log10(max_val) - np.log10(min_val)
                    new_v = np.clip(new_v, -log_range * 0.2, log_range * 0.2)
                    new_velocity[param] = new_v
                else:
                    new_v = (self.w * velocity[param] +
                            self.c1 * r1 * (p_best[param] - position[param]) +
                            self.c2 * r2 * (g_best[param] - position[param]))
                    
                    range_val = max_val - min_val
                    new_v = np.clip(new_v, -range_val * 0.2, range_val * 0.2)
                    new_velocity[param] = new_v
        
        return new_velocity
    
    def _update_position(self, position: Dict, velocity: Dict) -> Dict:
        """Update particle position"""
        new_position = {}
        for param in position.keys():
            if param.startswith('_'):
                continue
            param_config = self.search_space[param]
            
            if isinstance(param_config, list):
                if random.random() < 0.1:
                    new_position[param] = random.choice(param_config)
                else:
                    new_position[param] = position[param]
            elif isinstance(param_config, tuple) and len(param_config) == 3:
                min_val, max_val, scale = param_config
                
                if scale == 'log':
                    log_pos = np.log10(position[param])
                    new_log_pos = log_pos + velocity[param]
                    new_log_pos = np.clip(new_log_pos, np.log10(min_val), np.log10(max_val))
                    new_position[param] = 10 ** new_log_pos
                else:
                    new_pos = position[param] + velocity[param]
                    new_position[param] = np.clip(new_pos, min_val, max_val)
        
        return new_position
