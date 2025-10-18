"""
Evolutionary algorithms implementation using DEAP library
Includes Genetic Algorithm (GA), Differential Evolution (DE), and Particle Swarm Optimization (PSO)
Optimized for M1 Pro with parallelization support
"""

import random
import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional
from deap import base, creator, tools, algorithms
import copy
import time
import os
import pickle
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor, as_completed


class HyperparameterSpace:
    """Defines the hyperparameter search space"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hyperparams = config.get('hyperparameters', {})
        
        # Define parameter bounds and types
        self.param_info = self._setup_parameter_info()
        self.param_names = list(self.param_info.keys())
        self.dimension = len(self.param_names)
    
    def _setup_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """Setup parameter information with bounds and types"""
        param_info = {}
        
        # Learning rate (log scale)
        lr_config = self.hyperparams.get('learning_rate', {})
        param_info['learning_rate'] = {
            'type': 'continuous',
            'bounds': (lr_config.get('min', 0.0001), lr_config.get('max', 0.1)),
            'log_scale': lr_config.get('log_scale', True)
        }
        
        # Batch size (discrete choices)
        batch_sizes = self.hyperparams.get('batch_size', [32, 64, 128, 256])
        param_info['batch_size'] = {
            'type': 'discrete',
            'choices': batch_sizes
        }
        
        # Dropout rate
        dropout_config = self.hyperparams.get('dropout_rate', {})
        param_info['dropout_rate'] = {
            'type': 'continuous',
            'bounds': (dropout_config.get('min', 0.0), dropout_config.get('max', 0.5)),
            'log_scale': False
        }
        
        # Hidden units (discrete choices)
        hidden_units = self.hyperparams.get('hidden_units', [64, 128, 256, 512])
        param_info['hidden_units'] = {
            'type': 'discrete',
            'choices': hidden_units
        }
        
        # Optimizer (categorical)
        optimizers = self.hyperparams.get('optimizer', ['adam', 'sgd', 'rmsprop'])
        param_info['optimizer'] = {
            'type': 'categorical',
            'choices': optimizers
        }
        
        # Weight decay
        wd_config = self.hyperparams.get('weight_decay', {})
        param_info['weight_decay'] = {
            'type': 'continuous',
            'bounds': (wd_config.get('min', 0.0), wd_config.get('max', 0.01)),
            'log_scale': False
        }
        
        return param_info
    
    def encode_individual(self, hyperparams: Dict[str, Any]) -> List[float]:
        """Encode hyperparameters as a list of floats for evolutionary algorithms"""
        individual = []
        
        for param_name in self.param_names:
            param_info = self.param_info[param_name]
            value = hyperparams[param_name]
            
            if param_info['type'] == 'continuous':
                if param_info.get('log_scale', False):
                    # Convert to log scale
                    min_val, max_val = param_info['bounds']
                    log_min, log_max = np.log10(min_val), np.log10(max_val)
                    log_val = np.log10(value)
                    normalized = (log_val - log_min) / (log_max - log_min)
                else:
                    # Linear scale
                    min_val, max_val = param_info['bounds']
                    normalized = (value - min_val) / (max_val - min_val)
                individual.append(normalized)
                
            elif param_info['type'] in ['discrete', 'categorical']:
                # Encode as index position normalized to [0, 1]
                choices = param_info['choices']
                index = choices.index(value)
                normalized = index / (len(choices) - 1) if len(choices) > 1 else 0.0
                individual.append(normalized)
        
        return individual
    
    def decode_individual(self, individual: List[float]) -> Dict[str, Any]:
        """Decode a list of floats back to hyperparameters - Best Practice: Ensure numeric types"""
        hyperparams = {}
        
        for i, param_name in enumerate(self.param_names):
            param_info = self.param_info[param_name]
            # Best Practice: Convert to float to ensure numeric type compatibility
            try:
                raw_value = float(individual[i])  # Ensure numeric type
            except (ValueError, TypeError):
                raw_value = 0.5  # Safe fallback
            normalized_value = np.clip(raw_value, 0.0, 1.0)
            
            if param_info['type'] == 'continuous':
                if param_info.get('log_scale', False):
                    # Convert from log scale
                    min_val, max_val = param_info['bounds']
                    log_min, log_max = np.log10(min_val), np.log10(max_val)
                    log_val = log_min + normalized_value * (log_max - log_min)
                    value = 10 ** log_val
                else:
                    # Linear scale
                    min_val, max_val = param_info['bounds']
                    value = min_val + normalized_value * (max_val - min_val)
                hyperparams[param_name] = value
                
            elif param_info['type'] in ['discrete', 'categorical']:
                # Decode from index position
                choices = param_info['choices']
                index = int(normalized_value * (len(choices) - 1) + 0.5)
                index = max(0, min(index, len(choices) - 1))
                hyperparams[param_name] = choices[index]
        
        return hyperparams
    
    def random_individual(self) -> List[float]:
        """Generate a random individual"""
        return [random.random() for _ in range(self.dimension)]


class EvolutionaryOptimizer:
    """Base class for evolutionary optimization algorithms"""
    
    def __init__(self, config: Dict[str, Any], evaluation_function: Callable, experiment_manager=None):
        self.config = config
        self.evaluation_function = evaluation_function
        self.hyperparameter_space = HyperparameterSpace(config)
        self.experiment_manager = experiment_manager
        
        # Setup DEAP
        self._setup_deap()
        
        # Results storage
        self.results = {
            'best_individual': None,
            'best_fitness': 0.0,
            'fitness_history': [],
            'population_history': [],
            'convergence_data': []
        }
    
    def _setup_deap(self):
        """Setup DEAP framework"""
        # Create fitness and individual classes
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize accuracy
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        
        # Register functions - Best Practice: Ensure proper numeric types
        def random_float():
            return float(random.random())  # Explicit float conversion
        
        self.toolbox.register("random_float", random_float)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.random_float, n=self.hyperparameter_space.dimension)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_individual)
        
        # Register statistics
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
    
    def _evaluate_individual(self, individual: List[float]) -> Tuple[float]:
        """Evaluate an individual and return fitness"""
        try:
            # Decode hyperparameters
            hyperparams = self.hyperparameter_space.decode_individual(individual)
            
            # Evaluate using the provided function
            fitness = self.evaluation_function(hyperparams)
            
            return (fitness,)
        except Exception as e:
            print(f"Error evaluating individual: {e}")
            return (0.0,)
    
    def _evaluate_population_parallel(self, individuals: List) -> List[Tuple[float]]:
        """Evaluate population in parallel for speed"""
        try:
            # Get parallel processing settings
            hardware_config = self.config.get('hardware', {})
            use_parallel = hardware_config.get('use_multiprocessing', False)
            max_workers = hardware_config.get('max_parallel_processes', 2)
            
            if not use_parallel or len(individuals) < 2:
                # Fall back to sequential evaluation
                return [self._evaluate_individual(ind) for ind in individuals]
            
            # Simple parallel evaluation using map
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Map evaluation function to all individuals
                futures = [executor.submit(self._evaluate_individual, ind) for ind in individuals]
                
                # Collect results in order
                for future in futures:
                    try:
                        result = future.result(timeout=300)  # 5 min timeout per evaluation
                        results.append(result)
                    except Exception as e:
                        print(f"Parallel evaluation failed: {e}")
                        results.append((0.0,))  # Fallback fitness
            
            return results
            
        except Exception as e:
            print(f"Parallel evaluation error: {e}")
            # Fallback to sequential
            return [self._evaluate_individual(ind) for ind in individuals]
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run optimization (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def save_checkpoint(self, generation: int, population: List, filepath: str = None):
        """Save optimization checkpoint"""
        checkpoint_data = {
            'generation': generation,
            'population': population,
            'results': self.results,
            'algorithm_params': getattr(self, 'algorithm_params', {}),
            'hyperparameter_space_config': self.hyperparameter_space.config
        }
        
        # Use experiment manager if available, otherwise fall back to filepath
        if self.experiment_manager:
            algorithm_name = self.__class__.__name__.lower().replace('optimization', '').replace('algorithm', '')
            if not algorithm_name:  # Fallback if name extraction fails
                algorithm_name = 'unknown'
            print(f"   📁 Saving to experiment checkpoint: {algorithm_name}/generation_{generation:04d}")
            self.experiment_manager.save_checkpoint(algorithm_name, generation, checkpoint_data)
        elif filepath:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        else:
            # Default fallback
            os.makedirs("checkpoints", exist_ok=True)
            default_path = f"checkpoints/{self.__class__.__name__.lower()}_gen_{generation}.pkl"
            with open(default_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, filepath: str) -> Tuple[int, List]:
        """Load optimization checkpoint"""
        with open(filepath, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.results = checkpoint_data['results']
        return checkpoint_data['generation'], checkpoint_data['population']


class GeneticAlgorithm(EvolutionaryOptimizer):
    """Genetic Algorithm implementation"""
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Genetic Algorithm optimization"""
        # Extract parameters
        population_size = algorithm_params.get('population_size', 20)
        generations = algorithm_params.get('generations', 50)
        mutation_rate = algorithm_params.get('mutation_rate', 0.1)
        crossover_rate = algorithm_params.get('crossover_rate', 0.8)
        tournament_size = algorithm_params.get('tournament_size', 3)
        
        self.algorithm_params = algorithm_params
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=mutation_rate)
        self.toolbox.register("select", tools.selTournament, tournsize=tournament_size)
        
        # Initialize population
        population = self.toolbox.population(n=population_size)
        
        # Evaluate initial population (parallel)
        fitnesses = self._evaluate_population_parallel(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            
            # Select parents
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation
            for mutant in offspring:
                if random.random() < mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate invalid individuals (parallel)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:  # Only evaluate if there are invalid individuals
                fitnesses = self._evaluate_population_parallel(invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
            
            # Replace population
            population[:] = offspring
            
            # Record statistics
            fits = [ind.fitness.values[0] for ind in population]
            self.results['fitness_history'].append({
                'generation': generation,
                'min': min(fits),
                'max': max(fits),
                'avg': np.mean(fits),
                'std': np.std(fits)
            })
            
            # Update best individual
            best_ind = tools.selBest(population, 1)[0]
            if best_ind.fitness.values[0] > self.results['best_fitness']:
                self.results['best_individual'] = copy.deepcopy(best_ind)
                self.results['best_fitness'] = best_ind.fitness.values[0]
            
            # Save checkpoint every generation
            self.save_checkpoint(generation, population)
            print(f"   💾 Checkpoint saved: generation {generation + 1}")
        
        # Final results - with safety check
        if self.results['best_individual'] is not None:
            best_hyperparams = self.hyperparameter_space.decode_individual(self.results['best_individual'])
            self.results['best_hyperparameters'] = best_hyperparams
        else:
            # Fallback to random hyperparameters if no best individual found
            print("⚠️  Warning: No best individual found, using fallback hyperparameters")
            fallback_individual = self.hyperparameter_space.random_individual()
            best_hyperparams = self.hyperparameter_space.decode_individual(fallback_individual)
            self.results['best_hyperparameters'] = best_hyperparams
            self.results['best_individual'] = fallback_individual
        
        self.results['algorithm'] = 'Genetic Algorithm'
        
        return self.results


class DifferentialEvolution(EvolutionaryOptimizer):
    """Differential Evolution implementation"""
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Differential Evolution optimization"""
        # Extract parameters
        population_size = algorithm_params.get('population_size', 20)
        generations = algorithm_params.get('generations', 50)
        mutation_factor = algorithm_params.get('mutation_factor', 0.8)
        crossover_rate = algorithm_params.get('crossover_rate', 0.7)
        
        self.algorithm_params = algorithm_params
        
        # Initialize population
        population = self.toolbox.population(n=population_size)
        
        # Evaluate initial population (parallel)
        fitnesses = self._evaluate_population_parallel(population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Evolution loop
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            
            # Create all trials first, then evaluate in parallel
            trials = []
            for i, target in enumerate(population):
                # Select three random individuals (different from target)
                candidates = [j for j in range(len(population)) if j != i]
                a, b, c = random.sample(candidates, 3)
                
                # Create mutant vector
                mutant = []
                for j in range(len(target)):
                    gene = population[a][j] + mutation_factor * (population[b][j] - population[c][j])
                    gene = max(0.0, min(1.0, gene))  # Clip to [0, 1]
                    mutant.append(gene)
                
                # Create trial vector through crossover
                trial = creator.Individual()
                for j in range(len(target)):
                    if random.random() < crossover_rate or j == random.randrange(len(target)):
                        trial.append(mutant[j])
                    else:
                        trial.append(target[j])
                
                trials.append(trial)
            
            # Evaluate all trials in parallel
            trial_fitnesses = self._evaluate_population_parallel(trials)
            for trial, fit in zip(trials, trial_fitnesses):
                trial.fitness.values = fit
            
            # Selection - compare trials with targets
            new_population = []
            for i, (target, trial) in enumerate(zip(population, trials)):
                if trial.fitness.values[0] > target.fitness.values[0]:
                    new_population.append(trial)
                else:
                    new_population.append(copy.deepcopy(target))
            
            # Replace population
            population[:] = new_population
            
            # Record statistics
            fits = [ind.fitness.values[0] for ind in population]
            self.results['fitness_history'].append({
                'generation': generation,
                'min': min(fits),
                'max': max(fits),
                'avg': np.mean(fits),
                'std': np.std(fits)
            })
            
            # Update best individual
            best_ind = max(population, key=lambda x: x.fitness.values[0])
            if best_ind.fitness.values[0] > self.results['best_fitness']:
                self.results['best_individual'] = copy.deepcopy(best_ind)
                self.results['best_fitness'] = best_ind.fitness.values[0]
            
            # Save checkpoint every generation
            self.save_checkpoint(generation, population)
            print(f"   💾 Checkpoint saved: generation {generation + 1}")
        
        # Final results - with safety check  
        if self.results['best_individual'] is not None:
            best_hyperparams = self.hyperparameter_space.decode_individual(self.results['best_individual'])
            self.results['best_hyperparameters'] = best_hyperparams
        else:
            # Fallback to random hyperparameters if no best individual found
            print("⚠️  Warning: No best individual found, using fallback hyperparameters")
            fallback_individual = self.hyperparameter_space.random_individual()
            best_hyperparams = self.hyperparameter_space.decode_individual(fallback_individual)
            self.results['best_hyperparameters'] = best_hyperparams
            self.results['best_individual'] = fallback_individual
            
        self.results['algorithm'] = 'Differential Evolution'
        
        return self.results


class ParticleSwarmOptimization(EvolutionaryOptimizer):
    """Particle Swarm Optimization implementation"""
    
    def optimize(self, algorithm_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run PSO optimization"""
        # Extract parameters
        population_size = algorithm_params.get('population_size', 20)
        generations = algorithm_params.get('generations', 50)
        inertia_weight = algorithm_params.get('inertia_weight', 0.7)
        cognitive_factor = algorithm_params.get('cognitive_factor', 1.5)
        social_factor = algorithm_params.get('social_factor', 1.5)
        
        self.algorithm_params = algorithm_params
        
        # Initialize particles
        swarm = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for _ in range(population_size):
            particle = creator.Individual(self.hyperparameter_space.random_individual())
            velocity = [random.uniform(-1, 1) for _ in range(len(particle))]
            
            particle.fitness.values = self.toolbox.evaluate(particle)
            
            swarm.append(particle)
            velocities.append(velocity)
            personal_best.append(copy.deepcopy(particle))
            personal_best_fitness.append(particle.fitness.values[0])
        
        # Find global best
        global_best_idx = np.argmax(personal_best_fitness)
        global_best = copy.deepcopy(personal_best[global_best_idx])
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # Evolution loop
        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}")
            
            for i, particle in enumerate(swarm):
                # Update velocity
                for j in range(len(particle)):
                    cognitive_component = cognitive_factor * random.random() * (personal_best[i][j] - particle[j])
                    social_component = social_factor * random.random() * (global_best[j] - particle[j])
                    
                    velocities[i][j] = (inertia_weight * velocities[i][j] + 
                                      cognitive_component + social_component)
                    
                    # Update position
                    particle[j] += velocities[i][j]
                    particle[j] = max(0.0, min(1.0, particle[j]))  # Clip to [0, 1]
                
                # Evaluate particle
                particle.fitness.values = self.toolbox.evaluate(particle)
                
                # Update personal best
                if particle.fitness.values[0] > personal_best_fitness[i]:
                    personal_best[i] = copy.deepcopy(particle)
                    personal_best_fitness[i] = particle.fitness.values[0]
                    
                    # Update global best
                    if particle.fitness.values[0] > global_best_fitness:
                        global_best = copy.deepcopy(particle)
                        global_best_fitness = particle.fitness.values[0]
            
            # Record statistics
            fits = [p.fitness.values[0] for p in swarm]
            self.results['fitness_history'].append({
                'generation': generation,
                'min': min(fits),
                'max': max(fits),
                'avg': np.mean(fits),
                'std': np.std(fits)
            })
            
            # Update best individual
            self.results['best_individual'] = copy.deepcopy(global_best)
            self.results['best_fitness'] = global_best_fitness
            
            # Save checkpoint every generation
            self.save_checkpoint(generation, swarm)
            print(f"   💾 Checkpoint saved: generation {generation + 1}")
        
        # Final results - with safety check
        if self.results['best_individual'] is not None:
            best_hyperparams = self.hyperparameter_space.decode_individual(self.results['best_individual'])
            self.results['best_hyperparameters'] = best_hyperparams
        else:
            # Fallback to random hyperparameters if no best individual found
            print("⚠️  Warning: No best individual found, using fallback hyperparameters")
            fallback_individual = self.hyperparameter_space.random_individual()
            best_hyperparams = self.hyperparameter_space.decode_individual(fallback_individual)
            self.results['best_hyperparameters'] = best_hyperparams
            self.results['best_individual'] = fallback_individual
            
        self.results['algorithm'] = 'Particle Swarm Optimization'
        
        return self.results


def create_optimizer(algorithm: str, config: Dict[str, Any], 
                    evaluation_function: Callable, experiment_manager=None) -> EvolutionaryOptimizer:
    """Factory function to create optimizers"""
    algorithm = algorithm.lower()
    
    if algorithm in ['ga', 'genetic']:
        return GeneticAlgorithm(config, evaluation_function, experiment_manager)
    elif algorithm in ['de', 'differential']:
        return DifferentialEvolution(config, evaluation_function, experiment_manager)
    elif algorithm in ['pso', 'particle']:
        return ParticleSwarmOptimization(config, evaluation_function, experiment_manager)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


if __name__ == "__main__":
    # Test the evolutionary algorithms
    print("Testing evolutionary algorithms...")
    
    # Mock configuration
    config = {
        'hyperparameters': {
            'learning_rate': {'min': 0.0001, 'max': 0.1, 'log_scale': True},
            'batch_size': [32, 64, 128],
            'dropout_rate': {'min': 0.0, 'max': 0.5},
            'hidden_units': [64, 128, 256],
            'optimizer': ['adam', 'sgd'],
            'weight_decay': {'min': 0.0, 'max': 0.01}
        }
    }
    
    # Mock evaluation function
    def mock_evaluation(hyperparams):
        # Simple mock: return random fitness based on hyperparameters
        return random.random() * 100
    
    # Test hyperparameter space
    space = HyperparameterSpace(config)
    
    # Test encoding/decoding
    test_hyperparams = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'dropout_rate': 0.2,
        'hidden_units': 128,
        'optimizer': 'adam',
        'weight_decay': 0.001
    }
    
    encoded = space.encode_individual(test_hyperparams)
    decoded = space.decode_individual(encoded)
    
    print(f"Original: {test_hyperparams}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    print("Evolutionary algorithms test completed!")