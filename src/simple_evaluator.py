"""
Simple Sequential Evaluation - Fast and Reliable
No multiprocessing complications, optimized for speed and cross-platform compatibility
"""

import time
from typing import Dict, Any, Callable, List
import torch


class SimpleEvaluator:
    """Simple, fast sequential evaluator for hyperparameter optimization"""
    
    def __init__(self, config: Dict[str, Any], dataset: str, batch_size: int):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Initialize components once
        self._setup_data()
        self._setup_trainer()
        
    def _setup_data(self):
        """Setup data loaders once"""
        from data_loader import DataManager
        
        self.data_manager = DataManager(self.config)
        self.train_loader, self.val_loader, self.test_loader = self.data_manager.get_dataloaders(
            self.dataset, self.batch_size, self.config['training']['validation_split']
        )
        
    def _setup_trainer(self):
        """Setup trainer once"""
        from trainer import ModelTrainer
        
        device = self.config.get('hardware', {}).get('device', 'auto')
        self.trainer = ModelTrainer(device=device)
        
    def evaluate_individual(self, hyperparams: Dict[str, Any]) -> float:
        """Evaluate a single individual quickly"""
        try:
            # Merge with training config
            training_config = self.config.get('training', {})
            full_hyperparams = {**training_config, **hyperparams}
            
            # Quick evaluation
            fitness = self.trainer.evaluate_hyperparameters(
                full_hyperparams, self.train_loader, self.val_loader, 
                self.test_loader, self.dataset
            )
            
            return float(fitness) if fitness is not None else 0.0
            
        except Exception as e:
            print(f"   ⚠️ Evaluation failed: {e}")
            return 0.0
    
    def evaluate_population(self, population: List[Dict[str, Any]]) -> List[float]:
        """Evaluate a population sequentially with progress updates"""
        fitnesses = []
        total = len(population)
        
        for i, individual in enumerate(population):
            if i % max(1, total // 4) == 0:  # Progress updates
                progress = (i / total) * 100
                print(f"   🔄 Evaluating: {progress:.0f}% ({i}/{total})")
            
            fitness = self.evaluate_individual(individual)
            fitnesses.append(fitness)
            
        print(f"   ✅ Evaluation complete: {total} individuals")
        return fitnesses


def create_simple_evaluation_function(config: Dict[str, Any], dataset: str, batch_size: int) -> Callable:
    """Create a simple, fast evaluation function"""
    
    print("🚀 Creating simple sequential evaluator (fast and reliable)")
    
    evaluator = SimpleEvaluator(config, dataset, batch_size)
    
    def evaluate_hyperparams(hyperparams: Dict[str, Any]) -> float:
        """Simple evaluation function"""
        return evaluator.evaluate_individual(hyperparams)
    
    return evaluate_hyperparams