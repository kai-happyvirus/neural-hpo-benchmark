"""
Data persistence and checkpoint management system
Handles saving/loading experiment state, results, and models
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import yaml
import h5py
from pathlib import Path


class ExperimentManager:
    """Manages experiment data persistence and organization"""
    
    def __init__(self, base_dir: str = "./", experiment_name: Optional[str] = None):
        self.base_dir = Path(base_dir)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"hpo_experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.base_dir / "results" / experiment_name
        
        # Create directory structure
        self._setup_directories()
        
        # Initialize experiment metadata
        self.metadata = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'status': 'initialized',
            'algorithms_run': [],
            'datasets_used': [],
            'total_evaluations': 0
        }
        
        self._save_metadata()
    
    def _setup_directories(self):
        """Create experiment directory structure"""
        directories = [
            self.experiment_dir,
            self.experiment_dir / "checkpoints",
            self.experiment_dir / "results",
            self.experiment_dir / "models",
            self.experiment_dir / "figures",
            self.experiment_dir / "logs",
            self.experiment_dir / "config"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _save_metadata(self):
        """Save experiment metadata"""
        metadata_path = self.experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def update_metadata(self, **kwargs):
        """Update experiment metadata"""
        self.metadata.update(kwargs)
        self.metadata['last_updated'] = datetime.now().isoformat()
        self._save_metadata()
    
    def save_config(self, config: Dict[str, Any], filename: str = "experiment_config.yaml"):
        """Save experiment configuration"""
        config_path = self.experiment_dir / "config" / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def load_config(self, filename: str = "experiment_config.yaml") -> Dict[str, Any]:
        """Load experiment configuration"""
        config_path = self.experiment_dir / "config" / filename
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_checkpoint(self, algorithm: str, generation: int, data: Dict[str, Any]):
        """Save algorithm checkpoint"""
        checkpoint_dir = self.experiment_dir / "checkpoints" / algorithm
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"generation_{generation:04d}.pkl"
        
        checkpoint_data = {
            'algorithm': algorithm,
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, algorithm: str, generation: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Load algorithm checkpoint"""
        checkpoint_dir = self.experiment_dir / "checkpoints" / algorithm
        
        if not checkpoint_dir.exists():
            return None
        
        if generation is None:
            # Find latest checkpoint
            checkpoint_files = list(checkpoint_dir.glob("generation_*.pkl"))
            if not checkpoint_files:
                return None
            checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        else:
            checkpoint_path = checkpoint_dir / f"generation_{generation:04d}.pkl"
            if not checkpoint_path.exists():
                return None
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint_data
    
    def save_results(self, algorithm: str, dataset: str, results: Dict[str, Any], 
                    run_id: Optional[str] = None):
        """Save algorithm results"""
        if run_id is None:
            timestamp = datetime.now().strftime("%H%M%S")
            run_id = f"run_{timestamp}"
        
        results_dir = self.experiment_dir / "results" / algorithm / dataset
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = results_dir / f"{run_id}_results.json"
        
        # Convert numpy arrays and other non-serializable objects
        serializable_results = self._make_serializable(results)
        
        with open(json_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save as pickle (preserves all data types)
        pickle_path = results_dir / f"{run_id}_results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save convergence data as CSV if available
        if 'fitness_history' in results:
            csv_path = results_dir / f"{run_id}_convergence.csv"
            df = pd.DataFrame(results['fitness_history'])
            df.to_csv(csv_path, index=False)
        
        print(f"Results saved: {results_dir}/{run_id}")
        
        # Update metadata
        if algorithm not in self.metadata['algorithms_run']:
            self.metadata['algorithms_run'].append(algorithm)
        if dataset not in self.metadata['datasets_used']:
            self.metadata['datasets_used'].append(dataset)
        
        self._save_metadata()
    
    def load_results(self, algorithm: str, dataset: str, run_id: str) -> Optional[Dict[str, Any]]:
        """Load algorithm results"""
        results_dir = self.experiment_dir / "results" / algorithm / dataset
        pickle_path = results_dir / f"{run_id}_results.pkl"
        
        if not pickle_path.exists():
            return None
        
        with open(pickle_path, 'rb') as f:
            results = pickle.load(f)
        
        return results
    
    def get_all_results(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Get all results organized by algorithm and dataset"""
        all_results = {}
        results_dir = self.experiment_dir / "results"
        
        if not results_dir.exists():
            return all_results
        
        for algorithm_dir in results_dir.iterdir():
            if not algorithm_dir.is_dir():
                continue
            
            algorithm = algorithm_dir.name
            all_results[algorithm] = {}
            
            for dataset_dir in algorithm_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue
                
                dataset = dataset_dir.name
                all_results[algorithm][dataset] = []
                
                for result_file in dataset_dir.glob("*_results.pkl"):
                    with open(result_file, 'rb') as f:
                        result = pickle.load(f)
                    result['run_id'] = result_file.stem.replace('_results', '')
                    all_results[algorithm][dataset].append(result)
        
        return all_results
    
    def save_model(self, model: torch.nn.Module, algorithm: str, dataset: str, 
                  run_id: str, hyperparams: Dict[str, Any]):
        """Save trained model"""
        models_dir = self.experiment_dir / "models" / algorithm / dataset
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_path = models_dir / f"{run_id}_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Save model info
        model_info = {
            'run_id': run_id,
            'algorithm': algorithm,
            'dataset': dataset,
            'hyperparameters': hyperparams,
            'model_class': model.__class__.__name__,
            'saved_at': datetime.now().isoformat()
        }
        
        info_path = models_dir / f"{run_id}_model_info.json"
        with open(info_path, 'w') as f:
            json.dump(self._make_serializable(model_info), f, indent=2)
    
    def load_model(self, model_class, algorithm: str, dataset: str, run_id: str) -> Optional[torch.nn.Module]:
        """Load trained model"""
        models_dir = self.experiment_dir / "models" / algorithm / dataset
        model_path = models_dir / f"{run_id}_model.pth"
        info_path = models_dir / f"{run_id}_model_info.json"
        
        if not model_path.exists() or not info_path.exists():
            return None
        
        # Load model info
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # Create model instance
        model = model_class(model_info['hyperparameters'])
        
        # Load state dict
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        return model
    
    def save_figure(self, figure, filename: str, algorithm: Optional[str] = None):
        """Save matplotlib figure"""
        if algorithm:
            figure_dir = self.experiment_dir / "figures" / algorithm
        else:
            figure_dir = self.experiment_dir / "figures"
        
        figure_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as both PNG and PDF
        png_path = figure_dir / f"{filename}.png"
        pdf_path = figure_dir / f"{filename}.pdf"
        
        figure.savefig(png_path, dpi=300, bbox_inches='tight')
        figure.savefig(pdf_path, bbox_inches='tight')
        
        print(f"Figure saved: {png_path}")
    
    def export_results_csv(self, filename: str = "all_results.csv"):
        """Export all results to a single CSV file"""
        all_results = self.get_all_results()
        
        rows = []
        for algorithm, datasets in all_results.items():
            for dataset, runs in datasets.items():
                for run_data in runs:
                    row = {
                        'algorithm': algorithm,
                        'dataset': dataset,
                        'run_id': run_data.get('run_id', ''),
                        'best_fitness': run_data.get('best_fitness', 0),
                        'final_fitness': run_data.get('final_val_accuracy', 0),
                        'total_evaluations': run_data.get('total_evaluations', 0),
                        'total_time': run_data.get('total_time', 0),
                        'converged': run_data.get('converged', False)
                    }
                    
                    # Add best hyperparameters
                    best_hyperparams = run_data.get('best_hyperparameters', {})
                    for param, value in best_hyperparams.items():
                        row[f'best_{param}'] = value
                    
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = self.experiment_dir / filename
        df.to_csv(csv_path, index=False)
        
        print(f"Results exported to: {csv_path}")
        return df
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment summary statistics"""
        all_results = self.get_all_results()
        
        summary = {
            'experiment_name': self.experiment_name,
            'total_algorithms': len(all_results),
            'total_datasets': len(set(dataset for datasets in all_results.values() for dataset in datasets.keys())),
            'total_runs': sum(len(runs) for datasets in all_results.values() for runs in datasets.values()),
            'algorithms': list(all_results.keys()),
            'datasets': list(set(dataset for datasets in all_results.values() for dataset in datasets.keys())),
            'best_results': {}
        }
        
        # Find best results for each algorithm-dataset combination
        for algorithm, datasets in all_results.items():
            summary['best_results'][algorithm] = {}
            for dataset, runs in datasets.items():
                if runs:
                    best_run = max(runs, key=lambda r: r.get('best_fitness', 0))
                    summary['best_results'][algorithm][dataset] = {
                        'best_fitness': best_run.get('best_fitness', 0),
                        'run_id': best_run.get('run_id', ''),
                        'hyperparameters': best_run.get('best_hyperparameters', {})
                    }
        
        return summary


def cleanup_old_checkpoints(experiment_dir: str, keep_last_n: int = 5):
    """Clean up old checkpoint files, keeping only the last N"""
    experiment_path = Path(experiment_dir)
    checkpoints_dir = experiment_path / "checkpoints"
    
    if not checkpoints_dir.exists():
        return
    
    for algorithm_dir in checkpoints_dir.iterdir():
        if not algorithm_dir.is_dir():
            continue
        
        checkpoint_files = sorted(algorithm_dir.glob("generation_*.pkl"))
        
        if len(checkpoint_files) > keep_last_n:
            files_to_delete = checkpoint_files[:-keep_last_n]
            for file_path in files_to_delete:
                file_path.unlink()
                print(f"Deleted old checkpoint: {file_path}")


if __name__ == "__main__":
    # Test the experiment manager
    print("Testing ExperimentManager...")
    
    # Create test experiment
    exp_manager = ExperimentManager(experiment_name="test_experiment")
    
    # Test config saving/loading
    test_config = {
        'algorithms': ['ga', 'de', 'pso'],
        'datasets': ['mnist', 'cifar10'],
        'population_size': 20
    }
    
    exp_manager.save_config(test_config)
    loaded_config = exp_manager.load_config()
    print(f"Config saved and loaded: {loaded_config}")
    
    # Test results saving
    test_results = {
        'best_fitness': 95.5,
        'best_hyperparameters': {'learning_rate': 0.001, 'batch_size': 64},
        'fitness_history': [
            {'generation': 0, 'max': 80.0, 'avg': 70.0},
            {'generation': 1, 'max': 85.0, 'avg': 75.0},
            {'generation': 2, 'max': 90.0, 'avg': 80.0}
        ]
    }
    
    exp_manager.save_results('ga', 'mnist', test_results, 'test_run')
    
    # Test summary
    summary = exp_manager.get_experiment_summary()
    print(f"Experiment summary: {summary}")
    
    print("ExperimentManager test completed!")