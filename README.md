# Hyperparameter Optimization using Evolutionary Algorithms

Comparison of evolutionary algorithms (GA, DE, PSO) with traditional methods (Grid, Random) for neural network hyperparameter optimization on MNIST and CIFAR-10.

## Structure

```
Project_Report/
├── hpo_experiment.py           # Main experiment runner
├── notebooks/
│   ├── experiment_orchestrator.ipynb  # Run all experiments
│   ├── comprehensive_analysis.ipynb   # Analyze results  
│   └── demonstration.ipynb               # Demo for video
├── src/                        # Core implementations
├── results/                    # JSON experiment outputs
└── requirements.txt           # Dependencies
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
# Single experiment
python hpo_experiment.py --algorithm grid --dataset mnist

# All experiments
jupyter notebook notebooks/experiment_orchestrator.ipynb
```

### 3. Analyze Results
```bash
jupyter notebook notebooks/comprehensive_analysis.ipynb
```

## Available Algorithms
- **Grid Search**: Systematic parameter exploration
- **Random Search**: Random parameter sampling  
- **GA**: Genetic Algorithm
- **DE**: Differential Evolution
- **PSO**: Particle Swarm Optimization

## Datasets
- **MNIST**: 28×28 handwritten digits
- **CIFAR-10**: 32×32 color images

## Usage

```bash
python hpo_experiment.py --algorithm <algo> --dataset <data>

Arguments:
  --algorithm, -a    grid, random, ga, de, pso
  --dataset, -d      mnist, cifar10
  --runs, -r         Number of runs (default: 3)
  --evaluations, -e  Max evaluations (default: 20)
```

## Expected Runtime
- **MNIST**: 4-16 hours per algorithm (3 runs)
- **CIFAR-10**: 7-24 hours per algorithm (3 runs)
- **Total**: 55-200 hours for all experiments

**Note**: Evolutionary algorithms (GA, DE, PSO) take significantly longer than traditional methods (Grid, Random).

## Contact
Kai Cho - TokuEyes Ltd
