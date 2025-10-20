# Hyperparameter Optimization using Evolutionary Algorithms

Simple university-level implementation comparing evolutionary algorithms (GA, DE, PSO) with traditional methods (Grid, Random) for neural network hyperparameter optimization on MNIST and CIFAR-10.

## 📁 Project Structure

```
Project_Report/
├── simple_run.py          # Main experiment runner (one algorithm, one dataset)
├── plot_results.py        # Results visualization generator
├── src/                   # Core implementations
│   ├── evolutionary_algorithms.py
│   ├── baseline_methods.py
│   ├── models.py
│   ├── trainer.py
│   └── data_loader.py
├── results/               # Experiment outputs (JSON files only)
└── figures/               # Generated plots
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install torch torchvision numpy matplotlib
```

### 2. Run Experiments (Parallel Execution)

**Run multiple experiments simultaneously in different terminals:**

```bash
# Terminal 1: Grid Search on MNIST
python simple_run.py --algorithm grid --dataset mnist

# Terminal 2: Grid Search on CIFAR-10 
python simple_run.py --algorithm grid --dataset cifar10

# Terminal 3: Random Search on MNIST
python simple_run.py --algorithm random --dataset mnist

# Terminal 4: Random Search on CIFAR-10
python simple_run.py --algorithm random --dataset cifar10
```

**After baselines complete, run evolutionary algorithms:**

```bash
python simple_run.py --algorithm ga --dataset mnist
python simple_run.py --algorithm de --dataset mnist
python simple_run.py --algorithm pso --dataset mnist
```

### 3. Generate Plots

```bash
# Plot single algorithm
python plot_results.py --algorithm grid --dataset mnist

# Compare algorithms
python plot_results.py --compare grid random --dataset mnist

# Generate comprehensive comparison
python plot_results.py --all
```

## 📊 Expected Results

### Execution Times (Approximate)
- **MNIST**: 20-40 minutes per algorithm (3 runs)
- **CIFAR-10**: 1-3 hours per algorithm (3 runs)
- **Total**: 8-16 hours for all algorithms on both datasets

### Output Files
```
results/
├── grid_mnist_20251020_100000.json
├── grid_cifar10_20251020_120000.json
├── random_mnist_20251020_110000.json
└── ... (one JSON file per experiment)

figures/
├── grid_mnist.png
├── compare_grid_random_mnist.png
└── master_comparison.png
```

## 🎯 Key Features

### ✅ Simplified Design
- **No checkpoints**: Just run experiments to completion
- **No log files**: Output goes to console
- **Single JSON per experiment**: Clean, simple output
- **No complex folder structure**: Just results/ and figures/

### ✅ Parallel Execution
- Run MNIST and CIFAR-10 simultaneously in different terminals
- Run multiple algorithms at once
- Efficient use of system resources

### ✅ Independent Visualization
- Generate plots after experiments complete
- Compare any combination of algorithms
- Flexible analysis workflow

## 📖 Detailed Usage

### Running Experiments

```bash
python simple_run.py --algorithm <algo> --dataset <dataset> [options]

Required Arguments:
  --algorithm, -a    Algorithm: grid, random, ga, de, pso
  --dataset, -d      Dataset: mnist, cifar10

Optional Arguments:
  --runs, -r         Number of independent runs (default: 3)
  --evaluations, -e  Max evaluations per run (default: 20)

Examples:
  # Quick test (1 run, 10 evaluations)
  python simple_run.py -a random -d mnist -r 1 -e 10
  
  # Full experiment (3 runs, 20 evaluations)
  python simple_run.py -a ga -d mnist -r 3 -e 20
```

### Generating Plots

```bash
python plot_results.py [options]

Options:
  --algorithm, -a    Plot specific algorithm
  --dataset, -d      Specify dataset
  --compare          Compare multiple algorithms
  --all              Generate comprehensive comparison

Examples:
  # Single algorithm plot
  python plot_results.py -a grid -d mnist
  
  # Compare two algorithms
  python plot_results.py --compare grid random -d mnist
  
  # Compare all algorithms
  python plot_results.py --compare grid random ga de pso -d mnist
  
  # Master comparison (all results)
  python plot_results.py --all
```

## 🔧 Configuration

### Hyperparameter Search Space

**Baseline Methods (Grid/Random):**
```python
{
    'learning_rate': (0.0001, 0.01),    # log scale
    'batch_size': [64, 128, 256, 512],
    'dropout_rate': (0.0, 0.5),
    'hidden_units': [64, 128, 256, 512],
    'optimizer': ['adam', 'sgd', 'rmsprop'],
    'weight_decay': (0.0, 0.01)
}
```

**Evolutionary Algorithms (GA/DE/PSO):**
- Population size: 6
- Generations: 10
- Total evaluations: ~60 per run

### Neural Network Architectures

**MNIST (Simple MLP):**
- Input: 784 (28×28)
- Hidden layers: 2-3 layers
- Hidden units: 64-512 (tunable)
- Output: 10 classes

**CIFAR-10 (Simple CNN):**
- Input: 3×32×32
- Conv layers: 3 layers
- Hidden units: 64-512 (tunable)
- Output: 10 classes

## 📈 Monitoring Experiments

### Check Running Processes
```bash
# List all running experiments
ps aux | grep "simple_run.py" | grep -v grep

# Monitor system resources
htop  # or top on macOS
```

### Expected Console Output
```
======================================================================
🚀 Running GRID on MNIST
======================================================================

📊 Run 1/3
----------------------------------------
ℹ️ Evaluating configuration 1/20...
   Learning rate: 0.001, Batch size: 128, ...
   Fitness: 97.23%
...
✅ Run 1 complete!
   Best accuracy: 98.12%
   Time: 12.3 minutes

...

======================================================================
📈 EXPERIMENT SUMMARY
======================================================================
Algorithm: GRID
Dataset: MNIST
Runs: 3

Performance:
  Best accuracy: 98.45%
  Mean accuracy: 98.21% ± 0.18%
  Worst accuracy: 98.02%

Time:
  Total: 38.7 minutes
  Per run: 12.9 minutes
======================================================================

💾 Results saved to: results/grid_mnist_20251020_143022.json
```

## 🎓 For Jupyter Notebook

Load and analyze results:

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Load results
with open('results/grid_mnist_20251020_143022.json') as f:
    results = json.load(f)

# Extract best accuracies
best_accs = [run['best_fitness'] for run in results['runs']]

# Statistical analysis
print(f"Mean: {np.mean(best_accs):.2f}%")
print(f"Std: {np.std(best_accs):.2f}%")
print(f"Best: {max(best_accs):.2f}%")

# Plot convergence
for run in results['runs']:
    history = run['evaluation_history']
    evals = [h['evaluation'] for h in history]
    fitness = [h['fitness'] for h in history]
    plt.plot(evals, fitness, 'o-', alpha=0.7, label=f"Run {run['run']}")

plt.xlabel('Evaluation')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Grid Search Convergence on MNIST')
plt.show()
```

## ❓ Troubleshooting

### Experiment Taking Too Long?
- Reduce `--evaluations` (e.g., `-e 10`)
- Reduce `--runs` (e.g., `-r 1`)
- Use MNIST instead of CIFAR-10

### Out of Memory?
- Close other applications
- Reduce batch size in hyperparameter space
- Run fewer experiments in parallel

### Want to Stop Experiment?
- Press `Ctrl+C` in terminal
- Results are only saved after completion
- No intermediate checkpoints (by design)

## 📝 Citation

If you use this code, please cite:
```
Hyperparameter Optimization using Evolutionary Algorithms
COMP815 Nature-Inspired Computing Project
Auckland University of Technology, 2025
```

## 📧 Contact

Kai Cho - Auckland University of Technology

## 📄 License

Educational use only - Auckland University of Technology
