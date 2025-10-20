Portfolio Requirements
- [x] Choose a topic in an area of nature-inspired computing from the list of possible topics given below. If you want to work on a topic of your own choice, you must obtain prior approval from the course coordinator. 
- [x] Conduct a brief literature review on how this topic has been approached in the past. Gather a list of relevant references that are cited in your literature review. A typically adequate list will contain 15-30 references.
- [ ] Decide on an aspect of the algorithm that you will focus on for experimentation. Obtain a suitable dataset or generate your own data for your computer experiments that are appropriate for your topic.
- [ ] Obtain numerical results using Python. Summarise and present your code and results using tables, graphs or other suitable means in a Jupyter notebook.
- [ ] Draw suitable conclusions based on your analysis of the results.
- [ ] Write up a PDF report based on your selected topic and experimental results obtained using Python&Jupyter. 
- [ ] Prepare a video recording to demonstrate your Jupyter code can successfully obtain the results in your report.

## Detailed Implementation Requirements

### Topic Focus
- **Topic**: Hyperparameter optimization of simple neural networks using evolutionary algorithms
- **Algorithms**: Genetic Algorithm (GA), Differential Evolution (DE), Particle Swarm Optimization (PSO)
- **Baselines**: Grid Search, Random Search
- **Datasets**: MNIST and CIFAR-10

### Technical Specifications ✅ SIMPLIFIED
- [x] **Hardware**: MacBook Pro M1 Pro (32GB RAM, 16-core Metal GPU)
- [x] **Libraries**: PyTorch for neural networks, custom evolutionary algorithm implementations
- [x] **Parallelization**: Run MNIST and CIFAR-10 experiments in separate terminals (dataset-level parallelism)
- [x] **Neural Networks**: Simple MLP for MNIST, simple CNN for CIFAR-10

### Data Persistence ✅ SIMPLIFIED
- [x] **Result Storage**: Single JSON file per experiment (algorithm_dataset_timestamp.json)
- [x] **No Checkpoints**: Removed checkpoint system for simplicity
- [x] **No Logs**: Removed individual log files
- [x] **Clean Output**: results/ folder contains only JSON files

### Execution Modes ✅ SIMPLIFIED
- [x] **Single Task Mode**: Run one algorithm on one dataset at a time
- [x] **Parallel Execution**: Launch multiple experiments in different terminals
- [x] **Separate Visualization**: Generate plots after experiments complete
- [ ] **Removed**: Light mode, video demo mode (over-engineered)

### Output Generation ✅ SIMPLIFIED
- [x] **JSON Results**: Evaluation history, best hyperparameters, timing info
- [x] **Figures**: Generated separately using plot_results.py
- [x] **Statistical Summary**: Printed to console and included in plots
- [x] **Reproducibility**: Random seed set in config

### Code Structure Requirements ✅ SIMPLIFIED FOR UNIVERSITY PROJECT
- [x] **Simple Modular Design**: Core modules only
  - `src/evolutionary_algorithms.py` - GA, DE, PSO implementations
  - `src/baseline_methods.py` - Grid Search, Random Search
  - `src/models.py` - Neural network architectures
  - `src/trainer.py` - Training and evaluation logic
  - `src/data_loader.py` - Dataset loading and preprocessing
- [x] **Simple Execution Scripts**:
  - `simple_run.py` - Run single algorithm on single dataset
  - `plot_results.py` - Generate comparison figures
- [x] **Minimal Output**: Single JSON file per experiment
- [x] **No Unnecessary Features**: Removed checkpoints, logs, config folders, models folders, figure folders per experiment

### Jupyter Notebook Requirements
- [ ] **Load and Analyze**: Load saved JSON results and analyze
- [ ] **Comparison Analysis**: Statistical comparison between algorithms
- [ ] **Visualization**: Generate plots and tables
- [ ] **Conclusions**: Draw insights from experimental results
