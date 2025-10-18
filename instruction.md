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

### Technical Specifications
- [ ] **Hardware Optimization**: Configure for MacBook Pro M1 Pro (32GB RAM, 16-core Metal GPU)
- [ ] **Library Requirements**: Use DEAP library for evolutionary algorithms implementation
- [ ] **Parallelization**: Utilize full system capabilities with parallel processing
- [ ] **Neural Networks**: Simple but accurate architectures suitable for hyperparameter optimization

### Data Persistence & Resumability
- [ ] **Checkpoint System**: Implement saving/loading of experiment state
- [ ] **Result Storage**: Store intermediate and final results for each algorithm run
- [ ] **Progress Tracking**: Save convergence curves and performance metrics
- [ ] **Crash Recovery**: Ability to resume experiments from last checkpoint

### Execution Modes
- [ ] **Full Run Mode**: Execute all algorithms (GA, DE, PSO, Grid, Random) on both datasets
- [ ] **Specific Algorithm Mode**: Run individual algorithms for targeted experiments
- [ ] **Light Run Mode**: Quick demonstration run (few minutes) for video recording
- [ ] **Analysis Mode**: Generate figures, tables, and comparative analysis

### Output Generation
- [ ] **Figures**: Convergence curves, performance comparisons, algorithm behavior
- [ ] **Data Files**: CSV/JSON exports of results for statistical analysis
- [ ] **Reports**: Automated summary generation with key findings
- [ ] **Reproducibility**: Ensure consistent results across runs with proper random seeding

### Code Structure Requirements
- [ ] **Modular Design**: Separate modules for algorithms, models, data handling, and analysis
- [ ] **Configuration**: External configuration files for easy parameter adjustment
- [ ] **Documentation**: Clear code documentation and usage examples
- [ ] **Testing**: Unit tests for critical components

### Jupyter Notebook Requirements
- [ ] **Demo Section**: Quick demonstration of each algorithm
- [ ] **Full Analysis**: Comprehensive comparison and statistical analysis
- [ ] **Visualization**: Interactive plots and comparative charts
- [ ] **Reproducible Results**: Ability to regenerate all results from saved data
