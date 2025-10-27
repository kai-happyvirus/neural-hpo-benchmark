# Demo Script: HPO Benchmark Study

## Opening (30 seconds)
"I'm presenting my research on hyperparameter optimization for neural networks. I'm a software engineer interested in optimization, and at my company we use many AI models. The challenge we face is the heavy computational cost of hyperparameter tuning, which inspired this research."

## Research Gap (45 seconds)
"When I reviewed the literature, I found many studies comparing HPO methods, but they all focus on final performance. What's missing is a **fair benchmark under identical conditions**—same hardware, same search space, same evaluation budget.

So my question was: *Which HPO strategy actually works best when you have limited resources?*"

## Study Overview (1 minute)
"I compared three evolutionary algorithms—Genetic Algorithm, Differential Evolution, and Particle Swarm Optimization—against grid search and random search on MNIST and CIFAR-10.

Why these three? They represent different evolutionary approaches: GA uses selection and crossover inspired by biological evolution, DE uses differential mutation for continuous spaces, and PSO mimics social behavior of particles. All three are well-established, handle continuous hyperparameters well, and are suitable for parallelization.

**Key results:**
- CIFAR-10: Evolutionary algorithms beat grid search by 2.39-6.66%
- PSO converges fastest but can get stuck
- GA is most consistent
- DE handles parameter variations best

All on M1 Pro, 3-9 hours per run—real consumer hardware."

## Why Custom Implementation (45 seconds)
"You might ask why I didn't use DEAP or scikit-learn. The answer: **fair comparison**.

To truly benchmark these algorithms, I needed:
- Same search space for all algorithms
- Same neural network architecture
- Same evaluation framework
- Same stopping criteria

Using different libraries would introduce hidden biases. Custom implementation ensures we're comparing the algorithms themselves, not library implementations."

## Demo Transition & Methodology Explanation (1 minute 30 seconds)
"Now let me show you the actual code. But first, let me explain the execution strategy.

**Training Time Reality:**
Each full experiment takes 8-12 hours to complete—training 60 neural networks per algorithm is computationally intensive. So for this demonstration, I've created a special demo notebook with reduced parameters: just 1 run with 3 epochs instead of 50. This lets you see how each algorithm behaves in real-time.

**Actual Experiment Execution:**
For the real experiments that generated my results, I used two approaches:

1. **Local Execution (What I Did):** I have an M1 Pro with 8 CPU cores and 16 GPU cores, so I opened 3-4 terminals and ran multiple algorithms in parallel—one terminal per algorithm. This cut my total runtime from 50+ hours sequential to about 15-20 hours.

2. **Sequential Execution (For Reproducibility):** I also provide an 'experiment_orchestrator.ipynb' notebook that runs all experiments sequentially. This is important because:
   - Jupyter notebooks have parallelization limitations
   - It's compatible with Google Colab for reproducibility
   - Anyone can run it step-by-step without terminal access

The pre-generated results in my 'results/' folder came from the parallel terminal approach—that's the data we'll analyze today.

Now let me demonstrate the algorithms with this quick demo..."

---

## While Running Demo - Talk About:

**As each algorithm runs:**
- "Grid search: systematic but exhaustive"
- "Random search: surprisingly effective baseline"
- "GA: population evolves over generations"
- "PSO: particles explore collaboratively"
- "DE: mutation and crossover strategies"

**Key Points to Mention:**
- All algorithms use same hyperparameter ranges
- Same neural network architecture
- Results save to demo folder for analysis

## Closing (30 seconds)
"This study shows evolutionary algorithms offer meaningful improvements, especially on complex datasets, with consumer hardware. The framework provides practical guidance for choosing HPO strategies under resource constraints.

For industry use—like at my company—this suggests evolutionary methods are worth implementing for better models without more hardware.

Questions?"

---

## Quick Reference
- **Hardware**: M1 Pro, 32GB RAM
- **CIFAR-10**: +2.39-6.66% vs grid, +3.44% vs random
- **Why custom**: Fair comparison, identical conditions
- **Best for speed**: PSO
- **Best for consistency**: GA  
- **Best for robustness**: DE

