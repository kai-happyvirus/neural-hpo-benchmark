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

## Demo Transition (15 seconds)
"Now let me show you the actual code. I'll run quick experiments with reduced parameters so you can see how each algorithm behaves..."

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

