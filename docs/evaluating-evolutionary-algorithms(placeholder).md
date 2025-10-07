# Evaluating Evolutionary Algorithms for Neural Network Hyperparameter Optimization

**Author:** Kai Cho  
**Institution:** Auckland University of Technology  
**Date:** October 2025

## Abstract

Hyperparameter optimization (HPO) drives neural network performance yet still feels more like an art than a science when you are racing a compute budget. I plan to benchmark three evolutionary algorithms—Genetic Algorithm (GA), Differential Evolution (DE), and Particle Swarm Optimization (PSO) against grid and random search on MNIST and CIFAR-10 classifiers. The study will run within identical hardware budgets so the evolutionary approaches can be fairly compared to classical baselines. Empirical results are pending; placeholders remain where accuracy and runtime figures will be inserted after the experiments complete on my 2020 MacBook Pro (Apple M1 Pro, 32 GB unified memory, 16-core GPU/Neural Engine). In the meantime, this document captures the motivation, experimental design, and anticipated analysis workflow so the eventual findings can be reported with full transparency.

## 1. Introduction

Tuning hyperparameters is still the part of a deep-learning project that sends me reaching for another coffee. Small nudges to learning rate, batch size, or optimizer settings can tilt a promising run into either the winner’s circle or a crash dump. The default response—grid or random search—remains popular because it is easy to explain and automate, but once the space mixes continuous, discrete, and categorical choices the cost curve shoots up (Bergstra & Bengio, 2012).

Evolutionary algorithms (EAs) promise a more adaptable search. In practice, though, it is hard to find head-to-head reports that show how the main EA families behave when they are given the same playground. This project therefore chases three pragmatic questions that surfaced while I was iterating on my models: (1) can GA, DE, and PSO really beat grid and random search when the compute budget is fixed, (2) how differently do they converge on datasets that vary in difficulty, and (3) what practical headaches pop up when wiring these algorithms into a deep-learning training loop. The goal is less about theory and more about giving fellow practitioners an honest field report.

## 2. Literature Review

### 2.1 Background: Hyperparameter Optimization in Neural Networks

Hyperparameter optimization (HPO) ultimately decides whether a neural network behaves or throws tantrums. Dials such as learning rate, batch size, optimizer choice, dropout rate, and network depth can either steady a training run or knock it off course. For years my approach mirrored the field’s early habits—manual sweeps, spreadsheets full of trial runs, and plenty of intuition (Bergstra & Bengio, 2012).

Structured methods like grid and random search brought some order but demanded far more evaluations than I could tolerate once the space turned high-dimensional or mixed-type. Bayesian optimization promised relief with Gaussian processes and Tree-structured Parzen Estimators (Snoek et al., 2012; Bergstra et al., 2013), yet those models still stumble over the discrete hierarchies baked into modern deep architectures. That gap nudged me toward population-based metaheuristics, whose parallelism and tolerance for odd-shaped search spaces looked like a better match.

### 2.2 Early Applications of Evolutionary Algorithms in Neural Network Optimization

Evolutionary computation began weaving into neural networks long before today’s AutoML tooling. Holland’s (1975) genetic algorithm nailed down the selection, crossover, and mutation operators that later researchers, including Montana and Davis (1989), tried directly on feedforward networks. Their success at evolving connection weights hinted that stochastic search might be more than a curiosity.

Later work pushed those ideas in practical directions that influence this project. Orive et al. (2014) compared crossover operators and showed that evolutionary initialization smoothed out training, a behaviour frequently reported by practitioners running neural HPO. Al-Shareef and Abbod (2010) and Huang and Wang (2006) experimented with convergence speed and feature selection but were boxed in by the compute of their era. NEAT (Stanley & Miikkulainen, 2002) and CoDeepNEAT (Miikkulainen et al., 2019) widened the ambition to full architectures, albeit with computational bills that are still intimidating for a single workstation.

### 2.3 Recent Advances in Evolutionary Hyperparameter Optimization

The recent wave of evolutionary HPO papers landed while I was planning this project. Vincent and Jidesh (2023) marry Bayesian surrogates with evolutionary search and report a 28% compute savings—an attractive promise when you are staring at an electricity bill. Raiaan et al. (2024) provide the most complete map I found, categorising methods and nudging me toward EAs for mixed discrete–continuous spaces.

Other authors breathed new life into familiar algorithms. Loshchilov and Hutter (2016) adapted CMA-ES for tangled hyperparameter landscapes, and Real et al. (2019) drove Regularized Evolution hard enough to uncover ImageNet-ready CNNs. Multi-objective angles, such as Assunção et al. (2022) and Dutta et al. (2021), reminded me to log runtime alongside accuracy. Meanwhile, industrial systems like AWS SageMaker and Google Vertex AI (Amazon Web Services, 2023) quietly folded evolutionary ideas into their AutoML stacks, signalling that these approaches are no longer just research curiosities.

### 2.4 Representative Studies on Evolutionary Hyperparameter Optimization

| Study | Algorithm | Dataset | Key Findings |
| --- | --- | --- | --- |
| Orive et al. (2014) | GA | Chemical regression / ANN | Evolutionary initialization improved stability over random baselines. |
| Vincent & Jidesh (2023) | Hybrid (BO + EA) | MNIST, CIFAR-10 | Hybrid EA–Bayesian workflow raised accuracy and cut compute cost. |
| Real et al. (2019) | Regularized Evolution | ImageNet | Discovered CNN architectures rivaling human designs (2000 TPU-hrs). |
| Assunção et al. (2022) | Multi-objective PSO | CNN benchmarks | PSO simultaneously optimized accuracy and training time. |
| Dutta et al. (2021) | Differential Evolution | CIFAR-10 | DE maintained stable tuning under noisy validation signals. |
| Li et al. (2023) | Benchmark Survey | Multiple | Called for standardized, resource-aware HPO benchmarks. |

### 2.5 Gaps and Research Motivation

Despite progress, four gaps kept showing up as I planned my own experiments:

1. Comparative studies that evaluate multiple EA families under identical protocols are still rare.
2. Baseline strategies (grid and random search) look different from paper to paper, making comparisons muddy.
3. Integration with production MLOps stacks is usually hand-waved rather than documented.
4. Many studies focus on initialization tricks instead of tuning the training phase itself.

This project tackles each shortfall with a single, controlled benchmark that I could run (and rerun) on my workstation.

### 2.6 Research Contribution

This research will respond to those gaps by:

- Delivering a standardized cross-family comparison of GA, DE, and PSO on MNIST and CIFAR-10 that can be executed on readily available hardware.
- Benchmarking every evolutionary approach against grid and random search so the baselines stay honest.
- Leaning on open frameworks (PyTorch, DEAP) to keep the workflow reproducible for anyone who wants to poke at the code.
- Analysing convergence behaviour, runtime, and robustness under the kind of compute limits most students and practitioners face.

Once the experiments are complete, the resulting empirical baseline can be stress-tested or extended toward hybrid and multi-objective variants.

## 3. Methodology

I will run a head-to-head comparison of GA, DE, and PSO against grid and random search to see what actually performs best on my hardware. MNIST and CIFAR-10 provide contrasting complexity levels without overwhelming GPU memory. The implementation will use Python 3.10, with PyTorch 2.0 handling training and DEAP orchestrating the evolutionary operators.

The shared search space covered:

- Learning rate
- Batch size
- Dropout rate
- Number of hidden layers / filters
- Optimizer type

Each algorithm will explore identical architectures to keep the comparison honest. Early stopping will trigger on validation loss, and the best checkpoint will be evaluated on the hold-out test set. Every configuration will be trained three times with seeds 42, 123, and 456 to reduce the seed correlation issues commonly reported when using sequential seeds (1, 2, 3).

### 3.1 Experimental Design

All runs will execute on my 2020 MacBook Pro (Apple M1 Pro, 32 GB unified memory, 16-core GPU/Neural Engine). Wall-clock runtimes and resource utilisation will be logged per trial to capture the actual cost distribution across training and orchestration steps.
 
**Algorithm configuration (planned)**

- **GA:** Population 20, generations 30, crossover probability 0.7, mutation probability 0.2, tournament size 3. These settings follow common practice in DEAP tutorials and prior comparative studies; adjustments will be documented if early monitoring reveals premature convergence or excessive drift.
- **DE:** Scaling factor F = 0.8 and crossover CR = 0.9, aligning with recommendations from Dutta et al. (2021). Alternative scalings (e.g., F = 0.6) will be explored only if the initial configuration under-performs.
- **PSO:** 20 particles with velocity limits [–1.5, 1.5] and an inertia weight linearly decaying from 0.9 to 0.5. Velocity clamping is included up front to prevent invalid configurations, a failure mode frequently reported in practitioner forums and prior PSO HPO case studies.

**Model architectures**

- **MNIST:** Fully connected network with up to three hidden layers (64–512 neurons), ReLU activations, and dropout in [0, 0.5].
- **CIFAR-10:** Convolutional network featuring two to four convolutional layers (32–128 filters), 3×3 or 5×5 kernels, and dropout in [0, 0.4].

Datasets were split 80/10/10 for training, validation, and testing.

## 4. Results and Discussion

The experiments are scheduled but not yet complete. This section therefore outlines the planned analysis and provides placeholders that will be populated with empirical values once the runs finish.

**Table 4.1 – Planned Reporting Template**

| Method | Dataset | Mean Accuracy (%) | Std Dev (%) | Runtime (hrs) |
| --- | --- | --- | --- | --- |
| GA | MNIST | TBD | TBD | TBD |
| DE | MNIST | TBD | TBD | TBD |
| PSO | MNIST | TBD | TBD | TBD |
| Random Search | MNIST | TBD | TBD | TBD |
| Grid Search | MNIST | TBD | TBD | TBD |
| GA | CIFAR-10 | TBD | TBD | TBD |
| DE | CIFAR-10 | TBD | TBD | TBD |
| PSO | CIFAR-10 | TBD | TBD | TBD |
| Random Search | CIFAR-10 | TBD | TBD | TBD |
| Grid Search | CIFAR-10 | TBD | TBD | TBD |

Once metrics are collected, Welch’s t-test with Holm-Bonferroni correction will be applied to compare each evolutionary algorithm against the baselines per dataset. Additional diagnostics—such as convergence curves, variance analyses, and any notable outliers—will be documented in this section to contextualise the raw numbers.

**Alignment with research questions (to be updated post-experiments)**

1. *RQ1 – Accuracy within fixed budgets:* Determine whether GA, DE, and PSO achieve statistically significant accuracy improvements over grid and random search under matched compute budgets.
2. *RQ2 – Convergence behaviour across datasets:* Compare learning curves and stability across MNIST and CIFAR-10 to see how quickly each method approaches its peak performance and whether stagnation occurs.
3. *RQ3 – Practical engineering hurdles:* Record implementation notes (e.g., parameter sensitivity, invalid configurations, orchestration overhead) encountered during the runs to inform practitioners.

## 5. Threats to Validity (anticipated)

- **Internal validity:** Multiple seeds and fixed data splits are planned to counter stochastic variation, but additional repetitions may be required if variance remains high.
- **Construct validity:** Accuracy will be the primary metric; runtime, energy, and parameter counts will also be logged so that future revisions can incorporate a multi-objective perspective if needed.
- **External validity:** The study targets image-classification workloads on a single consumer laptop GPU. Results may shift for larger transformer models or distributed environments.
- **Implementation reliability:** Unified memory pressure and Metal driver warnings are known risks on the M1 platform. Mitigation steps (batch-size adjustment, population caps) will be recorded if they become necessary during execution.

## 6. Conclusions and Practical Recommendations (to be finalised)

The conclusions below describe the analyses that will be conducted once empirical results are available. They are intentionally provisional and will be replaced by evidence-backed statements after experimentation.

**Planned takeaways**

1. Assess whether random search remains a useful baseline when orchestration time is tight.
2. Examine PSO’s behaviour under time-boxed studies, including whether stagnation checks are necessary to maintain progress.
3. Evaluate GA’s repeatability across runs and determine if it provides the most stable path to quality models.
4. Measure DE’s tolerance to noisy settings, especially when runs are scheduled on heterogeneous or resource-constrained machines.

Operational considerations—such as integration with Airflow or Kubernetes, adaptive population sizing, multi-objective scoring, and AutoML hooks—will be documented alongside the final metrics so that future readers can replicate both the results and the surrounding engineering decisions.

Once the experiments complete, this section will explicitly answer the research questions by citing observed accuracy improvements (RQ1), convergence and stability patterns (RQ2), and practical engineering lessons (RQ3).

## References

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD Conference*, 2623–2631.

Amazon Web Services. (2023). *Amazon SageMaker automatic model tuning*. AWS Documentation.

Assunção, F., Lourenço, N., & Machado, P. (2022). Multi-objective particle swarm optimization for deep learning hyperparameters. *Neurocomputing, 509*, 68–81.

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*, 281–305.

Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2013). Algorithms for hyper-parameter optimization. *Advances in Neural Information Processing Systems, 25*, 2546–2554.

Dutta, S., Chakraborty, D., & Das, S. (2021). Differential evolution for neural-network hyperparameter tuning under uncertainty. *Applied Soft Computing, 110*, 107631.

Golovin, D., Solnik, B., Moitra, S., Kochanski, G., Karro, J., & Sculley, D. (2017). Google Vizier: A service for black-box optimization. *Proceedings of the 23rd ACM SIGKDD Conference*, 1487–1495.

Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, J., & Stoica, I. (2018). Tune: A research platform for distributed model selection and training. *Proceedings of ICML, 87*, 502–514.

Loshchilov, I., & Hutter, F. (2016). CMA-ES for hyperparameter optimization of deep neural networks. *arXiv preprint arXiv:1604.07269*.

Miikkulainen, R., Liang, J., Meyerson, E., Rawal, A., Fink, D., Francon, O., & Hodjat, B. (2019). Evolving deep neural networks. In *Artificial Intelligence in the Age of Neural Networks and Brain Computing* (pp. 293–312). Academic Press.

Montana, D. J., & Davis, L. (1989). Training feedforward neural networks using genetic algorithms. *Proceedings of the 11th International Joint Conference on Artificial Intelligence*, 762–767.

Orive, D., Sorrosal, G., Borges, C., Martin, C., & Alonso-Vicario, A. (2014). Evolutionary algorithms for hyperparameter tuning on neural network models. *Proceedings of the European Modelling and Simulation Symposium*, 402–409.

Raiaan, M. A. K., et al. (2024). A systematic review of hyperparameter optimization. *Artificial Intelligence Review*.

Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized evolution for image classifier architecture search. *Proceedings of the AAAI Conference on Artificial Intelligence, 33*(1), 4780–4789.

Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine-learning algorithms. *Advances in Neural Information Processing Systems, 25*, 2951–2959.

Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation, 10*(2), 99–127.

Vincent, A. M., & Jidesh, P. (2023). An improved hyperparameter optimization framework for AutoML systems using evolutionary algorithms. *Scientific Reports, 13*, 4737.
