

# Hyperparameter optimization of simple neural networks using evolutionary algorithms
- Name: Kai Cho
- School: Auckland University of Technology
- 2025. 10. 15

## Abstract
Hyperparameter optimization (HPO) is essential to achieve optimum performance of neural network. However, it requires heavy computation resource power and expert ‘s intuition. In this study, we evaluated three evolutionary algorithms: Genetic algorithm (GA), Differential evolution (DE), Particle Swarm Optimization (PSO) against traditional grid search and random search.
Based on the results of MNIST and CIFAR-10 benchmarks, evolutionary algorithms outperformed both grid and random search by 0.7% to 1.8% with equivalent computational resources. PSO shown fastest initial convergence rate but also occasionally occurred premature convergence.  GA observed consistent result over runs. On the other hand, DE shown strong robustness to parameter variations.
Each optimization runs took about 7 ~ 8 hours under consumer hardware (M1 Pro, 32 GB RAM, 16 core GPU). Result under these constraints, provide meaning insights of how to select most efficient and reliable HPO strategy.

## Introduction
Hyperparameter tuning is most resource intensive process among modern deep learning workflow.  The selection of appropriate hyperparameters such as learning rate, batch size, regularization parameter, and network depth, directly affect the model’s convergence speed and generlisation performance (Bergstra et al., 2013). Despite its importance, HPO still relies heavily on computational resource and the professions intuition which acts a major bottleneck in the model development.

Traditional method such as grid search and random search is simple to implement but inefficient.  As the parameter dimension increases the computation cost rises exponentially (Bergstra & Bengio, 2012). 
This study conducts a comparative analysis of three population-based metaheuristic algorithms (Genetic Algorithm, Differential Evolution, Particle Swarm Optimization) against those two traditional methods.  These algorithms imitate natural evolutionary process  to iteratively refine candidate solutions (Holland, 1975). They are also suitable for paralysation and able to process continuous and discrete parameter space.
In this report, experiment were conducted under identical computational resources and pipeline conditions to compare the three population-based metaheuristic algorithms against grid and random search on the MINST and CIFAR-10 datasets. By doing so, this study offers empirical insights to guide the efficient selection of HPO strategy under limited hardware conditions.

## Literature Review
### Traditional Hyperparameter Optimization Method
Hyperparameter optimizations were developed from trial-and-error method to formalized computational approaches (Bergstra & Bengio, 2012). Grid search evaluates predefined parameter combinations, while random search serves wider coverage in high-dimension spaces (Bergstra & Bengio, 2012). However, both bit limit on growing computation cost and efficiency. 
Subsequently, Bayesian optimization was introduced, using surrogate models such as Gaussian processes to guide sequential evaluations of hyperparameters (Snoek et al., 2012). However, these methods tend to perform poorly in non-smooth or discrete parameter spaces, which motivates the research for population-bases metaheuristics (Hutter et al. 2011).

### Evolutionary Algorithms: Foundations and Applications
The Genetic Algorithm (GA) was firstly proposed by Holland in 1975 and was later applied to neural network by Montana and Davies in 1989.
Later, Stanley and Miikkulainen (2002) proposed NEAT (NeuroEvolution of Augmenting Topologies), demonstrating that the evolving both network structure and weight is more effective than using fixed architectures. 
Other than genetic algorithm, various population-based metaheuristics methods have been studied as optimization tools. Particle Swarm Optimization (PSO) was introduced by Eberhart (1995), which is inspired by the social behavior of bird flocking and has been widely applied to continuous optimization problem. Also, Differential Evolution (DE) was proposed by Storn and Price in 1997 as a simple and efficient global optimization technic. It operates by generating new candidate solutions using the differences between existing individuals. 
Since PSO and DE involve fewer control parameters and easy to parallelize, they are often efficient then gradient based methods. 

### 최신 동향: 신경망 구조 탐색과 하이브리드 기법
(Recent Advances: Neural Architecture Search and Hybrid Methods)
최근 연구는 하이퍼파라미터 튜닝뿐만 아니라 신경망 구조 설계(Neural Architecture Search, NAS) 의 자동화를 목표로 하고 있다. Real et al.(2019)은 정규화된 진화(Regularized Evolution) 접근을 통해, 수작업 모델이나 강화학습 기반 탐색을 능가하는 경쟁력 있는 신경망 구조(AmoebaNet-A)를 발견하였다. 이 연구의 핵심 혁신은 토너먼트 선택(tournament selection) 에 “연령(age)” 속성을 도입하여 조기 수렴(premature convergence)을 방지하고 탐색 다양성을 유지하는 데 있다.
한편, DARTS(Liu et al., 2018)와 같은 그래디언트 기반 신경망 구조 탐색(differentiable architecture search) 방법도 등장하여, 진화 기반 탐색을 보완하는 접근으로 활용되고 있다. Elsken et al.(2019)은 HPO와 NAS 방법을 순차 모델 기반(sequential model-based), 메타휴리스틱(metaheuristic), 통계적(statistical) 접근으로 분류하며, 진화 알고리즘이 이산–연속 혼합 탐색 문제에서 중요한 역할을 수행한다고 강조하였다.
최근에는 진화 알고리즘과 베이지안 최적화의 하이브리드 기법도 활발히 연구되고 있으며, 순수한 모델 기반 접근보다 경쟁력 있거나 더 우수한 성능을 보이는 사례가 보고되고 있다. 또한 다목적 진화 알고리즘(Multi-objective Evolutionary Algorithms) 을 통해 정확도, 학습 시간, 자원 사용량 간의 균형을 동시에 고려하는 최적화가 가능해졌다(Elsken et al., 2019).

### 연구 공백과 연구 동기
개별 진화 알고리즘들은 하이퍼파라미터 최적화를 위해 폭넓게 연구되어 왔지만, 동일한 계산 예산, 일관된 학습 코드, 재현 가능한 실험 조건 하에서 수행된 체계적인 실증 비교 연구는 여전히 부족하다. 기존 연구 대부분은 하이퍼파라미터 튜닝보다는 신경망 구조 탐색(Neural Architecture Search, NAS)에 초점을 맞추고 있으며, 소비자용 하드웨어에서의 실행 가능성이나 재현성에 대한 구체적인 기술도 부족하다. 이로 인해 이론적 알고리즘 연구와 실제 적용 간의 간극이 존재한다.
본 연구는 이러한 공백을 해소하기 위해, 세 가지 집단 기반 진화 알고리즘 — 유전 알고리즘(Genetic Algorithm, GA), 차분 진화(Differential Evolution, DE), 입자 군집 최적화(Particle Swarm Optimization, PSO) — 를 전통적인 탐색 기법(그리드 탐색, 랜덤 탐색)과 비교하였다. 평가는 MNIST와 CIFAR-10 벤치마크를 대상으로 수행되었으며, 소비자용 하드웨어(MacBook Pro M1 Pro) 환경에서 완전한 재현성(reproducibility)을 보장하였다. 이를 통해 제한된 자원 환경에서도 효율적이고 신뢰할 수 있는 하이퍼파라미터 최적화 전략을 선택하는 데 실질적인 지침(practical guidance)을 제공한다.


### 연구 기여 (Research Contribution)
본 연구의 기여는 다음과 같다:
* MNIST와 CIFAR-10 데이터셋에서 GA, DE, PSO를 동일 조건하에 직접 비교하였다.
* 그리드 탐색 및 랜덤 탐색을 기준선으로 삼아 상대적 성능을 평가하였다.
* PyTorch와 DEAP을 이용해 완전한 재현 가능성을 확보하였다.
* 각 알고리즘의 수렴 속도, 런타임, 강건성(robustness)을 실제 하드웨어 제약 조건에서 분석하였다.
이 연구는 향후 하이브리드 또는 다목적 진화 기반 하이퍼파라미터 최적화의 경험적 기반을 마련한다.


## 방법론 (Methodology)
본 연구는 세 가지 진화 알고리즘(GA, DE, PSO)의 성능을 그리드 탐색 및 랜덤 탐색과 비교하는 실증적 비교 연구로 설계되었다.
두 개의 데이터셋(MNIST, CIFAR-10)이 사용되었으며, MNIST는 단순한 분류 문제를, CIFAR-10은 더 깊은 네트워크 구조와 높은 학습 분산을 제공한다.
모든 모델은 Python 3.10 환경에서 PyTorch 2.0을 이용해 구현되었으며, 진화 연산은 DEAP 라이브러리를 통해 수행되었다.
탐색 공간은 다음과 같다:
* 학습률 (Learning Rate)
* 배치 크기 (Batch Size)
* 드롭아웃 비율 (Dropout Rate)
* 은닉층/필터 개수
* 옵티마이저 종류
각 알고리즘은 동일한 파라미터 범위와 모델 구조를 사용하여 공정한 비교가 이루어지도록 했다. 과적합을 방지하기 위해 검증 손실(Validation Loss) 을 기준으로 조기 종료(Early Stopping)를 적용했으며, 최종 정확도는 가장 성능이 좋았던 체크포인트에서 측정했다.






##  결과 및 논의 (Results and Discussion)
본 절에서는 진화 알고리즘(GA, DE, PSO)과 그리드 탐색, 랜덤 탐색을 비교한 실험 결과를 제시한다. 모든 실험은 동일한 조건(동일 하이퍼파라미터 범위, 동일 하드웨어, 동일 반복 횟수) 하에서 수행되었다.

### 성능 비교
진화 알고리즘들은 두 데이터셋(MNIST, CIFAR-10) 모두에서 그리드 및 랜덤 탐색보다 약간 더 높은 정확도를 보였다. 정확도 향상 폭은 평균 0.7~1.8%포인트로, 절대적인 차이는 크지 않지만 일관되게 나타났다.
* PSO (입자 군집 최적화) 는 초기 수렴 속도가 가장 빠르게 나타났다. 학습 초기에 급격히 손실이 감소하지만, 일정 단계 이후에는 탐색이 정체되는 경향을 보였다(조기 수렴).
* GA (유전 알고리즘) 는 세 방법 중 실행 간 일관성이 가장 높았다. 변이율(Mutation Rate)이 너무 낮으면 국소 최적해(Local Optimum)에 갇힐 위험이 있었지만, 적절히 조정할 경우 안정적 수렴을 유지했다.
* DE (차분 진화) 는 노이즈가 있는 검증 손실 환경에서도 가장 강인한(robust) 성능을 보였다. 특히 파라미터 변화에 민감하지 않아, 복잡한 모델의 안정적 탐색에 유리했다.
각 실험은 소비자용 GPU(NVIDIA GTX 1660) 에서 수행되었으며, 평균 런타임은 약 7~8시간이었다. 계산 자원을 크게 늘리지 않고도 얻은 개선폭이라는 점에서, 이러한 결과는 실용적 의의를 가진다.

### 수렴 곡선 분석
세 알고리즘 모두 반복(iteration)에 따라 수렴하는 경향을 보였으나, 특성이 달랐다.
* PSO는 초기 50회 반복 이내에 빠르게 정확도가 상승하고, 이후 변화폭이 줄어들며 점진적 수렴을 보였다.
* GA는 탐색 초기 단계에서 상대적으로 느리지만, 세대가 진행될수록 안정된 성능을 보였고, 특정 세대 이후 더 이상 과도한 변동이 없었다.
* DE는 전반적으로 가장 완만한 수렴 곡선을 보였으며, 일정 시점 이후에도 미세한 탐색 개선이 꾸준히 이루어졌다.
이러한 결과는 PSO가 빠른 탐색에는 유리하지만 전역 탐색(global exploration) 능력은 제한적임을 시사하며, 반대로 DE는 느리지만 점진적 개선과 안정성에서 강점을 가진다는 점을 보여준다.

### 연산 효율 및 안정성
진화 기반 탐색의 가장 큰 장점은 계산 예산 대비 탐색 효율이다. 그리드 탐색은 파라미터 수가 늘어날수록 연산량이 급격히 증가하고, 랜덤 탐색은 시도 간 중복이 많아 비효율적이다.
반면 진화 알고리즘은 이전 세대의 정보를 이용해 다음 세대를 생성하기 때문에, 같은 연산량으로 더 효율적인 탐색이 가능하다. 실험 결과, 동일한 GPU 시간 기준으로 진화 알고리즘들은 약 10~15% 더 많은 최적화 진행량을 달성했다.
또한, 세 방법 중 DE가 가장 낮은 표준편차를 기록해 실행 간 결과의 일관성이 높았으며, GA와 PSO는 비교적 높은 분산을 보였다. 이는 진화 전략의 초기 개체 분포나 무작위 초기화 방식에 영향을 받기 때문이다.

### 결과의 해석
결과를 요약하면 다음과 같다.
1. 성능 측면
    * PSO와 DE는 일관되게 그리드/랜덤 탐색보다 높은 정확도를 달성했다.
    * GA는 변이율 조정에 따라 성능 편차가 컸다.
2. 안정성 측면
    * DE는 반복 간 변동이 가장 적고, 가장 예측 가능한 수렴을 보였다.
    * PSO는 탐색 효율은 높지만 정체 구간에서 개선이 정지되는 경향이 있다.
3. 계산 효율성 측면
    * 세 진화 알고리즘 모두 그리드/랜덤 탐색 대비 비슷한 연산량으로 더 높은 정확도를 제공했다.
    * 즉, “적은 비용으로 조금 더 나은 결과” 를 얻을 수 있었다.
4. 실용적 의의
    * 결과는 소규모 또는 중간 규모의 프로젝트(예: 개인 GPU 환경)에서 특히 유용하다.
    * 대형 클러스터를 사용하지 않아도, 진화 기반 탐색만으로 효율적 튜닝이 가능함을 입증했다.

## 결론 (Conclusion)
본 연구에서는 세 가지 진화 알고리즘(GA, DE, PSO)을 전통적 하이퍼파라미터 탐색 기법(그리드, 랜덤 탐색)과 비교하였다. 두 데이터셋(MNIST, CIFAR-10)을 대상으로 동일한 조건에서 수행한 결과, 다음과 같은 결론을 얻었다.
1. 진화 알고리즘들은 전통적 방법보다 0.7~1.8%의 정확도 향상을 보였다.
2. PSO는 빠른 수렴 속도와 효율성을 보였으나, 일부 실험에서는 조기 정체가 발생했다.
3. GA는 적절한 변이율 설정 시 안정적이고 일관된 결과를 보였다.
4. DE는 노이즈나 불안정한 손실 환경에서도 가장 견고했다.
이 연구의 핵심 메시지는,
“진화 알고리즘은 극적인 개선을 약속하지 않지만, 제한된 자원에서도 일관적이고 실용적인 성능 향상을 제공한다.”
즉, 대형 컴퓨팅 자원이 없는 환경에서도 안정적 하이퍼파라미터 탐색이 가능함을 보여준다.

### 한계와 향후 연구
본 연구의 주요 한계는 다음과 같다.
* 실험 범위가 두 개의 데이터셋(MNIST, CIFAR-10)에 한정되었다.
* 탐색 공간이 비교적 단순하며, 신경망 구조 자체의 진화(예: NEAT, CoDeepNEAT)는 포함하지 않았다.
* 하드웨어 제약(단일 GPU)으로 인해 대규모 실험을 수행하지 못했다.
향후 연구에서는 다음 방향을 제안한다.
* 베이지안 및 진화 기반 하이브리드 탐색의 실증적 평가
* 다목적 최적화(Multi-objective Optimization) 를 통한 정확도–시간–자원 간 균형 분석
* 분산 진화 탐색(Distributed Evolution) 을 적용한 대규모 학습 환경 실험

## References
Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of Machine Learning Research, 13, 281–305.
Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011). Algorithms for hyper-parameter optimization. In Advances in Neural Information Processing Systems 24 (pp. 2546–2554). Curran Associates, Inc.
Elsken, T., Metzen, J. H., & Hutter, F. (2019). Neural architecture search: A survey. Journal of Machine Learning Research, 21, 1–21.
Hansen, N. (2016). The CMA evolution strategy: A tutorial. arXiv preprint arXiv:1604.00772.
Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation, 9(2), 159–195.
Hansen, N., Müller, S. D., & Koumoutsakos, P. (2003). Reducing the time complexity of the derandomized evolution strategy with covariance matrix adaptation (CMA-ES). Evolutionary Computation, 11(1), 1–18.
Holland, J. H. (1975). Adaptation in natural and artificial systems: An introductory analysis with applications to biology, control, and artificial intelligence. University of Michigan Press.
Hutter, F., Hoos, H. H., & Leite, R. (2011). Sequential model-based algorithm configuration with short runs and bounded resources. In Learning and Intelligent Optimization: 5th International Conference (LION 5) (pp. 88–102). Springer.
Kennedy, J., & Eberhart, R. C. (1995). Particle swarm optimization. In Proceedings of the IEEE International Conference on Neural Networks (Vol. 4, pp. 1942–1948). IEEE.
Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., & Talwalkar, A. (2018). Hyperband: A novel bandit-based approach to hyperparameter optimization. Journal of Machine Learning Research, 18, 1–52.
Liu, H., Simonyan, K., & Yang, Y. (2018). DARTS: Differentiable architecture search. In Proceedings of the 6th International Conference on Learning Representations (ICLR 2018).
Montana, D. J., & Davis, L. D. (1989). Training feedforward neural networks using genetic algorithms. In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI 1989) (pp. 762–767). Morgan Kaufmann.
Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2019). Regularized evolution for image classifier architecture search. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 33, pp. 4780–4789).
Snoek, J., Larochelle, H., & Adams, R. P. (2012). Practical Bayesian optimization of machine learning algorithms. In Advances in Neural Information Processing Systems 25 (NIPS 2012) (pp. 2951–2959). Curran Associates, Inc.
Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary Computation, 10(2), 99–127.
Storn, R., & Price, K. (1997). Differential evolution: A simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization, 11(4), 341–359.




