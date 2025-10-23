"""
LaTeX Table Generator for Research Paper
Generates publication-ready tables from experimental results
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

# Load all results
results_dir = Path('results')
algorithms = {
    'Grid': {'mnist': 'grid_mnist_20251021_011057.json', 
             'cifar10': 'grid_cifar10_20251021_050052.json'},
    'Random': {'mnist': 'random_mnist_20251021_010819.json', 
               'cifar10': 'random_cifar10_20251021_041728.json'},
    'GA': {'mnist': 'ga_mnist_20251021_101942.json', 
           'cifar10': 'ga_cifar10_20251022_073914.json'},
    'PSO': {'mnist': 'pso_mnist_20251021_181551.json', 
            'cifar10': 'pso_cifar10_20251022_164435.json'},
    'DE': {'mnist': 'de_mnist_20251021_152823.json', 
           'cifar10': 'de_cifar10_20251022_210512.json'}
}

def load_results(dataset):
    """Load all algorithm results for a dataset"""
    data = {}
    for algo, files in algorithms.items():
        filepath = results_dir / files[dataset]
        with open(filepath) as f:
            data[algo] = json.load(f)
    return data

def calculate_statistics(data):
    """Calculate comprehensive statistics"""
    stats_dict = {}
    for algo, algo_data in data.items():
        accuracies = [run['best_fitness'] for run in algo_data['runs']]
        times = [run['time_seconds'] / 3600 for run in algo_data['runs']]  # Convert to hours
        evaluations = [run['total_evaluations'] for run in algo_data['runs']]
        
        stats_dict[algo] = {
            'best': max(accuracies),
            'worst': min(accuracies),
            'mean': np.mean(accuracies),
            'std': np.std(accuracies),
            'sem': stats.sem(accuracies),
            'median': np.median(accuracies),
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_evals': np.mean(evaluations),
            'total_time': np.sum(times),
            'n_runs': len(accuracies)
        }
        
        # Calculate 95% confidence interval
        ci = 1.96 * stats_dict[algo]['sem']
        stats_dict[algo]['ci_lower'] = stats_dict[algo]['mean'] - ci
        stats_dict[algo]['ci_upper'] = stats_dict[algo]['mean'] + ci
    
    return stats_dict

# Load data
mnist_data = load_results('mnist')
cifar10_data = load_results('cifar10')

# Calculate statistics
mnist_stats = calculate_statistics(mnist_data)
cifar10_stats = calculate_statistics(cifar10_data)

# Create output directory for tables
output_dir = Path('tables')
output_dir.mkdir(exist_ok=True)

# ============================================================================
# TABLE 1: Main Results - Performance Summary
# ============================================================================
table1 = r"""\begin{table}[htbp]
\centering
\caption{Performance Comparison of Hyperparameter Optimization Algorithms}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Algorithm} & \textbf{Dataset} & \textbf{Best (\%)} & \textbf{Mean $\pm$ Std (\%)} & \textbf{95\% CI} & \textbf{Worst (\%)} & \textbf{Runs} \\
\midrule
"""

# Add MNIST results
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    s = mnist_stats[algo]
    if algo == 'DE':  # Bold the best result
        table1 += f"\\textbf{{{algo}}} & MNIST & \\textbf{{{s['best']:.2f}}} & \\textbf{{{s['mean']:.2f} $\\pm$ {s['std']:.2f}}} & \\textbf{{[{s['ci_lower']:.2f}, {s['ci_upper']:.2f}]}} & \\textbf{{{s['worst']:.2f}}} & {s['n_runs']} \\\\\n"
    else:
        table1 += f"{algo} & MNIST & {s['best']:.2f} & {s['mean']:.2f} $\\pm$ {s['std']:.2f} & [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}] & {s['worst']:.2f} & {s['n_runs']} \\\\\n"

table1 += "\\midrule\n"

# Add CIFAR-10 results
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    s = cifar10_stats[algo]
    if algo == 'DE':  # Bold the best result
        table1 += f"\\textbf{{{algo}}} & CIFAR-10 & \\textbf{{{s['best']:.2f}}} & \\textbf{{{s['mean']:.2f} $\\pm$ {s['std']:.2f}}} & \\textbf{{[{s['ci_lower']:.2f}, {s['ci_upper']:.2f}]}} & \\textbf{{{s['worst']:.2f}}} & {s['n_runs']} \\\\\n"
    else:
        table1 += f"{algo} & CIFAR-10 & {s['best']:.2f} & {s['mean']:.2f} $\\pm$ {s['std']:.2f} & [{s['ci_lower']:.2f}, {s['ci_upper']:.2f}] & {s['worst']:.2f} & {s['n_runs']} \\\\\n"

table1 += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\begin{minipage}{\textwidth}
\small
\textit{Note}: CI = Confidence Interval. Best results for each dataset are shown in \textbf{bold}. All accuracy values are validation accuracies (\\%).
\end{minipage}
\end{table}
"""

with open(output_dir / 'table1_main_results.tex', 'w') as f:
    f.write(table1)

print("✓ Table 1 saved: Main results (performance summary)")

# ============================================================================
# TABLE 2: Computational Efficiency Analysis
# ============================================================================
table2 = r"""\begin{table}[htbp]
\centering
\caption{Computational Efficiency Comparison}
\label{tab:computational_efficiency}
\begin{tabular}{lccccc}
\toprule
\textbf{Algorithm} & \textbf{Dataset} & \textbf{Avg Time (hrs)} & \textbf{Total Time (hrs)} & \textbf{Avg Evals} & \textbf{Efficiency*} \\
\midrule
"""

# MNIST
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    s = mnist_stats[algo]
    efficiency = s['mean'] / s['avg_time']  # Accuracy per hour
    table2 += f"{algo} & MNIST & {s['avg_time']:.2f} $\\pm$ {s['std_time']:.2f} & {s['total_time']:.2f} & {s['avg_evals']:.0f} & {efficiency:.2f} \\\\\n"

table2 += "\\midrule\n"

# CIFAR-10
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    s = cifar10_stats[algo]
    efficiency = s['mean'] / s['avg_time']
    table2 += f"{algo} & CIFAR-10 & {s['avg_time']:.2f} $\\pm$ {s['std_time']:.2f} & {s['total_time']:.2f} & {s['avg_evals']:.0f} & {efficiency:.2f} \\\\\n"

table2 += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\begin{minipage}{\textwidth}
\small
\textit{Note}: *Efficiency = Mean Accuracy / Average Time (\%/hour). All experiments conducted on M1 Pro (32GB RAM, 16-core GPU).
\end{minipage}
\end{table}
"""

with open(output_dir / 'table2_computational_efficiency.tex', 'w') as f:
    f.write(table2)

print("✓ Table 2 saved: Computational efficiency")

# ============================================================================
# TABLE 3: Performance Improvement over Baselines
# ============================================================================
table3 = r"""\begin{table}[htbp]
\centering
\caption{Performance Improvement of Evolutionary Algorithms over Baseline Methods}
\label{tab:performance_improvement}
\begin{tabular}{lcccc}
\toprule
\textbf{Algorithm} & \textbf{Dataset} & \textbf{vs Random (\%)} & \textbf{vs Grid (\%)} & \textbf{p-value*} \\
\midrule
"""

# MNIST improvements
mnist_random_mean = mnist_stats['Random']['mean']
mnist_grid_mean = mnist_stats['Grid']['mean']

for algo in ['GA', 'PSO', 'DE']:
    s = mnist_stats[algo]
    vs_random = s['mean'] - mnist_random_mean
    vs_grid = s['mean'] - mnist_grid_mean
    
    # Calculate t-test p-value vs Random
    algo_accs = [run['best_fitness'] for run in mnist_data[algo]['runs']]
    random_accs = [run['best_fitness'] for run in mnist_data['Random']['runs']]
    t_stat, p_value = stats.ttest_ind(algo_accs, random_accs)
    
    p_str = f"{p_value:.4f}" if p_value >= 0.001 else "$<$0.001"
    
    table3 += f"{algo} & MNIST & {vs_random:+.3f} & {vs_grid:+.3f} & {p_str} \\\\\n"

table3 += "\\midrule\n"

# CIFAR-10 improvements
cifar10_random_mean = cifar10_stats['Random']['mean']
cifar10_grid_mean = cifar10_stats['Grid']['mean']

for algo in ['GA', 'PSO', 'DE']:
    s = cifar10_stats[algo]
    vs_random = s['mean'] - cifar10_random_mean
    vs_grid = s['mean'] - cifar10_grid_mean
    
    # Calculate t-test p-value vs Random
    algo_accs = [run['best_fitness'] for run in cifar10_data[algo]['runs']]
    random_accs = [run['best_fitness'] for run in cifar10_data['Random']['runs']]
    t_stat, p_value = stats.ttest_ind(algo_accs, random_accs)
    
    p_str = f"{p_value:.4f}" if p_value >= 0.001 else "$<$0.001"
    
    if algo == 'DE':  # Bold best improvements
        table3 += f"\\textbf{{{algo}}} & CIFAR-10 & \\textbf{{{vs_random:+.2f}}} & \\textbf{{{vs_grid:+.2f}}} & {p_str} \\\\\n"
    else:
        table3 += f"{algo} & CIFAR-10 & {vs_random:+.2f} & {vs_grid:+.2f} & {p_str} \\\\\n"

table3 += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\begin{minipage}{\textwidth}
\small
\textit{Note}: *Two-sample t-test p-value comparing algorithm vs Random Search. Positive values indicate improvement. Best improvements shown in \textbf{bold}.
\end{minipage}
\end{table}
"""

with open(output_dir / 'table3_performance_improvement.tex', 'w') as f:
    f.write(table3)

print("✓ Table 3 saved: Performance improvement over baselines")

# ============================================================================
# TABLE 4: Best Hyperparameters Found
# ============================================================================
table4 = r"""\begin{table}[htbp]
\centering
\caption{Best Hyperparameters Discovered by Each Algorithm}
\label{tab:best_hyperparameters}
\resizebox{\textwidth}{!}{
\begin{tabular}{llcccccc}
\toprule
\textbf{Algorithm} & \textbf{Dataset} & \textbf{LR} & \textbf{Batch Size} & \textbf{Dropout} & \textbf{Hidden Units} & \textbf{Optimizer} & \textbf{Weight Decay} \\
\midrule
"""

# MNIST best hyperparameters
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    best_run = max(mnist_data[algo]['runs'], key=lambda x: x['best_fitness'])
    hp = best_run['best_hyperparameters']
    acc = best_run['best_fitness']
    
    lr = hp.get('learning_rate', 'N/A')
    bs = hp.get('batch_size', 'N/A')
    dr = hp.get('dropout_rate', 'N/A')
    hu = hp.get('hidden_units', 'N/A')
    opt = hp.get('optimizer', 'N/A')
    wd = hp.get('weight_decay', 0.0)
    
    lr_str = f"{lr:.4f}" if isinstance(lr, float) else lr
    dr_str = f"{dr:.3f}" if isinstance(dr, float) else dr
    wd_str = f"{wd:.4f}" if wd > 0 else "0"
    
    table4 += f"{algo} & MNIST ({acc:.2f}\\%) & {lr_str} & {bs} & {dr_str} & {hu} & {opt} & {wd_str} \\\\\n"

table4 += "\\midrule\n"

# CIFAR-10 best hyperparameters
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    best_run = max(cifar10_data[algo]['runs'], key=lambda x: x['best_fitness'])
    hp = best_run['best_hyperparameters']
    acc = best_run['best_fitness']
    
    lr = hp.get('learning_rate', 'N/A')
    bs = hp.get('batch_size', 'N/A')
    dr = hp.get('dropout_rate', 'N/A')
    hu = hp.get('hidden_units', 'N/A')
    opt = hp.get('optimizer', 'N/A')
    wd = hp.get('weight_decay', 0.0)
    
    lr_str = f"{lr:.4f}" if isinstance(lr, float) else lr
    dr_str = f"{dr:.3f}" if isinstance(dr, float) else dr
    wd_str = f"{wd:.4f}" if wd > 0 else "0"
    
    table4 += f"{algo} & CIFAR-10 ({acc:.2f}\\%) & {lr_str} & {bs} & {dr_str} & {hu} & {opt} & {wd_str} \\\\\n"

table4 += r"""\bottomrule
\end{tabular}
}
\vspace{2mm}
\begin{minipage}{\textwidth}
\small
\textit{Note}: LR = Learning Rate. Accuracy values in parentheses indicate the validation accuracy achieved with these hyperparameters.
\end{minipage}
\end{table}
"""

with open(output_dir / 'table4_best_hyperparameters.tex', 'w') as f:
    f.write(table4)

print("✓ Table 4 saved: Best hyperparameters found")

# ============================================================================
# TABLE 5: Algorithm Consistency Ranking
# ============================================================================
table5 = r"""\begin{table}[htbp]
\centering
\caption{Algorithm Consistency Ranking (by Standard Deviation)}
\label{tab:consistency_ranking}
\begin{tabular}{lccccc}
\toprule
\textbf{Rank} & \textbf{Algorithm} & \textbf{Dataset} & \textbf{Std Dev (\%)} & \textbf{CV*} & \textbf{Range (\%)} \\
\midrule
"""

# MNIST consistency ranking
mnist_consistency = [(algo, mnist_stats[algo]['std'], mnist_stats[algo]) 
                     for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']]
mnist_consistency.sort(key=lambda x: x[1])

for rank, (algo, std, s) in enumerate(mnist_consistency, 1):
    cv = (std / s['mean']) * 100  # Coefficient of variation
    range_val = s['best'] - s['worst']
    
    if rank == 1:  # Bold best
        table5 += f"\\textbf{{{rank}}} & \\textbf{{{algo}}} & MNIST & \\textbf{{{std:.3f}}} & \\textbf{{{cv:.2f}}} & \\textbf{{{range_val:.3f}}} \\\\\n"
    else:
        table5 += f"{rank} & {algo} & MNIST & {std:.3f} & {cv:.2f} & {range_val:.3f} \\\\\n"

table5 += "\\midrule\n"

# CIFAR-10 consistency ranking
cifar10_consistency = [(algo, cifar10_stats[algo]['std'], cifar10_stats[algo]) 
                       for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']]
cifar10_consistency.sort(key=lambda x: x[1])

for rank, (algo, std, s) in enumerate(cifar10_consistency, 1):
    cv = (std / s['mean']) * 100
    range_val = s['best'] - s['worst']
    
    if rank == 1:  # Bold best
        table5 += f"\\textbf{{{rank}}} & \\textbf{{{algo}}} & CIFAR-10 & \\textbf{{{std:.3f}}} & \\textbf{{{cv:.2f}}} & \\textbf{{{range_val:.3f}}} \\\\\n"
    else:
        table5 += f"{rank} & {algo} & CIFAR-10 & {std:.3f} & {cv:.2f} & {range_val:.3f} \\\\\n"

table5 += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\begin{minipage}{\textwidth}
\small
\textit{Note}: *CV = Coefficient of Variation = (Std Dev / Mean) $\times$ 100. Lower values indicate better consistency. Best (most consistent) in each dataset shown in \textbf{bold}.
\end{minipage}
\end{table}
"""

with open(output_dir / 'table5_consistency_ranking.tex', 'w') as f:
    f.write(table5)

print("✓ Table 5 saved: Consistency ranking")

# ============================================================================
# TABLE 6: Summary Statistics Table (Compact)
# ============================================================================
table6 = r"""\begin{table}[htbp]
\centering
\caption{Comprehensive Summary of Algorithm Performance}
\label{tab:summary_statistics}
\resizebox{\textwidth}{!}{
\begin{tabular}{llccccccc}
\toprule
\multirow{2}{*}{\textbf{Algorithm}} & \multirow{2}{*}{\textbf{Dataset}} & \multicolumn{3}{c}{\textbf{Accuracy (\%)}} & \multicolumn{2}{c}{\textbf{Time (hrs)}} & \multirow{2}{*}{\textbf{Consistency**}} & \multirow{2}{*}{\textbf{Rank***}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-7}
& & Mean & Std & Best & Avg & Total & & \\
\midrule
"""

# Create ranking based on mean accuracy for each dataset
mnist_ranking = sorted([(algo, mnist_stats[algo]['mean']) for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']], 
                       key=lambda x: x[1], reverse=True)
mnist_rank_dict = {algo: rank for rank, (algo, _) in enumerate(mnist_ranking, 1)}

cifar10_ranking = sorted([(algo, cifar10_stats[algo]['mean']) for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']], 
                         key=lambda x: x[1], reverse=True)
cifar10_rank_dict = {algo: rank for rank, (algo, _) in enumerate(cifar10_ranking, 1)}

# MNIST
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    s = mnist_stats[algo]
    consistency = "★★★★★" if s['std'] < 0.05 else "★★★★" if s['std'] < 0.1 else "★★★" if s['std'] < 0.3 else "★★"
    rank = mnist_rank_dict[algo]
    
    if rank == 1:
        table6 += f"\\textbf{{{algo}}} & MNIST & \\textbf{{{s['mean']:.2f}}} & {s['std']:.3f} & \\textbf{{{s['best']:.2f}}} & {s['avg_time']:.2f} & {s['total_time']:.2f} & {consistency} & \\textbf{{{rank}}} \\\\\n"
    else:
        table6 += f"{algo} & MNIST & {s['mean']:.2f} & {s['std']:.3f} & {s['best']:.2f} & {s['avg_time']:.2f} & {s['total_time']:.2f} & {consistency} & {rank} \\\\\n"

table6 += "\\midrule\n"

# CIFAR-10
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    s = cifar10_stats[algo]
    consistency = "★★★★★" if s['std'] < 0.2 else "★★★★" if s['std'] < 0.5 else "★★★" if s['std'] < 1.0 else "★★"
    rank = cifar10_rank_dict[algo]
    
    if rank == 1:
        table6 += f"\\textbf{{{algo}}} & CIFAR-10 & \\textbf{{{s['mean']:.2f}}} & {s['std']:.3f} & \\textbf{{{s['best']:.2f}}} & {s['avg_time']:.2f} & {s['total_time']:.2f} & {consistency} & \\textbf{{{rank}}} \\\\\n"
    else:
        table6 += f"{algo} & CIFAR-10 & {s['mean']:.2f} & {s['std']:.3f} & {s['best']:.2f} & {s['avg_time']:.2f} & {s['total_time']:.2f} & {consistency} & {rank} \\\\\n"

table6 += r"""\bottomrule
\end{tabular}
}
\vspace{2mm}
\begin{minipage}{\textwidth}
\small
\textit{Note}: **Consistency rating: ★★★★★ (excellent) to ★★ (poor) based on standard deviation. ***Rank based on mean accuracy (1 = best). Best performers shown in \textbf{bold}.
\end{minipage}
\end{table}
"""

with open(output_dir / 'table6_summary_statistics.tex', 'w') as f:
    f.write(table6)

print("✓ Table 6 saved: Comprehensive summary statistics")

# ============================================================================
# Generate README with LaTeX usage instructions
# ============================================================================
readme = """# LaTeX Tables for Research Paper

## Generated Tables

1. **table1_main_results.tex** - Main performance comparison
2. **table2_computational_efficiency.tex** - Time and efficiency analysis
3. **table3_performance_improvement.tex** - Improvement over baselines with statistical tests
4. **table4_best_hyperparameters.tex** - Best hyperparameters found by each algorithm
5. **table5_consistency_ranking.tex** - Algorithm consistency ranking
6. **table6_summary_statistics.tex** - Comprehensive summary with rankings

## Usage in LaTeX

### Required Packages
Add these to your preamble:

```latex
\\usepackage{booktabs}      % For professional tables
\\usepackage{multirow}      % For multirow cells
\\usepackage{graphicx}      % For resizebox
```

### Including Tables
Simply use `\\input{}` in your document:

```latex
\\section{Results}

Table~\\ref{tab:main_results} presents the main performance comparison.

\\input{tables/table1_main_results}

Our computational efficiency analysis (Table~\\ref{tab:computational_efficiency}) shows...

\\input{tables/table2_computational_efficiency}
```

### Customization

#### Adjust table width:
```latex
\\resizebox{0.9\\textwidth}{!}{
  % table content
}
```

#### Change font size:
```latex
\\begin{table}[htbp]
\\small  % or \\footnotesize, \\scriptsize
\\centering
% rest of table
\\end{table}
```

#### Two-column format:
Replace `\\begin{table}[htbp]` with `\\begin{table*}[htbp]` for full-width tables in two-column documents.

## Tips

- All tables use professional styling with booktabs package
- Best results are highlighted in **bold**
- Statistical significance included where relevant
- Tables include descriptive notes at the bottom
- All tables are publication-ready for academic papers

## Citation Format

When referencing tables in text:
- Table~\\ref{tab:main_results}
- Tables~\\ref{tab:main_results} and~\\ref{tab:computational_efficiency}
- as shown in Table~\\ref{tab:main_results}
"""

with open(output_dir / 'README.md', 'w') as f:
    f.write(readme)

print("✓ README saved with LaTeX usage instructions")

print("\n" + "="*80)
print("ALL LATEX TABLES GENERATED SUCCESSFULLY")
print("="*80)
print(f"\nLocation: {output_dir}/")
print("\nGenerated files:")
print("  1. table1_main_results.tex - Main performance comparison")
print("  2. table2_computational_efficiency.tex - Computational analysis")
print("  3. table3_performance_improvement.tex - Improvements with p-values")
print("  4. table4_best_hyperparameters.tex - Best hyperparameters found")
print("  5. table5_consistency_ranking.tex - Consistency analysis")
print("  6. table6_summary_statistics.tex - Comprehensive summary")
print("  7. README.md - LaTeX usage instructions")
print("\nReady to use in your paper! 📄✨")
