import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Create output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

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

# Load both datasets
mnist_data = load_results('mnist')
cifar10_data = load_results('cifar10')

# ============================================================================
# FIGURE 1: Box Plot Comparison - Both Datasets
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MNIST
mnist_results = []
mnist_labels = []
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    accuracies = [run['best_fitness'] for run in mnist_data[algo]['runs']]
    mnist_results.append(accuracies)
    mnist_labels.append(algo)

bp1 = ax1.boxplot(mnist_results, labels=mnist_labels, patch_artist=True,
                   showmeans=True, meanline=True)
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax1.set_title('MNIST Performance Distribution', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(97.5, 98.6)

# CIFAR-10
cifar10_results = []
cifar10_labels = []
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    accuracies = [run['best_fitness'] for run in cifar10_data[algo]['runs']]
    cifar10_results.append(accuracies)
    cifar10_labels.append(algo)

bp2 = ax2.boxplot(cifar10_results, labels=cifar10_labels, patch_artist=True,
                   showmeans=True, meanline=True)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax2.set_title('CIFAR-10 Performance Distribution', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(74, 84)

plt.tight_layout()
plt.savefig(output_dir / 'figure1_boxplot_comparison.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure1_boxplot_comparison.pdf', bbox_inches='tight')
print("✓ Figure 1 saved: Box plot comparison")

# ============================================================================
# FIGURE 2: Mean Performance with Error Bars
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Calculate statistics for MNIST
mnist_stats = {}
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    accuracies = [run['best_fitness'] for run in mnist_data[algo]['runs']]
    mnist_stats[algo] = {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'sem': stats.sem(accuracies)
    }

# MNIST Bar Chart
algos = list(mnist_stats.keys())
means = [mnist_stats[a]['mean'] for a in algos]
stds = [mnist_stats[a]['std'] for a in algos]

bars1 = ax1.bar(algos, means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.errorbar(algos, means, yerr=stds, fmt='none', ecolor='black', 
             capsize=5, capthick=2, linewidth=1.5)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars1, means, stds)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
             f'{mean:.2f}%\n±{std:.2f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylabel('Mean Accuracy (%)', fontweight='bold')
ax1.set_title('MNIST: Mean Performance with Standard Deviation', fontweight='bold')
ax1.set_ylim(97.5, 99)
ax1.grid(axis='y', alpha=0.3)

# Calculate statistics for CIFAR-10
cifar10_stats = {}
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    accuracies = [run['best_fitness'] for run in cifar10_data[algo]['runs']]
    cifar10_stats[algo] = {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'sem': stats.sem(accuracies)
    }

# CIFAR-10 Bar Chart
means = [cifar10_stats[a]['mean'] for a in algos]
stds = [cifar10_stats[a]['std'] for a in algos]

bars2 = ax2.bar(algos, means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.errorbar(algos, means, yerr=stds, fmt='none', ecolor='black',
             capsize=5, capthick=2, linewidth=1.5)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars2, means, stds)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.3,
             f'{mean:.2f}%\n±{std:.2f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Mean Accuracy (%)', fontweight='bold')
ax2.set_title('CIFAR-10: Mean Performance with Standard Deviation', fontweight='bold')
ax2.set_ylim(74, 85)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'figure2_mean_performance.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure2_mean_performance.pdf', bbox_inches='tight')
print("✓ Figure 2 saved: Mean performance with error bars")

# ============================================================================
# FIGURE 3: Performance Improvement over Baselines
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MNIST improvements
mnist_random_mean = mnist_stats['Random']['mean']
mnist_grid_mean = mnist_stats['Grid']['mean']
mnist_improvements = {
    'GA': {
        'vs_random': mnist_stats['GA']['mean'] - mnist_random_mean,
        'vs_grid': mnist_stats['GA']['mean'] - mnist_grid_mean
    },
    'PSO': {
        'vs_random': mnist_stats['PSO']['mean'] - mnist_random_mean,
        'vs_grid': mnist_stats['PSO']['mean'] - mnist_grid_mean
    },
    'DE': {
        'vs_random': mnist_stats['DE']['mean'] - mnist_random_mean,
        'vs_grid': mnist_stats['DE']['mean'] - mnist_grid_mean
    }
}

x = np.arange(len(mnist_improvements))
width = 0.35

vs_random = [mnist_improvements[a]['vs_random'] for a in ['GA', 'PSO', 'DE']]
vs_grid = [mnist_improvements[a]['vs_grid'] for a in ['GA', 'PSO', 'DE']]

bars1 = ax1.bar(x - width/2, vs_random, width, label='vs Random', 
                color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, vs_grid, width, label='vs Grid',
                color='#3498db', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.3f}%',
                ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=9, fontweight='bold')

ax1.set_ylabel('Accuracy Improvement (%)', fontweight='bold')
ax1.set_title('MNIST: Improvement Over Baselines', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['GA', 'PSO', 'DE'])
ax1.legend()
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax1.grid(axis='y', alpha=0.3)

# CIFAR-10 improvements
cifar10_random_mean = cifar10_stats['Random']['mean']
cifar10_grid_mean = cifar10_stats['Grid']['mean']
cifar10_improvements = {
    'GA': {
        'vs_random': cifar10_stats['GA']['mean'] - cifar10_random_mean,
        'vs_grid': cifar10_stats['GA']['mean'] - cifar10_grid_mean
    },
    'PSO': {
        'vs_random': cifar10_stats['PSO']['mean'] - cifar10_random_mean,
        'vs_grid': cifar10_stats['PSO']['mean'] - cifar10_grid_mean
    },
    'DE': {
        'vs_random': cifar10_stats['DE']['mean'] - cifar10_random_mean,
        'vs_grid': cifar10_stats['DE']['mean'] - cifar10_grid_mean
    }
}

vs_random = [cifar10_improvements[a]['vs_random'] for a in ['GA', 'PSO', 'DE']]
vs_grid = [cifar10_improvements[a]['vs_grid'] for a in ['GA', 'PSO', 'DE']]

bars1 = ax2.bar(x - width/2, vs_random, width, label='vs Random',
                color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, vs_grid, width, label='vs Grid',
                color='#3498db', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

ax2.set_ylabel('Accuracy Improvement (%)', fontweight='bold')
ax2.set_title('CIFAR-10: Improvement Over Baselines', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['GA', 'PSO', 'DE'])
ax2.legend()
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'figure3_improvements.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure3_improvements.pdf', bbox_inches='tight')
print("✓ Figure 3 saved: Performance improvements")

# ============================================================================
# FIGURE 4: Consistency Analysis (Standard Deviation Comparison)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

datasets = ['MNIST', 'CIFAR-10']
x = np.arange(len(algorithms))
width = 0.35

mnist_stds = [mnist_stats[a]['std'] for a in ['Grid', 'Random', 'GA', 'PSO', 'DE']]
cifar10_stds = [cifar10_stats[a]['std'] for a in ['Grid', 'Random', 'GA', 'PSO', 'DE']]

bars1 = ax.bar(x - width/2, mnist_stds, width, label='MNIST',
               color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, cifar10_stds, width, label='CIFAR-10',
               color='#e67e22', alpha=0.8, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Standard Deviation (%)', fontweight='bold')
ax.set_title('Algorithm Consistency Comparison (Lower is Better)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Grid', 'Random', 'GA', 'PSO', 'DE'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'figure4_consistency.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure4_consistency.pdf', bbox_inches='tight')
print("✓ Figure 4 saved: Consistency analysis")

# ============================================================================
# FIGURE 5: Convergence Behavior (First evaluation vs Best)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MNIST convergence
for algo, color in zip(['GA', 'PSO', 'DE'], ['#2ecc71', '#f39c12', '#9b59b6']):
    convergence = []
    for run in mnist_data[algo]['runs']:
        history = run['evaluation_history']
        if history:
            accuracies = [h['fitness'] for h in history]
            convergence.append(accuracies)
    
    if convergence:
        mean_conv = np.mean(convergence, axis=0)
        std_conv = np.std(convergence, axis=0)
        x_vals = range(1, len(mean_conv) + 1)
        
        ax1.plot(x_vals, mean_conv, marker='o', label=algo, 
                linewidth=2, color=color, markersize=6)
        ax1.fill_between(x_vals, 
                        np.array(mean_conv) - np.array(std_conv),
                        np.array(mean_conv) + np.array(std_conv),
                        alpha=0.2, color=color)

ax1.set_xlabel('Evaluation Number', fontweight='bold')
ax1.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax1.set_title('MNIST: Convergence Behavior', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# CIFAR-10 convergence
for algo, color in zip(['GA', 'PSO', 'DE'], ['#2ecc71', '#f39c12', '#9b59b6']):
    convergence = []
    for run in cifar10_data[algo]['runs']:
        history = run['evaluation_history']
        if history:
            accuracies = [h['fitness'] for h in history]
            convergence.append(accuracies)
    
    if convergence:
        mean_conv = np.mean(convergence, axis=0)
        std_conv = np.std(convergence, axis=0)
        x_vals = range(1, len(mean_conv) + 1)
        
        ax2.plot(x_vals, mean_conv, marker='o', label=algo,
                linewidth=2, color=color, markersize=6)
        ax2.fill_between(x_vals,
                        np.array(mean_conv) - np.array(std_conv),
                        np.array(mean_conv) + np.array(std_conv),
                        alpha=0.2, color=color)

ax2.set_xlabel('Evaluation Number', fontweight='bold')
ax2.set_ylabel('Validation Accuracy (%)', fontweight='bold')
ax2.set_title('CIFAR-10: Convergence Behavior', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'figure5_convergence.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure5_convergence.pdf', bbox_inches='tight')
print("✓ Figure 5 saved: Convergence behavior")

# ============================================================================
# FIGURE 6: Computational Efficiency
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# MNIST efficiency
mnist_times = {}
mnist_accuracies = {}
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    times = [run['time_seconds'] / 3600 for run in mnist_data[algo]['runs']]  # hours
    accs = [run['best_fitness'] for run in mnist_data[algo]['runs']]
    mnist_times[algo] = np.mean(times)
    mnist_accuracies[algo] = np.mean(accs)

algos = list(mnist_times.keys())
times = [mnist_times[a] for a in algos]
accs = [mnist_accuracies[a] for a in algos]

scatter1 = ax1.scatter(times, accs, c=colors, s=300, alpha=0.8, 
                      edgecolors='black', linewidth=2)

for i, algo in enumerate(algos):
    ax1.annotate(algo, (times[i], accs[i]), 
                fontsize=10, fontweight='bold', ha='center', va='center')

ax1.set_xlabel('Average Time (hours)', fontweight='bold')
ax1.set_ylabel('Mean Accuracy (%)', fontweight='bold')
ax1.set_title('MNIST: Accuracy vs Computational Time', fontweight='bold')
ax1.grid(alpha=0.3)

# CIFAR-10 efficiency
cifar10_times = {}
cifar10_accuracies = {}
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    times = [run['time_seconds'] / 3600 for run in cifar10_data[algo]['runs']]
    accs = [run['best_fitness'] for run in cifar10_data[algo]['runs']]
    cifar10_times[algo] = np.mean(times)
    cifar10_accuracies[algo] = np.mean(accs)

times = [cifar10_times[a] for a in algos]
accs = [cifar10_accuracies[a] for a in algos]

scatter2 = ax2.scatter(times, accs, c=colors, s=300, alpha=0.8,
                      edgecolors='black', linewidth=2)

for i, algo in enumerate(algos):
    ax2.annotate(algo, (times[i], accs[i]),
                fontsize=10, fontweight='bold', ha='center', va='center')

ax2.set_xlabel('Average Time (hours)', fontweight='bold')
ax2.set_ylabel('Mean Accuracy (%)', fontweight='bold')
ax2.set_title('CIFAR-10: Accuracy vs Computational Time', fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'figure6_efficiency.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure6_efficiency.pdf', bbox_inches='tight')
print("✓ Figure 6 saved: Computational efficiency")

# ============================================================================
# FIGURE 7: Summary Heatmap
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# MNIST heatmap data
mnist_metrics = []
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    row = [
        mnist_stats[algo]['mean'],
        -mnist_stats[algo]['std'],  # Negative because lower is better
        mnist_times[algo]
    ]
    mnist_metrics.append(row)

mnist_metrics = np.array(mnist_metrics)
# Normalize each column
mnist_normalized = (mnist_metrics - mnist_metrics.min(axis=0)) / (mnist_metrics.max(axis=0) - mnist_metrics.min(axis=0))

sns.heatmap(mnist_normalized.T, annot=mnist_metrics.T, fmt='.2f',
            xticklabels=['Grid', 'Random', 'GA', 'PSO', 'DE'],
            yticklabels=['Mean Acc (%)', 'Consistency', 'Time (hrs)'],
            cmap='RdYlGn', center=0.5, ax=ax1, cbar_kws={'label': 'Normalized Score'})
ax1.set_title('MNIST: Algorithm Performance Summary', fontweight='bold')

# CIFAR-10 heatmap data
cifar10_metrics = []
for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
    row = [
        cifar10_stats[algo]['mean'],
        -cifar10_stats[algo]['std'],
        cifar10_times[algo]
    ]
    cifar10_metrics.append(row)

cifar10_metrics = np.array(cifar10_metrics)
cifar10_normalized = (cifar10_metrics - cifar10_metrics.min(axis=0)) / (cifar10_metrics.max(axis=0) - cifar10_metrics.min(axis=0))

sns.heatmap(cifar10_normalized.T, annot=cifar10_metrics.T, fmt='.2f',
            xticklabels=['Grid', 'Random', 'GA', 'PSO', 'DE'],
            yticklabels=['Mean Acc (%)', 'Consistency', 'Time (hrs)'],
            cmap='RdYlGn', center=0.5, ax=ax2, cbar_kws={'label': 'Normalized Score'})
ax2.set_title('CIFAR-10: Algorithm Performance Summary', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'figure7_summary_heatmap.png', bbox_inches='tight')
plt.savefig(output_dir / 'figure7_summary_heatmap.pdf', bbox_inches='tight')
print("✓ Figure 7 saved: Summary heatmap")

# ============================================================================
# Generate Statistical Summary Table
# ============================================================================
print("\n" + "="*80)
print("STATISTICAL SUMMARY FOR PAPER")
print("="*80)

for dataset_name, data, stats in [('MNIST', mnist_data, mnist_stats), 
                                    ('CIFAR-10', cifar10_data, cifar10_stats)]:
    print(f"\n{dataset_name}:")
    print("-"*80)
    print(f"{'Algorithm':<12} {'Best':<10} {'Mean':<10} {'Std':<10} {'95% CI':<20} {'Runs':<8}")
    print("-"*80)
    
    for algo in ['Grid', 'Random', 'GA', 'PSO', 'DE']:
        accuracies = [run['best_fitness'] for run in data[algo]['runs']]
        best = max(accuracies)
        mean = stats[algo]['mean']
        std = stats[algo]['std']
        ci = 1.96 * stats[algo]['sem']  # 95% confidence interval
        
        print(f"{algo:<12} {best:>6.2f}%    {mean:>6.2f}%    "
              f"{std:>6.3f}%    [{mean-ci:>6.2f}, {mean+ci:>6.2f}]    {len(accuracies):<8}")

print("\n✓ All figures generated successfully in 'figures/' directory")
print("  - PNG format for presentations")
print("  - PDF format for publication")