# LaTeX Tables for Research Paper

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
\usepackage{booktabs}      % For professional tables
\usepackage{multirow}      % For multirow cells
\usepackage{graphicx}      % For resizebox
```

### Including Tables
Simply use `\input{}` in your document:

```latex
\section{Results}

Table~\ref{tab:main_results} presents the main performance comparison.

\input{tables/table1_main_results}

Our computational efficiency analysis (Table~\ref{tab:computational_efficiency}) shows...

\input{tables/table2_computational_efficiency}
```

### Customization

#### Adjust table width:
```latex
\resizebox{0.9\textwidth}{!}{
  % table content
}
```

#### Change font size:
```latex
\begin{table}[htbp]
\small  % or \footnotesize, \scriptsize
\centering
% rest of table
\end{table}
```

#### Two-column format:
Replace `\begin{table}[htbp]` with `\begin{table*}[htbp]` for full-width tables in two-column documents.

## Tips

- All tables use professional styling with booktabs package
- Best results are highlighted in **bold**
- Statistical significance included where relevant
- Tables include descriptive notes at the bottom
- All tables are publication-ready for academic papers

## Citation Format

When referencing tables in text:
- Table~\ref{tab:main_results}
- Tables~\ref{tab:main_results} and~\ref{tab:computational_efficiency}
- as shown in Table~\ref{tab:main_results}
