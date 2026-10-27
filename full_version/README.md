# 🔵 Full (Advanced) Version — 10 In-Depth Programs

Comprehensive solutions (120–250 lines): multi-panel figures, statistical tests,
model diagnostics and detailed printed reports. Everything saves to
`../outputs/advanced/`.

## How to run

```bash
python full_version/q01_advanced_categorical_numerical.py
```

## Extra libraries used (beyond the simple versions)

| Library | Used in | Why |
|---------|---------|-----|
| `scipy.stats` | Q5, Q6, Q7 | Shapiro normality, Pearson r + p-values, QQ plots |
| `statsmodels` | Q7, Q8 | full OLS summary, Cook's distance, lowess, prediction intervals |
| `scikit-learn` | Q7, Q9, Q10 | regression, PCA, t-SNE |
| `umap-learn` | Q10 | UMAP embeddings (graceful fallback if missing) |
| `mpl_toolkits.mplot3d` | Q9 | 3D PCA scatter |

## The 10 programs

| File | Question | Key advanced content | Outputs |
|------|----------|----------------------|---------|
| `q01_advanced_categorical_numerical.py` | Categorical × numerical | grouped bars, overlaid density histograms, gender boxes, group tables | `q01_grouped_bar.png`, `q01_stacked_histogram.png`, `q01_grouped_boxplot.png` |
| `q02_advanced_time_series.py` | Time series | dual y-axes, linear trend, month/week seasonal boxes, peak annotations | `q02_multi_series.png`, `q02_seasonal.png` |
| `q03_advanced_redesign.py` | Misleading redesign | 6 issues, before/after grid, mini fix-demos, report card, Stevens' power law | `q03_misleading.png`, `q03_before_after.png` |
| `q04_advanced_encoding.py` | Encoding theory | 3 encodings side by side, Cleveland–McGill ranking, redesign | `q04_position_vs_size.png`, `q04_redesign.png` |
| `q05_advanced_eda.py` | EDA | hist+KDE, mean markers, violin+box, QQ plots, skew/kurtosis table, IQR vs z-score | `q05_combined_eda.png`, `q05_outlier_detection.png` |
| `q06_advanced_correlation.py` | Correlation deep dive | top-3 regression scatters, KDE pair plot, significance-annotated heatmap | `q06_advanced_scatter.png`, `q06_pairplot.png`, `q06_annotated_heatmap.png` |
| `q07_advanced_regression.py` | Regression diagnostics | sklearn + statsmodels OLS, 2×2 diagnostics, Cook's distance, CI vs PI | `q07_regression_diagnostics.png`, `q07_confidence_prediction.png` |
| `q08_advanced_rolling_uncertainty.py` | Uncertainty | SE, 95 % CI bands, error bars, 2/4/6-week window comparison | `q08_uncertainty_band.png`, `q08_error_bars.png` |
| `q09_advanced_pca.py` | PCA full pipeline | all 11 components, scree + Kaiser, cumulative thresholds, biplot, 3D | `q09_scree_plot.png`, `q09_biplot.png` |
| `q10_advanced_tsne_umap.py` | Embedding comparison | perplexity 5/30/50 × n_neighbors 5/15/50 grid, silhouette study | `q10_comparison.png`, `q10_parameter_study.png` |

> The older `exp1…exp10*.py` files are the original lab attempts and still run
> against the legacy CSV in this folder.

## What makes each different from the simple version

| Q | Simple asks… | Advanced adds… |
|---|--------------|----------------|
| 1 | "What does one category look like?" | two-way groups, counts, per-group statistics |
| 2 | "Is the line going up?" | trend slope per week, sample-size caution, seasonality |
| 3 | "Spot 4 mistakes" | 6 mistakes, theory (Stevens, pre-attentive), report card |
| 4 | "Which is easier to read?" | formal Cleveland–McGill ranking + third encoding |
| 5 | "Describe the shape" | normality tests, QQ plots, two outlier methods compared |
| 6 | "Which r is biggest?" | p-values, significance stars, strength labels, causation caveat |
| 7 | "Draw the fit line" | full OLS inference, 4 diagnostics, influential points, CI vs PI |
| 8 | "Smooth the line" | standard error, 95 % CI, window-size sensitivity |
| 9 | "Show 2 PCs" | all components, Kaiser rule, biplot loadings, 3D view |
| 10 | "Run t-SNE once" | parameter sweeps, silhouette comparison, method critique |

## Statistical concepts you can quote in the viva

- **Pearson p-value** — probability of seeing r this large if there were truly no relationship
- **Adjusted R²** — R² penalised for adding useless predictors
- **Cook's distance** — how much the fit would move if that one point were removed
- **Homoscedasticity** — residuals should have constant spread (scale-location plot flat)
- **Kaiser criterion** — keep components with eigenvalue > 1
- **Silhouette score** — how cleanly groups separate in a given 2D view (−1 … +1)
