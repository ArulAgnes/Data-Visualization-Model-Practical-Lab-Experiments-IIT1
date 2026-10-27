# 🟢 Simple Version — 10 Beginner-Friendly Programs

Short, heavily commented scripts (80–120 lines). No classes, no decorators —
every line is explainable in a viva. Each one loads the shared dataset, saves
PNGs to `../outputs/simple/`, and **prints an interpretation**.

## How to run

From the **repository root**:

```bash
python simple_version/q01_bar_histogram_boxplot.py
```

or from inside this folder:

```bash
cd simple_version
python q01_bar_histogram_boxplot.py
```

Both work — every script anchors its paths to its own file location.

## The 10 programs

| File | Question | Outputs (in `outputs/simple/`) |
|------|----------|-------------------------------|
| `q01_bar_histogram_boxplot.py` | Bar, histogram, box plot | `q01_bar_chart.png`, `q01_histogram.png`, `q01_boxplot.png` |
| `q02_time_series_line_chart.py` | Time-series line chart | `q02_time_series.png` |
| `q03_misleading_chart_fix.py` | Misleading chart + fix | `q03_misleading.png`, `q03_fixed.png` |
| `q04_encoding_comparison.py` | Two encodings compared | `q04_encoding_1.png`, `q04_encoding_2.png`, `q04_redesigned.png` |
| `q05_eda_distributions.py` | EDA distributions | `q05_histogram.png`, `q05_boxplot.png`, `q05_density.png`, `q05_violin.png` |
| `q06_scatter_pairplot_heatmap.py` | Scatter / pair / heatmap | `q06_scatter.png`, `q06_pairplot.png`, `q06_heatmap.png` |
| `q07_regression_line.py` | Linear regression | `q07_regression.png`, `q07_residuals.png` |
| `q08_rolling_avg_uncertainty.py` | Rolling average + band | `q08_rolling_avg.png` |
| `q09_pca_visualization.py` | PCA to 2D | `q09_pca_scatter.png`, `q09_variance.png` |
| `q10_tsne_umap.py` | t-SNE + UMAP | `q10_tsne.png`, `q10_umap.png` |

> The older `exp1…exp10*.py` files are the original lab attempts (they read the
> legacy CSV sitting next to them). Keep them for reference — the `q*.py` files
> are the final structured solutions.

## Beginner tips

1. **Read the top docstring first** — every file says what it does in 3 lines.
2. **Run one file, then open the PNG** in `outputs/simple/` and match it to the code.
3. The 4-line block at the top of every script (BASE/CSV/OUT) is just path plumbing.
4. Charts always follow the same rhythm:
   *create figure → plot → title/xlabel/ylabel → tight_layout → savefig → close*.
5. The `print(...)` block at the end is your **spoken answer** for the exam.
6. Uncomment `# plt.show()` to get pop-up windows on your laptop.
7. Stuck on a missing package? `pip install -r requirements.txt`.

## Expected console output (example, Q1)

```
Shape (rows, columns): (1200, 20)
INTERPRETATION
- Section A has the highest average marks (58.6)
- Marks range from 11.5 to 100.0
- Section D shows the most variation in marks (std = 15.5)
```
