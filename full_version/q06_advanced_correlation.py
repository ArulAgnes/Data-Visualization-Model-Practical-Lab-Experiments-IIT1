"""
Question 6: Correlation Analysis (Deep Dive)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Top correlated pairs with fitted regression lines and annotated
outliers, a pair plot with KDE diagonals, and an annotated heatmap with
Pearson p-values and strength labels. Includes a strong reminder that
correlation is not causation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CSV = os.path.join(ROOT, "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(ROOT, "outputs", "advanced")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")

num_cols = df.select_dtypes(include="number").columns.tolist()
corr = df[num_cols].corr()

# ---------------------------------------------------------------------------
# TOP-3 CORRELATED PAIRS WITH THE TARGET (final_exam_marks)
# ---------------------------------------------------------------------------
target = "final_exam_marks"
others = corr[target].drop(target).abs().sort_values(ascending=False)
top3 = list(others.index[:3])
print("TOP-3 PREDICTORS OF final_exam_marks:")
print(corr[target].drop(target).reindex(top3).round(3).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, top3):
    sns.regplot(x=col, y=target, data=df, ax=ax, scatter_kws={"alpha": 0.4},
                line_kws={"color": "#FF5722"}, ci=95)
    r, p = stats.pearsonr(df[col], df[target])
    # highlight statistical outliers (|z|>3 on both axes) in red
    zx = np.abs((df[col] - df[col].mean()) / df[col].std())
    zy = np.abs((df[target] - df[target].mean()) / df[target].std())
    odd = df[(zx > 3) | (zy > 3)]
    ax.scatter(odd[col], odd[target], s=60, facecolors="none",
               edgecolors="black", linewidths=1.5, label="outliers |z|>3")
    ax.set_title(f"{col} vs {target}\nr={r:.3f}, p={'<0.001' if p < 0.001 else f'{p:.3f}'}",
                 fontsize=11)
    ax.legend(fontsize=8)
fig.suptitle("Top 3 Correlated Pairs with Final Exam Marks", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q06_advanced_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# PAIR PLOT with KDE on the diagonal, coloured by department
# ---------------------------------------------------------------------------
pair_vars = ["study_hours_per_week", "attendance_percentage",
             "lab_marks", "final_exam_marks"]
g = sns.pairplot(df, vars=pair_vars, hue="department", palette="Set1",
                 diag_kind="kde", plot_kws={"alpha": 0.5, "s": 22})
g.figure.suptitle("Pair Plot (KDE diagonal) by Department", y=1.02)
g.savefig(os.path.join(OUT, "q06_pairplot.png"), dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------------------
# ANNOTATED HEATMAP: r values + significance stars
# ---------------------------------------------------------------------------
pvals = pd.DataFrame(np.ones((len(num_cols), len(num_cols))),
                     index=num_cols, columns=num_cols)
for a in num_cols:
    for b in num_cols:
        pvals.loc[a, b] = stats.pearsonr(df[a], df[b])[1]


def strength(r):
    """Label a correlation coefficient by strength."""
    ar = abs(r)
    if ar > 0.6:
        return "S"
    if ar > 0.3:
        return "M"
    return "W"


labels = corr.round(2).astype(str)
for a in num_cols:
    for b in num_cols:
        star = "***" if pvals.loc[a, b] < 0.001 else (
            "**" if pvals.loc[a, b] < 0.01 else (
                "*" if pvals.loc[a, b] < 0.05 else "ns"))
        labels.loc[a, b] = f"{corr.loc[a, b]:.2f}\n{star}"

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=labels, fmt="", cmap="coolwarm", vmin=-1, vmax=1,
            linewidths=0.5, square=True,
            annot_kws={"size": 7})
plt.title("Correlation Heatmap with Pearson p-values\n"
          "(*** p<0.001, ** p<0.01, * p<0.05, ns = not significant)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q06_annotated_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# COMPREHENSIVE REPORT
# ---------------------------------------------------------------------------
print("\nCOMPREHENSIVE CORRELATION REPORT")
print(f"{'Pair':56s} {'r':>7} {'p':>9} {'strength':>9}")
pairs_done = set()
for a in num_cols:
    for b in num_cols:
        if a >= b or (a, b) in pairs_done:
            continue
        pairs_done.add((a, b))
        r, p = corr.loc[a, b], pvals.loc[a, b]
        if abs(r) > 0.3:
            sig = "<0.001" if p < 0.001 else f"{p:.4f}"
            print(f"{a + ' <-> ' + b:56s} {r:+7.3f} {sig:>9} {strength(r):>9}")

print("\nSIGN LEGEND: S = strong (|r|>0.6), M = moderate (0.3-0.6), W = weak (<0.3)")
print("\n*** CORRELATION IS NOT CAUSATION ***")
print("- High study hours and high marks move together, but this data is")
print("  observational: a hidden factor (student ability) could drive both.")
print("- To claim causation you would need a controlled experiment or a")
print("  carefully justified causal model, not just a large r value.")
print("- Also beware: support sessions correlate NEGATIVELY with marks, yet")
print("  sessions HELP - struggling students are the ones attending them.")
