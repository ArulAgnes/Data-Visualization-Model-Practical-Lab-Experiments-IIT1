"""
Question 5: EDA (Comprehensive Statistical)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Full exploratory data analysis: histograms with KDE, box plots with mean
markers, violin plots with embedded boxes, QQ plots for normality, a
formatted statistics table (mean/median/mode/std/skew/kurtosis) and
outlier detection by BOTH the IQR and z-score methods.
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

cols = ["attendance_percentage", "study_hours_per_week", "lab_marks",
        "internal_marks", "final_exam_marks"]
short = ["Attendance", "Study Hours", "Lab Marks", "Internal", "Final Marks"]

# ---------------------------------------------------------------------------
# FIGURE 1 - COMBINED EDA (2x2: hist+KDE, box with means, violin, QQ)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) histogram + KDE overlay for every variable (melted frame)
long = df[cols].melt(var_name="variable", value_name="value")
sns.histplot(data=long, x="value", hue="variable", element="step",
             stat="density", common_norm=False, kde=True, ax=axes[0, 0],
             palette="Set2")
axes[0, 0].set_title("(a) Histograms with KDE overlay", fontsize=12)
axes[0, 0].set_xlabel("Value")
axes[0, 0].legend(fontsize=7, title="")

# (b) box plots with explicit mean markers
sns.boxplot(data=df[cols], orient="h", palette="Set2", ax=axes[0, 1])
means = df[cols].mean()
for i, m in enumerate(means):
    axes[0, 1].plot(m, i, marker="D", color="red", markersize=7, zorder=5)
axes[0, 1].set_title("(b) Box plots (red diamond = mean)", fontsize=12)
axes[0, 1].set_xlabel("Value")

# (c) violin plots with embedded mini box plots
sns.violinplot(data=long, x="variable", y="value", inner="box",
               palette="Set2", ax=axes[1, 0], cut=0)
axes[1, 0].set_title("(c) Violin plots with inner box", fontsize=12)
axes[1, 0].set_xlabel("")
axes[1, 0].tick_params(axis="x", rotation=15)

# (d) QQ plots: points on the line = normally distributed
for col, color in zip(cols, sns.color_palette("Set2")):
    stats.probplot(df[col].dropna(), dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("(d) Normal QQ plots (all variables)", fontsize=12)
axes[1, 1].get_lines()[0].set(marker="o", markersize=2.5, color="#2196F3")
for line in axes[1, 1].get_lines()[1:]:
    line.set_color("#FF5722")

fig.suptitle("Comprehensive EDA of Key Variables", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q05_combined_eda.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# STATISTICS TABLE
# ---------------------------------------------------------------------------
print("=" * 96)
print(f"{'Variable':24s} {'Mean':>7} {'Median':>7} {'Mode':>7} {'Std':>7} "
      f"{'Skew':>7} {'Kurt':>7} {'Normal?':>9}")
print("-" * 96)
stats_rows = []
for col in cols:
    s = df[col]
    # Shapiro test on a sample (test breaks on n>5000; 1200 is fine)
    _, p = stats.shapiro(s.sample(min(len(s), 500), random_state=42))
    normal = "yes" if p > 0.05 else "no"
    row = dict(var=col, mean=s.mean(), median=s.median(),
               mode=s.mode().iloc[0], std=s.std(),
               skew=s.skew(), kurt=s.kurtosis(), p=p, normal=normal)
    stats_rows.append(row)
    print(f"{col:24s} {row['mean']:7.2f} {row['median']:7.2f} {row['mode']:7.2f} "
          f"{row['std']:7.2f} {row['skew']:7.2f} {row['kurt']:7.2f} {normal:>9}")
print("-" * 96)
print("(Shapiro-Wilk normality test on 500-row sample; p>0.05 => compatible with normal)")

# ---------------------------------------------------------------------------
# OUTLIER DETECTION: IQR rule vs z-score rule
# ---------------------------------------------------------------------------
iqr_counts, z_counts = {}, {}
for col in cols:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    iqr_counts[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    z = np.abs((df[col] - df[col].mean()) / df[col].std())
    z_counts[col] = int((z > 3).sum())

out_tbl = pd.DataFrame({"IQR_method": iqr_counts, "zscore_method": z_counts})
out_tbl["difference"] = out_tbl["IQR_method"] - out_tbl["zscore_method"]
print("\nOUTLIER DETECTION COMPARISON")
print(out_tbl.to_string())

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(cols))
w = 0.36
ax.bar(x - w / 2, out_tbl["IQR_method"], w, label="IQR (Q1-1.5*IQR fences)",
       color="#2196F3", edgecolor="black")
ax.bar(x + w / 2, out_tbl["zscore_method"], w, label="z-score (|z| > 3)",
       color="#FF5722", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(short)
ax.set_ylabel("Number of Outliers")
ax.set_title("Outlier Counts: IQR vs z-score", fontsize=13)
ax.legend()
for i, (a, b) in enumerate(zip(out_tbl["IQR_method"], out_tbl["zscore_method"])):
    ax.text(i - w / 2, a + 0.4, str(a), ha="center", fontsize=9)
    ax.text(i + w / 2, b + 0.4, str(b), ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q05_outlier_detection.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
print("\nINTERPRETATION")
for row in stats_rows:
    shape = "right-skewed" if row["skew"] > 0.5 else (
        "left-skewed" if row["skew"] < -0.5 else "roughly symmetric")
    print(f"- {row['var']}: {shape} (skew={row['skew']:.2f}), "
          f"normal? {row['normal']}")
print("- The IQR rule always flags MORE outliers than |z|>3 because it uses")
print("  quartiles instead of mean/std and reacts to skew.")
print("- QQ points hugging the red line => that variable is close to normal;")
print("  systematic S-curves or fanning => non-normal.")
