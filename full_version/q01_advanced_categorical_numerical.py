"""
Question 1: Categorical and Numerical Visualization (Comprehensive)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Comprehensive categorical analysis: grouped bars (section x department),
overlaid histograms, box plots coloured by gender, a count plot, plus a
full statistical summary table per group.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CSV = os.path.join(ROOT, "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(ROOT, "outputs", "advanced")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")
PALETTE = ["#2196F3", "#4CAF50", "#FF9800"]


def group_summary(data, by):
    """Return counts, mean, median and std of final marks for every group."""
    out = data.groupby(by)["final_exam_marks"].agg(
        count="count", mean="mean", median="median", std="std")
    return out.round(2)


# ---------------------------------------------------------------------------
# FIGURE 1 - GROUPED BAR: average marks by section AND department
# ---------------------------------------------------------------------------
sec_dept = df.pivot_table(index="section", columns="department",
                          values="final_exam_marks", aggfunc=np.mean)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sec_dept.plot.bar(ax=axes[0], color=PALETTE, edgecolor="black", width=0.75)
axes[0].set_title("Average Final Marks: Section x Department", fontsize=13)
axes[0].set_xlabel("Section")
axes[0].set_ylabel("Average Marks")
axes[0].set_ylim(0, 100)
axes[0].legend(title="Department")
axes[0].tick_params(axis="x", rotation=0)
for container in axes[0].containers:                 # value labels on every bar
    axes[0].bar_label(container, fmt="%.1f", fontsize=8, padding=2)

# count plot: how many students per department (context for the averages)
dept_count = df["department"].value_counts().reindex(["CSE", "CSE-AI", "CSE-DS"])
bars = axes[1].bar(dept_count.index, dept_count.values,
                   color=PALETTE, edgecolor="black")
for bar, v in zip(bars, dept_count.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                 str(v), ha="center", fontweight="bold")
axes[1].set_title("Number of Students per Department", fontsize=13)
axes[1].set_xlabel("Department")
axes[1].set_ylabel("Student Count")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q01_grouped_bar.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 2 - STACKED/OVERLAID HISTOGRAM: marks distribution by department
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
bins = np.arange(0, 105, 5)
for dept, color in zip(["CSE", "CSE-AI", "CSE-DS"], PALETTE):
    subset = df.loc[df["department"] == dept, "final_exam_marks"]
    ax.hist(subset, bins=bins, density=True, alpha=0.55, edgecolor="black",
            label=f"{dept} (n={len(subset)})", color=color)
    ax.axvline(subset.mean(), color=color, linestyle="--", linewidth=2)
ax.set_title("Distribution of Final Marks by Department (density)", fontsize=13)
ax.set_xlabel("Final Exam Marks")
ax.set_ylabel("Density")
ax.legend(title="Department")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q01_stacked_histogram.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 3 - GROUPED BOX PLOT: marks by section, coloured by gender
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
sns.boxplot(x="section", y="final_exam_marks", hue="gender", data=df,
            order=["A", "B", "C", "D"], palette="Set2", ax=ax)
# overlay group means as diamonds
means = df.groupby(["section", "gender"])["final_exam_marks"].mean().reset_index()
sns.stripplot(x="section", y="final_exam_marks", hue="gender", data=means,
              order=["A", "B", "C", "D"], palette="Set2", marker="D",
              size=9, ax=ax, dodge=True, legend=False, color="black")
ax.set_title("Final Marks by Section, Coloured by Gender (diamond = group mean)",
             fontsize=13)
ax.set_xlabel("Section")
ax.set_ylabel("Final Exam Marks")
ax.legend(title="Gender", bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q01_grouped_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# DETAILED STATISTICAL SUMMARY
# ---------------------------------------------------------------------------
print("=" * 72)
print("SUMMARY BY SECTION")
print(group_summary(df, "section").to_string())
print("\nSUMMARY BY DEPARTMENT")
print(group_summary(df, "department").to_string())
print("\nSUMMARY BY GENDER")
print(group_summary(df, "gender").to_string())
print("\nCROSS-TAB: section x department")
print(pd.crosstab(df["section"], df["department"]).to_string())

print("\nINTERPRETATION")
sec_means = df.groupby("section")["final_exam_marks"].mean()
print(f"- Section means differ by only {sec_means.max() - sec_means.min():.1f} "
      "marks - sections are quite similar")
print(f"- CSE has {len(df[df.department == 'CSE'])} students, so its averages "
      "are the most reliable; CSE-DS (n=100) is noisiest")
print("- Grouped bars compare two categories at once; overlaid histograms show "
      "shape; box plots expose outliers inside each group")
