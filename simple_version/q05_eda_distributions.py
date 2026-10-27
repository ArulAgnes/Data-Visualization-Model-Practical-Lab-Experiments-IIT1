"""
Question 5: Exploratory Data Analysis (EDA)
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program explores the shape of 5 key numerical variables using
histograms, box plots, density plots and violin plots, plus statistics
(mean, median, std, skewness and IQR outlier counts).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- paths are anchored to THIS file so the program runs from any folder ---
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "..", "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(BASE, "..", "outputs", "simple")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")

# the 5 numerical variables we explore
cols = ["attendance_percentage", "study_hours_per_week", "lab_marks",
        "internal_marks", "final_exam_marks"]
short = ["Attendance", "Study Hours", "Lab Marks", "Internal", "Final Marks"]

# ---------------- CHART 1: HISTOGRAMS (2x3 grid) ----------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()                     # flatten 2x3 array into a 6-item list
for i, col in enumerate(cols):
    axes[i].hist(df[col], bins=20, color="#42A5F5", edgecolor="black", alpha=0.85)
    axes[i].set_title(short[i])
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")
axes[5].set_visible(False)              # 6th panel is empty
fig.suptitle("Histograms of Key Variables", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q05_histogram.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------- CHART 2: BOX PLOTS (side by side) -------------------------
plt.figure(figsize=(12, 5))
df[cols].boxplot(rot=15)
plt.title("Box Plots of Key Variables", fontsize=14)
plt.ylabel("Value")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q05_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------- CHART 3: DENSITY (KDE) PLOTS ------------------------------
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for i, col in enumerate(cols):
    sns.kdeplot(df[col], ax=axes[i], fill=True, color="#2196F3")
    axes[i].set_title(short[i])
    axes[i].set_xlabel(col)
fig.suptitle("Density Plots of Key Variables", fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q05_density.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------- CHART 4: VIOLIN PLOTS -------------------------------------
plt.figure(figsize=(12, 5))
df_long = df[cols].melt(var_name="variable", value_name="value")  # wide -> long
sns.violinplot(x="variable", y="value", data=df_long, palette="Set2", inner="quartile")
plt.title("Violin Plots of Key Variables", fontsize=14)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q05_violin.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------- STATISTICS -------------------------------------------------
print("STATISTICS FOR EACH VARIABLE")
print(f"{'Variable':24s} {'Mean':>7} {'Median':>7} {'Std':>7} {'Skew':>7} {'IQR outliers':>13}")
for col in cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1                                  # inter-quartile range
    low = q1 - 1.5 * iqr                           # classic outlier fence
    high = q3 + 1.5 * iqr
    n_out = int(((df[col] < low) | (df[col] > high)).sum())
    print(f"{col:24s} {df[col].mean():7.2f} {df[col].median():7.2f} "
          f"{df[col].std():7.2f} {df[col].skew():7.2f} {n_out:13d}")

print("\nINTERPRETATION")
print(f"- final_exam_marks: mean={df['final_exam_marks'].mean():.1f}, "
      f"median={df['final_exam_marks'].median():.1f}, "
      f"skewness={df['final_exam_marks'].skew():.2f} (close to symmetric)")
print(f"- study_hours_per_week: skewness={df['study_hours_per_week'].skew():.2f} "
      "-> strong RIGHT skew (a few students study a lot)")
print(f"- lab_marks: skewness={df['lab_marks'].skew():.2f} -> LEFT skew "
      "(most students score well)")
print("- Violin plots show the full shape; box plots only show quartiles.")
