"""
Question 1: Bar Chart, Histogram, Box Plot
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program makes three basic charts: a bar chart to compare sections,
a histogram to show the shape of the marks distribution, and a box plot
to compare the spread of marks between sections.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- paths are anchored to THIS file so the program runs from any folder ---
BASE = os.path.dirname(os.path.abspath(__file__))          # simple_version/
CSV = os.path.join(BASE, "..", "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(BASE, "..", "outputs", "simple")
os.makedirs(OUT, exist_ok=True)                             # create folder if missing

df = pd.read_csv(CSV)
print("Shape (rows, columns):", df.shape)
print(df.head())

sns.set_style("whitegrid")           # light background with grid lines

# ---------------- CHART 1: BAR CHART (average marks per section) ------------
sec_mean = df.groupby("section")["final_exam_marks"].mean()   # group then average
plt.figure(figsize=(8, 5))
bars = plt.bar(sec_mean.index, sec_mean.values, color="#2196F3", edgecolor="black")
for bar, value in zip(bars, sec_mean.values):                 # label on top of bars
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{value:.1f}", ha="center", fontweight="bold")
plt.title("Average Final Exam Marks by Section", fontsize=14)
plt.xlabel("Section")
plt.ylabel("Average Marks")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q01_bar_chart.png"), dpi=150, bbox_inches="tight")
# plt.show()   # uncomment to open the window on your own laptop
plt.close()

# ---------------- CHART 2: HISTOGRAM (distribution of all marks) ------------
plt.figure(figsize=(8, 5))
plt.hist(df["final_exam_marks"], bins=20, color="#4CAF50", edgecolor="black", alpha=0.85)
mean_marks = df["final_exam_marks"].mean()
plt.axvline(mean_marks, color="red", linestyle="--", linewidth=2,
            label=f"Mean = {mean_marks:.1f}")                 # vertical mean line
plt.title("Distribution of Final Exam Marks", fontsize=14)
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q01_histogram.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------- CHART 3: BOX PLOT (marks spread inside each section) -------
plt.figure(figsize=(8, 5))
sns.boxplot(x="section", y="final_exam_marks", data=df, order=["A", "B", "C", "D"],
            hue="section", palette="Set2", legend=False)
plt.title("Final Exam Marks Distribution by Section", fontsize=14)
plt.xlabel("Section")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q01_boxplot.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------- INTERPRETATION --------------------------------------------
spread = df.groupby("section")["final_exam_marks"].std()       # variation per section
print("\nINTERPRETATION")
print(f"- Section {sec_mean.idxmax()} has the highest average marks ({sec_mean.max():.1f})")
print(f"- Marks range from {df['final_exam_marks'].min():.1f} to "
      f"{df['final_exam_marks'].max():.1f}")
print(f"- Section {spread.idxmax()} shows the most variation in marks "
      f"(std = {spread.max():.1f})")
print("- Bar charts compare categories, histograms show distribution shape, "
      "box plots show spread and outliers.")
