"""
Question 3: Misleading Chart and Its Fix
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program deliberately draws a BAD (misleading) chart of department
averages, then draws an honest FIXED version, and explains the 4 issues.
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

# average marks of each department - the SAME data is used for both charts
dept_mean = df.groupby("department")["final_exam_marks"].mean()

# ===========================================================================
# MISLEADING CHART - deliberately broken on purpose (do NOT copy this style!)
# ===========================================================================
plt.figure(figsize=(8, 5))
# Issue 1: axis starts at 50 -> tiny differences look huge
# Issue 2: red/green colours (unreadable for colour-blind viewers)
# Issue 4: fake 3D effect = a dark copy of every bar drawn behind it
for dept, value in dept_mean.items():
    plt.bar(dept, value, color="#00C853", edgecolor="black", zorder=3)
    plt.bar(dept, value - 4, color="#FF1744", edgecolor="black",
            width=0.35, zorder=2)                       # dark "3D shadow" layer
plt.ylim(50, 60)                                        # Issue 1: truncated axis
plt.xticks(rotation=15)                                 # crowded labels
plt.yticks([])                                          # Issue 3: no axis scale
# Issue 3: no title, no x-label, no y-label
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q03_misleading.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ===========================================================================
# FIXED CHART - honest, readable, accessible
# ===========================================================================
plt.figure(figsize=(8, 5))
bars = plt.bar(dept_mean.index, dept_mean.values,
               color="#2196F3", edgecolor="black")      # Fix 2: colourblind-safe blue
for bar, value in zip(bars, dept_mean.values):          # Fix 4: clear value labels
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f"{value:.1f}", ha="center", fontweight="bold")
plt.ylim(0, 100)                                        # Fix 1: axis starts at 0
plt.title("Average Final Marks by Department", fontsize=14)   # Fix 3: title
plt.xlabel("Department")                                # Fix 3: x-label
plt.ylabel("Average Marks")                             # Fix 3: y-label
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q03_fixed.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ===========================================================================
# REPORT CARD - the 4 issues and their fixes
# ===========================================================================
print("ISSUES FOUND IN THE MISLEADING CHART")
print("Issue 1: Truncated y-axis starting at 50 exaggerated small differences")
print("Fix 1:   Y-axis now starts at 0 for honest comparison")
print("Issue 2: Red-green colours are not accessible for colour-blind viewers")
print("Fix 2:   Used colourblind-friendly blue palette")
print("Issue 3: Missing title and labels made the chart uninterpretable")
print("Fix 3:   Added descriptive title and axis labels")
print("Issue 4: Fake 3D shadow effects distorted data perception")
print("Fix 4:   Used a clean 2D bar chart with plain value labels")
print("\nINTERPRETATION")
print(f"- The real department averages are very close: "
      f"{dept_mean.min():.1f} to {dept_mean.max():.1f} marks")
print("- The truncated axis made a ~2 mark difference look like a 10x difference.")
print("- Always start bar-chart axes at 0 and label everything clearly.")
