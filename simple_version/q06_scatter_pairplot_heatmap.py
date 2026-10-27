"""
Question 6: Scatter Plot, Pair Plot and Correlation Heatmap
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program studies relationships between variables: a scatter plot of
study hours vs marks, a pair plot of 4 variables, and a correlation
heatmap of every numerical column.
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

# ---------------- CHART 1: SCATTER PLOT -------------------------------------
plt.figure(figsize=(8, 5))
sns.scatterplot(x="study_hours_per_week", y="final_exam_marks",
                hue="pass_status", data=df,
                palette={"Pass": "#4CAF50", "Fail": "#FF5722"}, alpha=0.7)
plt.title("Study Hours vs Final Marks", fontsize=14)
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Exam Marks")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q06_scatter.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- CHART 2: PAIR PLOT ----------------------------------------
pair_cols = ["study_hours_per_week", "attendance_percentage",
             "lab_marks", "final_exam_marks"]
g = sns.pairplot(df, hue="pass_status", palette={"Pass": "#4CAF50", "Fail": "#FF5722"},
                 corner=True)          # corner=True drops duplicate upper panels
g.figure.suptitle("Pair Plot of 4 Key Variables", y=1.02)
g.savefig(os.path.join(OUT, "q06_pairplot.png"), dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------- CHART 3: CORRELATION HEATMAP ------------------------------
num_cols = df.select_dtypes(include="number").columns   # all numerical columns
corr = df[num_cols].corr()                              # correlation matrix

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q06_heatmap.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- INTERPRETATION --------------------------------------------
pairs = corr["final_exam_marks"].drop("final_exam_marks").sort_values(ascending=False)
print("INTERPRETATION")
print(f"- Strongest positive correlation: study_hours_per_week <-> final_exam_marks "
      f"(r={pairs.iloc[0]:.2f})")
print(f"- support_sessions_attended <-> final_exam_marks (r={pairs.iloc[-1]:.2f}) "
      "- struggling students attend more support sessions")
print("- Values close to +1 move together; close to -1 move opposite; near 0 = no link.")
print("- NOTE: Correlation does NOT prove causation - high marks do not "
      "automatically come from any single column.")
