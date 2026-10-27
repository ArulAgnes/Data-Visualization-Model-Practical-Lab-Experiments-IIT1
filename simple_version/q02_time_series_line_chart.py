"""
Question 2: Time-Series Line Chart
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program plots the average student performance on each assessment date
across the semester to see whether performance improves, drops or stays stable.
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

# Step 1: convert the date text column into real datetime objects
df["assessment_date"] = pd.to_datetime(df["assessment_date"])

# Step 2: average the marks of every student who was assessed on each date
by_date = (df.groupby("assessment_date")["final_exam_marks"]
             .mean()
             .reset_index()          # turn the grouped index back into a column
             .sort_values("assessment_date"))   # oldest date first

print("Number of assessment dates:", len(by_date))
print(by_date.head())

sns.set_style("whitegrid")

# ---------------- LINE CHART -------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(by_date["assessment_date"], by_date["final_exam_marks"],
         marker="o", color="#2196F3", linewidth=2, markersize=6)
plt.title("Average Student Performance Over Semester", fontsize=14)
plt.xlabel("Assessment Date")
plt.ylabel("Average Final Exam Marks")
plt.xticks(rotation=45)              # tilt date labels so they do not overlap
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q02_time_series.png"), dpi=150, bbox_inches="tight")
# plt.show()   # uncomment to open the window locally
plt.close()

# ---------------- INTERPRETATION --------------------------------------------
first_val = by_date["final_exam_marks"].iloc[0]
last_val = by_date["final_exam_marks"].iloc[-1]
peak = by_date.loc[by_date["final_exam_marks"].idxmax()]
valley = by_date.loc[by_date["final_exam_marks"].idxmin()]

if last_val > first_val + 1:
    trend = "INCREASING"
elif last_val < first_val - 1:
    trend = "DECREASING"
else:
    trend = "STABLE"

print("\nINTERPRETATION")
print(f"- Performance shows an {trend} trend "
      f"(first date {first_val:.1f} -> last date {last_val:.1f})")
print(f"- Highest average was on {peak['assessment_date'].date()} "
      f"with {peak['final_exam_marks']:.1f}")
print(f"- Lowest average was on {valley['assessment_date'].date()} "
      f"with {valley['final_exam_marks']:.1f}")
print("- Note: each date assesses a different random group of students, "
      "so small jumps may be sampling noise, not a real trend.")
