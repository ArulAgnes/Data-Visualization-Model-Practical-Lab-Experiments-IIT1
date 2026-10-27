"""
Question 8: Rolling Average and Uncertainty Band
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program smooths the weekly assessment averages with a 3-point rolling
mean and draws a +/- 1 standard-deviation band to show how variable the
performance is around the trend.
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
df["assessment_date"] = pd.to_datetime(df["assessment_date"])

# weekly mean and standard deviation of the marks
weekly = (df.groupby("assessment_date")["final_exam_marks"]
            .agg(mean="mean", std="std", count="count")
            .reset_index()
            .sort_values("assessment_date"))

# 3-point rolling average: each point = average of itself + previous 2 dates
weekly["rolling_mean"] = weekly["mean"].rolling(3).mean()
weekly["rolling_std"] = weekly["std"].rolling(3).std()

# uncertainty band = rolling mean +/- 1 rolling standard deviation
upper = weekly["rolling_mean"] + weekly["rolling_std"]
lower = weekly["rolling_mean"] - weekly["rolling_std"]

sns.set_style("whitegrid")
plt.figure(figsize=(12, 5))

# thin line = raw weekly mean, thick line = smoothed trend
plt.plot(weekly["assessment_date"], weekly["mean"],
         marker="o", linewidth=1.2, alpha=0.6, label="Weekly Mean")
plt.plot(weekly["assessment_date"], weekly["rolling_mean"],
         linewidth=2.5, color="#FF5722", label="3-Week Rolling Average")
plt.fill_between(weekly["assessment_date"], lower, upper,
                 alpha=0.3, color="#FF5722", label="+/-1 Std Dev Band")

plt.title("Performance Trend with Rolling Average and Uncertainty Band", fontsize=14)
plt.xlabel("Assessment Date")
plt.ylabel("Average Final Exam Marks")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q08_rolling_avg.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- INTERPRETATION --------------------------------------------
print(weekly[["assessment_date", "mean", "rolling_mean", "rolling_std"]].round(2).to_string(index=False))
print("\nINTERPRETATION")
print("- The rolling average smooths out weekly fluctuations to reveal the trend.")
print("- Wider bands mean the students assessed on those dates were more mixed.")
widest = weekly.loc[weekly["rolling_std"].idxmax()]
print(f"- Widest variability around {widest['assessment_date'].date()} "
      f"(std = {widest['rolling_std']:.1f}) - worth investigating.")
print("- Note: this band is +/-1 standard deviation (data spread), "
      "NOT a confidence interval of the mean.")
