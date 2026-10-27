"""
Question 2: Time Series (Multi-Series)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Plots several measures over the semester on dual axes, fits a linear
trend line to performance, annotates peak/valley weeks, and checks
whether the pattern is a real trend or small-sample noise.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CSV = os.path.join(ROOT, "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(ROOT, "outputs", "advanced")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
df["assessment_date"] = pd.to_datetime(df["assessment_date"])
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Weekly aggregation (smoother than raw dates)
# ---------------------------------------------------------------------------
df["week"] = df["assessment_date"].dt.isocalendar().week.astype(int)
weekly = df.groupby("week").agg(
    date=("assessment_date", "min"),
    marks=("final_exam_marks", "mean"),
    attendance=("attendance_percentage", "mean"),
    study=("study_hours_per_week", "mean"),
    n=("final_exam_marks", "count"),
).reset_index()
print("WEEKLY AGGREGATES")
print(weekly.round(2).to_string(index=False))

# linear trend on marks: y = slope * time + intercept
time_num = np.arange(len(weekly))            # week index 0,1,2,...
slope, intercept = np.polyfit(time_num, weekly["marks"], 1)
trend = slope * time_num + intercept
trend_dir = "UPWARD" if slope > 0 else "DOWNWARD"

peak = weekly.loc[weekly["marks"].idxmax()]
valley = weekly.loc[weekly["marks"].idxmin()]

# ---------------------------------------------------------------------------
# FIGURE 1 - MULTI-SERIES with dual y-axes
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(weekly["date"], weekly["marks"], marker="o", color="#2196F3",
         linewidth=2, label="Avg Final Marks (left)")
ax1.plot(weekly["date"], weekly["attendance"], marker="s", color="#4CAF50",
         linewidth=1.8, linestyle="--", label="Avg Attendance % (left)")
ax1.plot(weekly["date"], trend, color="#2196F3", linestyle=":", linewidth=2.5,
         label=f"Linear trend ({slope:+.2f} marks/week)")
ax1.set_ylabel("Marks / Attendance (%)")
ax1.set_ylim(0, 100)
ax1.set_xlabel("Week of Semester")

# right axis for study hours (different unit -> second axis is justified here)
ax2 = ax1.twinx()
ax2.plot(weekly["date"], weekly["study"], marker="^", color="#FF5722",
         linewidth=1.8, label="Avg Study Hours (right)")
ax2.set_ylabel("Study Hours per Week")
ax2.set_ylim(0, 30)

# highlight start / mid / end of semester
ax1.axvspan(weekly["date"].iloc[0], weekly["date"].iloc[3],
            alpha=0.08, color="green", label="Start of semester")
ax1.axvspan(weekly["date"].iloc[len(weekly) // 2 - 1],
            weekly["date"].iloc[len(weekly) // 2 + 1],
            alpha=0.08, color="orange", label="Mid semester")
ax1.axvspan(weekly["date"].iloc[-4], weekly["date"].iloc[-1],
            alpha=0.08, color="red", label="End of semester")

# annotate peak and valley
ax1.annotate(f"Peak {peak['marks']:.1f}", xy=(peak["date"], peak["marks"]),
             xytext=(10, 15), textcoords="offset points",
             arrowprops=dict(arrowstyle="->"), fontsize=10, fontweight="bold")
ax1.annotate(f"Valley {valley['marks']:.1f}", xy=(valley["date"], valley["marks"]),
             xytext=(10, -25), textcoords="offset points",
             arrowprops=dict(arrowstyle="->"), fontsize=10, fontweight="bold")

# proper date formatting
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
ax1.set_title("Semester Performance: Marks, Attendance, Study Hours and Trend",
              fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q02_multi_series.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 2 - SEASONAL view: marks grouped by month + week box plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
df["month"] = df["assessment_date"].dt.strftime("%Y-%m")
sns.boxplot(x="month", y="final_exam_marks", data=df, palette="Blues", ax=axes[0])
axes[0].set_title("Marks by Month (seasonal view)", fontsize=13)
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Final Exam Marks")

# week-to-week spread shows how noisy each assessment was
sns.boxplot(x="week", y="final_exam_marks", data=df, palette="viridis", ax=axes[1])
axes[1].set_title("Marks by Week (week-to-week spread)", fontsize=13)
axes[1].set_xlabel("Week number")
axes[1].set_ylabel("")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q02_seasonal.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------------------------
month_means = df.groupby("month")["final_exam_marks"].mean()
print("\nINTERPRETATION")
print(f"- Trend: {trend_dir} at {slope:+.3f} marks per week "
      f"(total change over semester ~ {slope * len(weekly):+.1f} marks)")
print(f"- Peak week {int(peak['week'])} ({peak['date'].date()}) = {peak['marks']:.1f}; "
      f"valley week {int(valley['week'])} ({valley['date'].date()}) = {valley['marks']:.1f}")
print("- Monthly means:", ", ".join(f"{m}={v:.1f}" for m, v in month_means.items()))
print(f"- Sample sizes per week range {weekly['n'].min()}-{weekly['n'].max()} students; "
      "weeks with small n fluctuate more - check before calling a jump a 'trend'.")
print("- A genuine trend should persist across many weeks AND have decent sample "
      "sizes; a single wild week is usually noise.")
