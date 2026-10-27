"""
Question 8: Time Series Uncertainty (Detailed)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Weekly means with standard errors and 95% confidence intervals, a
4-week rolling average with BOTH a +/-1 std band and a 95% CI band,
error bars, and a comparison of 2/4/6-week smoothing windows.
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
# Weekly aggregation: mean, std, count, standard error, 95% CI
# ---------------------------------------------------------------------------
wk = (df.groupby("assessment_date")["final_exam_marks"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values("assessment_date"))
wk["se"] = wk["std"] / np.sqrt(wk["n"])                 # standard error of mean
wk["ci_lo"] = wk["mean"] - 1.96 * wk["se"]              # 95% confidence interval
wk["ci_hi"] = wk["mean"] + 1.96 * wk["se"]
wk["sd_lo"] = wk["mean"] - wk["std"]                    # +/-1 std dev band
wk["sd_hi"] = wk["mean"] + wk["std"]
wk["roll4"] = wk["mean"].rolling(4).mean()              # 4-week rolling mean

print("WEEKLY STATISTICS")
print(wk[["assessment_date", "mean", "std", "n", "se", "ci_lo", "ci_hi"]]
      .round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# FIGURE 1 - UNCERTAINTY BANDS (std band vs 95% CI band)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(wk["assessment_date"], wk["mean"], marker="o", linewidth=1.3,
        alpha=0.65, color="#2196F3", label="Weekly mean")
ax.plot(wk["assessment_date"], wk["roll4"], linewidth=2.6, color="#FF5722",
        label="4-week rolling average")
ax.fill_between(wk["assessment_date"], wk["sd_lo"], wk["sd_hi"],
                alpha=0.18, color="#4CAF50", label="+/-1 std dev (data spread)")
ax.fill_between(wk["assessment_date"], wk["ci_lo"], wk["ci_hi"],
                alpha=0.4, color="#FF5722", label="95% CI of the mean")
ax.set_title("Weekly Performance: Rolling Average, Std Band and 95% Confidence Band",
             fontsize=13)
ax.set_xlabel("Assessment Date")
ax.set_ylabel("Final Exam Marks")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q08_uncertainty_band.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 2 - ERROR BARS + WINDOW COMPARISON
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# (a) error bars showing the 95% CI at every point
axes[0].errorbar(wk["assessment_date"], wk["mean"], yerr=1.96 * wk["se"],
                 fmt="o", capsize=4, markersize=5, color="#2196F3",
                 ecolor="#FF5722", elinewidth=1.5,
                 label="mean +/- 1.96 SE")
axes[0].set_title("(a) Weekly means with 95% CI error bars", fontsize=12)
axes[0].set_xlabel("Assessment Date")
axes[0].set_ylabel("Mean Marks")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[0].legend(fontsize=9)

# (b) same series smoothed with 2-, 4- and 6-week windows
for window, color in [(2, "#9C27B0"), (4, "#FF5722"), (6, "#2196F3")]:
    axes[1].plot(wk["assessment_date"], wk["mean"].rolling(window).mean(),
                 linewidth=2, color=color, label=f"{window}-week rolling")
axes[1].plot(wk["assessment_date"], wk["mean"], linewidth=1, alpha=0.4,
             color="gray", label="raw weekly mean")
axes[1].set_title("(b) Rolling window comparison", fontsize=12)
axes[1].set_xlabel("Assessment Date")
axes[1].set_ylabel("Mean Marks")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
axes[1].legend(fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "q08_error_bars.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
widest = wk.loc[wk["std"].idxmax()]
narrowest = wk.loc[wk["se"].idxmax()]
print("\nINTERPRETATION")
print("- Std dev shows how SPREAD OUT the students were on that date.")
print("- The 95% CI shows how precisely the MEAN is estimated: "
      "mean +/- 1.96 * std/sqrt(n).")
print(f"- Widest spread: {widest['assessment_date'].date()} "
      f"(std = {widest['std']:.1f}, n = {int(widest['n'])})")
print(f"- Widest CI (least precise mean): {narrowest['assessment_date'].date()} "
      f"(+/- {1.96 * narrowest['se']:.1f} marks)")
print("- The std band is much wider than the CI band: individual students vary a")
print("  lot, but a mean of ~60 students is estimated quite precisely.")
print("- 2-week windows track noise; 6-week windows hide real turning points -")
print("  pick the window based on how smooth your question needs the answer to be.")
