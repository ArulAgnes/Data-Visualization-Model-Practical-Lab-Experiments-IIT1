"""
Question 3: Misleading Visualization Redesign (Detailed)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Builds a deliberately WORSE chart with 6 distinct misleading issues,
puts it side-by-side with an honest redesign, demonstrates two fixes as
mini-visualizations, and prints a report card citing perception theory.
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

dept_mean = df.groupby("department")["final_exam_marks"].mean()
depts = list(dept_mean.index)
values = list(dept_mean.values)
print("HONEST VALUES:", {d: round(v, 2) for d, v in zip(depts, values)})

# rainbow colours = pretty but meaningless and inaccessible
RAINBOW = ["#ff0000", "#ff8800", "#ffee00", "#00cc00", "#0066ff", "#8800cc"]

# ===========================================================================
# THE 6 MISLEADING ISSUES
#   1. Truncated y-axis (starts at 55) exaggerates differences
#   2. 3D-style pie chart for data that suits bars
#   3. Rainbow colour scheme (colour-blind hostile, implies fake order)
#   4. Missing title, axis labels and legend
#   5. Overcrowded overlapping labels
#   6. Dual y-axes scaled to force a fake "trend"
# ===========================================================================

# --- mini chart of issues 2,3,4,5: a broken 3D-ish pie ---
fig = plt.figure(figsize=(13, 9))
ax_pie = fig.add_subplot(2, 2, 1)
wedges, _, autotexts = ax_pie.pie(
    values, labels=None, colors=RAINBOW[:3], autopct="%1.1f%%",
    startangle=90, explode=(0.12, 0.12, 0.12), shadow=True,  # issue 2: decoration
    textprops=dict(fontsize=8))
for i, t in enumerate(autotexts):                 # issue 5: crowded labels
    t.set_position((t.get_position()[0] * 0.55, t.get_position()[1] * 0.55))
    t.set_fontsize(6)
# issue 4: NO title, NO labels, NO legend

# --- issue 1 + 6: truncated-axis bars with a fake dual axis ---
ax_bad = fig.add_subplot(2, 2, 2)
x = np.arange(len(depts))
ax_bad.bar(x, values, color=RAINBOW[:3], edgecolor="black", width=0.6)  # issue 3
ax_bad.set_ylim(55, 60)                           # issue 1: truncated axis
ax_bad.set_yticks([])                             # hides the truncation
ax_bad.set_xticks(x)
ax_bad.set_xticklabels([])                        # issue 4: no labels
ax2 = ax_bad.twinx()                              # issue 6: needless 2nd axis
noise = np.linspace(0.1, 0.9, len(depts))
ax2.plot(x, noise, color="red", linewidth=3)      # fake "trend" on dummy scale
ax2.set_ylim(0, 1)
ax2.set_yticks([])
for i, (xi, yi) in enumerate(zip(x, noise)):      # issue 5: overlapping text
    ax_bad.text(xi, 55.1, f"{values[i]:.2f} marks dept {depts[i]} avg score",
                rotation=90, fontsize=7, ha="center")

fig.suptitle("", fontsize=1)   # issue 4: intentionally no overall title
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q03_misleading.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# BEFORE / AFTER COMPARISON
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# BEFORE (re-drawn compactly with issues 1,3,4)
axes[0, 0].bar(x, values, color=RAINBOW[:3], edgecolor="black", width=0.6)
axes[0, 0].set_ylim(55, 60)
axes[0, 0].set_yticks([])
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels([])
axes[0, 0].set_title("BEFORE: misleading", fontsize=12)

# AFTER - honest bar chart
bars = axes[0, 1].bar(depts, values, color="#2196F3", edgecolor="black")
for b, v in zip(bars, values):
    axes[0, 1].text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6,
                    f"{v:.1f}", ha="center", fontweight="bold")
axes[0, 1].set_ylim(0, 100)
axes[0, 1].set_title("AFTER: title + labels + zero-based axis", fontsize=12)
axes[0, 1].set_xlabel("Department")
axes[0, 1].set_ylabel("Average Final Marks")

# MINI-FIX 1: same bars, two axis scales side by side
axes[1, 0].bar(depts, values, color="#4CAF50", edgecolor="black")
axes[1, 0].set_ylim(55, 60)
axes[1, 0].set_title("Fix demo: axis from 55 (dishonest)", fontsize=11)
mini = axes[1, 0].inset_axes([0.55, 0.08, 0.42, 0.35])   # small inset with axis at 0
mini.bar(depts, values, color="#4CAF50", edgecolor="black")
mini.set_ylim(0, 100)
mini.set_title("axis from 0 (honest)", fontsize=8)
mini.tick_params(labelsize=7)

# MINI-FIX 2: rainbow vs single accessible colour
axes[1, 1].bar(depts, values, color=RAINBOW[:3], edgecolor="black")
axes[1, 1].set_ylim(0, 100)
axes[1, 1].set_title("Fix demo: rainbow (bad) vs single colour (good)", fontsize=11)
for i, v in enumerate(values):
    axes[1, 1].bar(i + 0.02, v, width=0.28, color="#2196F3", edgecolor="black")

fig.suptitle("Misleading Chart vs Honest Redesign", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q03_before_after.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===========================================================================
# REPORT CARD
# ===========================================================================
spread_real = max(values) - min(values)
print("\nREPORT CARD")
print(f"{'#':>2} {'ISSUE':52s} {'FIX'}")
rows = [
    (1, "Truncated y-axis (55-60) makes {0:.1f} mark gaps look huge".format(spread_real),
     "Axis starts at 0; differences shown honestly"),
    (2, "3D/pie chart: angle+area are hard to compare, adds clutter",
     "Plain 2D bars: position on a common scale"),
    (3, "Rainbow colours: inaccessible & imply meaningless order",
     "Single colourblind-safe blue"),
    (4, "No title/labels/legend: chart is uninterpretable",
     "Descriptive title and axis labels added"),
    (5, "Overcrowded overlapping text hides the message",
     "Few, bold value labels only"),
    (6, "Dual y-axes rescaled to invent a fake trend",
     "One axis per unit; second axis only when units differ"),
]
for num, issue, fix in rows:
    print(f"{num:>2} {issue:52s} {fix}")

print("\nINTERPRETATION (perception theory)")
print("- Stevens' power law: perceived quantity grows as a POWER of intensity -")
print("  bar length (exponent ~1) is far more accurate than area (~0.7) or angle.")
print("- Pre-attentive features (colour, position) are processed instantly -")
print("  use them for the ONE message you want the reader to see first.")
print(f"- The true department spread is only {spread_real:.1f} marks; the misleading")
print("  chart made it look like a factor-of-10 difference.")
