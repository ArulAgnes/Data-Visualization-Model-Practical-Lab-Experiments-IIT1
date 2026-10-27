"""
Question 4: Encoding Comparison (Theory-Backed)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Compares three encodings of the same numbers - grouped bars (position +
length), bubbles (size + colour) and a heatmap (colour intensity) -
ranks them with the Cleveland & McGill hierarchy, then redesigns the
weakest encoding.
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

# the SAME numbers encoded three different ways
sec_mean = df.groupby("section")["final_exam_marks"].mean()
sections = list(sec_mean.index)
marks = list(sec_mean.values)

# ---------------------------------------------------------------------------
# ENCODINGS: (1) position+length  (2) area+colour  (3) colour intensity
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1 - grouped/simple bar: position on a common scale
axes[0].bar(sections, marks, color="#2196F3", edgecolor="black", width=0.6)
for i, v in enumerate(marks):
    axes[0].text(i, v + 0.8, f"{v:.1f}", ha="center", fontweight="bold")
axes[0].set_ylim(0, 100)
axes[0].set_title("1. Position + Length\n(most accurate)", fontsize=12)
axes[0].set_xlabel("Section")
axes[0].set_ylabel("Average Marks")

# 2 - bubble: area encodes the number (hard to compare)
sizes = [m * 14 for m in marks]
sc = axes[1].scatter(range(len(sections)), [1] * len(sections), s=sizes,
                     c=marks, cmap="Oranges", alpha=0.75, edgecolors="black")
for i, m in enumerate(marks):
    axes[1].annotate(f"{m:.1f}", (i, 1), ha="center", va="center",
                     fontweight="bold", fontsize=9)
axes[1].set_xticks(range(len(sections)))
axes[1].set_xticklabels(sections)
axes[1].set_yticks([])
axes[1].set_title("2. Area + Colour\n(weak for exact values)", fontsize=12)
axes[1].set_xlabel("Section")
fig.colorbar(sc, ax=axes[1], label="Marks")

# 3 - heatmap: colour intensity only (worst for precise reading)
heat = np.array(marks).reshape(1, -1)
im = axes[2].imshow(heat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
axes[2].set_xticks(range(len(sections)))
axes[2].set_xticklabels(sections)
axes[2].set_yticks([])
axes[2].set_title("3. Colour intensity\n(least accurate)", fontsize=12)
axes[2].set_xlabel("Section")
for i, m in enumerate(marks):
    axes[2].text(i, 0, f"{m:.1f}", ha="center", va="center", fontweight="bold")
fig.colorbar(im, ax=axes[2], label="Marks")

fig.suptitle("Same Data, Three Encodings", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q04_position_vs_size.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# CLEVELAND & McGILL HIERARCHY (1984 experiment on graphical perception)
# ---------------------------------------------------------------------------
ranking = [
    (1, "Position on common scale", "Bar chart / dot plot - error ~5%"),
    (2, "Position on non-aligned scale", "Two separate axes - error ~10%"),
    (3, "Length", "Bars, line segments - error ~13%"),
    (4, "Direction / angle", "Slopes - error ~20%"),
    (5, "Area (size)", "Bubbles, circles - error ~27%"),
    (6, "Volume / 3D", "3D shapes - error >35%"),
    (7, "Colour saturation / hue", "Heatmaps - worst for quantity"),
]
print("\nCLEVELAND & McGILL EFFECTIVENESS RANKING")
for rank, name, note in ranking:
    print(f"  {rank}. {name:30s} <- {note}")

print("\nWHY POSITION BEATS SIZE")
print("- Human area judgement is compressive (Stevens' power law: perceived")
print("  area ~ physical area^0.7), so doubling a number does NOT look doubled.")
print("- A 58 vs 62 bubble pair is almost indistinguishable, while two bars on")
print("  a common axis show the same gap instantly and accurately.")

# ---------------------------------------------------------------------------
# REDESIGN of the least effective encoding (heatmap -> bar + value labels)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# old: colour-only heatmap
sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd", vmin=0, vmax=100,
            xticklabels=sections, yticklabels=["avg marks"], ax=axes[0],
            cbar_kws={"label": "Marks"})
axes[0].set_title("Before: heatmap (colour only)", fontsize=12)

# new: position-based bars with labels
axes[1].bar(sections, marks, color="#2196F3", edgecolor="black", width=0.6)
for i, v in enumerate(marks):
    axes[1].text(i, v + 1, f"{v:.1f}", ha="center", fontweight="bold")
axes[1].set_ylim(0, 100)
axes[1].axhline(np.mean(marks), color="#FF5722", linestyle="--",
                label=f"Overall mean {np.mean(marks):.1f}")
axes[1].set_title("After: position on a common scale", fontsize=12)
axes[1].set_xlabel("Section")
axes[1].set_ylabel("Average Marks")
axes[1].legend()

fig.suptitle("Redesign: from colour intensity to position", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q04_redesign.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
print("\nINTERPRETATION")
print(f"- Section averages: " + ", ".join(f"{s}={m:.1f}" for s, m in zip(sections, marks)))
print("- All three charts show the SAME numbers, but only the bar chart lets you")
print("  read differences accurately without a colour legend.")
print("- Rule of thumb: encode quantities with position/length; use colour for")
print("  categories or emphasis, not for precise values.")
