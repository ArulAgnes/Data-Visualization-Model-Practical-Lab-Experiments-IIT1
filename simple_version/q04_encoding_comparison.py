"""
Question 4: Compare Two Graphical Encodings
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program shows the same data encoded two ways - bar chart (position +
length) vs bubble chart (size) - then explains which encoding is more
accurate and redesigns the weaker one.
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

# summary per section: average marks and average study hours
sec_stats = df.groupby("section").agg(
    avg_marks=("final_exam_marks", "mean"),
    avg_hours=("study_hours_per_week", "mean"),
).reset_index()
sec_stats["section_num"] = range(1, len(sec_stats) + 1)   # sections as 1,2,3,4
print(sec_stats.round(2))

# ================= ENCODING 1: BAR CHART (position + length) ================
plt.figure(figsize=(8, 5))
bars = plt.bar(sec_stats["section"], sec_stats["avg_marks"],
               color="#2196F3", edgecolor="black")
for bar, value in zip(bars, sec_stats["avg_marks"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{value:.1f}", ha="center", fontweight="bold")
plt.title("Encoding 1: Bar Chart (Position + Length)", fontsize=14)
plt.xlabel("Section")
plt.ylabel("Average Marks")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q04_encoding_1.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ================= ENCODING 2: BUBBLE CHART (size) ===========================
plt.figure(figsize=(8, 5))
# bubble AREA (not radius) is proportional to the marks value
sizes = sec_stats["avg_marks"] * 12
plt.scatter(sec_stats["section_num"], sec_stats["avg_hours"], s=sizes,
            c="#FF5722", alpha=0.65, edgecolors="black")
for x, y, m in zip(sec_stats["section_num"], sec_stats["avg_hours"],
                   sec_stats["avg_marks"]):
    plt.annotate(f"{m:.1f}", (x, y), ha="center", va="center",
                 fontweight="bold", color="white")
plt.title("Encoding 2: Bubble Chart (Size)", fontsize=14)
plt.xlabel("Section (1=A, 2=B, 3=C, 4=D)")
plt.ylabel("Average Study Hours / Week")
plt.xticks(sec_stats["section_num"], sec_stats["section"])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q04_encoding_2.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ================= COMPARISON AND REDESIGN ===================================
print("\nWHICH ENCODING IS BETTER?")
print("Bar chart is MORE EFFECTIVE because:")
print("- Human eyes compare bar LENGTHS much more accurately than circle AREAS")
print("- Position on a common scale is the most accurate visual encoding")
print("- Bubble sizes are hard to compare precisely (area doubles look ~same)")

# REDESIGN: bubble chart replaced by a clean 2-panel bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(sec_stats["section"], sec_stats["avg_marks"],
            color="#2196F3", edgecolor="black")
axes[0].set(title="Average Marks by Section", xlabel="Section",
            ylabel="Average Marks", ylim=(0, 100))
axes[1].bar(sec_stats["section"], sec_stats["avg_hours"],
            color="#4CAF50", edgecolor="black")
axes[1].set(title="Average Study Hours by Section", xlabel="Section",
            ylabel="Hours per Week", ylim=(0, 20))
fig.suptitle("Redesign: Two Honest Bar Charts (one scale each)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q04_redesigned.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

print("\nINTERPRETATION")
print("- The redesign uses one clear scale per panel instead of guessing from areas.")
print("- Cleveland & McGill rank: position > length > area > colour for numbers.")
