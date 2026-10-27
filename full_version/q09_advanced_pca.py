"""
Question 9: PCA (Comprehensive)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Full PCA pipeline: all components, scree plot with Kaiser criterion,
cumulative variance with 80%/95% thresholds, a biplot with loading
vectors, and a 3D scatter of the first three components.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D   # registers the 3D projection

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CSV = os.path.join(ROOT, "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(ROOT, "outputs", "advanced")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")

features = ["attendance_percentage", "study_hours_per_week", "lab_marks",
            "assignment_marks", "internal_marks", "previous_semester_cgpa",
            "online_assessment_score", "library_visits_per_month",
            "coding_practice_hours_per_week", "support_sessions_attended",
            "final_exam_marks"]
X = df[features]
X_scaled = StandardScaler().fit_transform(X)      # PCA needs standardized data

# ---------------------------------------------------------------------------
# PCA with ALL 11 components (not just 2)
# ---------------------------------------------------------------------------
pca = PCA(n_components=len(features))
scores = pca.fit_transform(X_scaled)
evr = pca.explained_variance_ratio_
cum = np.cumsum(evr)
n80 = int(np.searchsorted(cum, 0.80) + 1)         # components for 80% variance
n95 = int(np.searchsorted(cum, 0.95) + 1)         # components for 95% variance
kaiser = int((pca.explained_variance_ > 1).sum())  # Kaiser: eigenvalue > 1

print("=" * 60)
print("PCA - EXPLAINED VARIANCE")
print("=" * 60)
for i, (v, c) in enumerate(zip(evr, cum), 1):
    print(f"  PC{i:<2d} {v * 100:5.1f}%   cumulative {c * 100:5.1f}%")
print(f"- Kaiser criterion (eigenvalue>1): keep {kaiser} components")
print(f"- {n80} components needed for 80% variance")
print(f"- {n95} components needed for 95% variance")
print(f"- 11 features -> 2 PCs retains {cum[1] * 100:.1f}% "
      f"(loses {(1 - cum[1]) * 100:.1f}%)")

# ---------------------------------------------------------------------------
# FIGURE 1 - SCREE PLOT + CUMULATIVE VARIANCE
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
comps = np.arange(1, len(features) + 1)

axes[0].bar(comps, evr * 100, color="#2196F3", edgecolor="black",
            label="variance explained")
axes[0].plot(comps, pca.explained_variance_, "o-", color="#FF5722",
             label="eigenvalue (right of Kaiser line = 1)", markersize=5)
axes[0].axhline(1, color="gray", linestyle="--", linewidth=1.5)
axes[0].set_title("Scree Plot (Kaiser criterion: eigenvalue > 1)", fontsize=12)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%) / Eigenvalue")
axes[0].legend(fontsize=9)

axes[1].plot(comps, cum * 100, "o-", color="#4CAF50", markersize=5)
axes[1].axhline(80, color="orange", linestyle="--", label="80% threshold")
axes[1].axhline(95, color="red", linestyle="--", label="95% threshold")
axes[1].axvline(n80, color="orange", linestyle=":", alpha=0.6)
axes[1].axvline(n95, color="red", linestyle=":", alpha=0.6)
axes[1].set_title("Cumulative Explained Variance", fontsize=12)
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance (%)")
axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q09_scree_plot.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 2 - BIPLOT + 3D SCATTER
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(1, 2, 1)

grade_order = ["O", "A+", "A", "B+", "B", "C", "F"]
palette = dict(zip(grade_order, sns.color_palette("viridis", len(grade_order))))
for grade in grade_order:
    mask = df["final_grade"] == grade
    ax1.scatter(scores[mask, 0], scores[mask, 1], s=22, alpha=0.65,
                color=palette[grade], label=grade, edgecolors="none")

# loading vectors: show which original features push students in each direction
loadings = pca.components_[:2].T                 # 11 features x 2 PCs
scale = 4.0                                     # arrows scaled for visibility
for i, feat in enumerate(features):
    ax1.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale,
              color="#FF5722", width=0.03, head_width=0.35, alpha=0.85)
    ax1.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15,
             feat.replace("_per_week", "").replace("_percentage", ""),
             fontsize=7.5, ha="center", color="black")
ax1.set_xlabel(f"PC1 ({evr[0] * 100:.1f}% variance)")
ax1.set_ylabel(f"PC2 ({evr[1] * 100:.1f}% variance)")
ax1.set_title("Biplot: scores + feature loadings", fontsize=12)
ax1.axhline(0, color="gray", linewidth=0.5)
ax1.axvline(0, color="gray", linewidth=0.5)
ax1.legend(title="Grade", fontsize=8, loc="lower right")

# 3D scatter of the first three components
ax2 = fig.add_subplot(1, 2, 2, projection="3d")
for grade in grade_order:
    mask = df["final_grade"] == grade
    ax2.scatter(scores[mask, 0], scores[mask, 1], scores[mask, 2],
                s=18, alpha=0.7, color=palette[grade], label=grade)
ax2.set_xlabel(f"PC1 {evr[0] * 100:.0f}%")
ax2.set_ylabel(f"PC2 {evr[1] * 100:.0f}%")
ax2.set_zlabel(f"PC3 {evr[2] * 100:.0f}%")
ax2.set_title("First 3 Principal Components", fontsize=12)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "q09_biplot.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# LOADING MATRIX + INTERPRETATION
# ---------------------------------------------------------------------------
print("\nLOADING MATRIX (first 3 components)")
load_df = pd.DataFrame(
    pca.components_[:3].T,
    index=features, columns=["PC1", "PC2", "PC3"]).round(3)
print(load_df.to_string())
print("\nTOP CONTRIBUTORS")
for pc in ["PC1", "PC2", "PC3"]:
    top = load_df[pc].abs().sort_values(ascending=False).index[:3]
    print(f"  {pc}: " + ", ".join(
        f"{f} ({load_df.loc[f, pc]:+.2f})" for f in top))

print("\nINTERPRETATION")
print(f"- PC1 ({evr[0] * 100:.1f}%) is the main 'overall performance' axis:")
print("  marks, internals and study load push one way, support sessions the other.")
print(f"- Keeping only 2 PCs retains {cum[1] * 100:.1f}% of the information -")
print(f"  {(1 - cum[1]) * 100:.1f}% is discarded for the convenience of a 2D picture.")
print("- Loadings tell you WHAT each component means; long arrows = strong influence.")
