"""
Question 10: t-SNE and UMAP (Comparison Study)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Runs t-SNE at perplexity 5/30/50 and UMAP at n_neighbors 5/15/50,
builds a 2x3 comparison grid, scores each embedding with the silhouette
coefficient, and summarizes the strengths/weaknesses of both methods.
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")           # silence UMAP/t-SNE user warnings

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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
X = StandardScaler().fit_transform(df[features])
labels = df["final_grade"].values            # used as reference groups for scoring
grade_order = ["O", "A+", "A", "B+", "B", "C", "F"]


def silhouette(emb):
    """Silhouette of the embedding, using final_grade as reference labels.
    Higher = grade groups are better separated in that 2D view."""
    return silhouette_score(emb, labels)


# ---------------------------------------------------------------------------
# RUN BOTH METHODS ACROSS THEIR KEY PARAMETER
# ---------------------------------------------------------------------------
TSNE_PARAMS = [5, 30, 50]          # perplexity values
UMAP_PARAMS = [5, 15, 50]          # n_neighbors values

tsne_embs, tsne_scores = [], []
print("Running t-SNE (3 fits, ~5-15 s)...")
for perp in TSNE_PARAMS:
    emb = TSNE(n_components=2, random_state=42, perplexity=perp,
               init="pca").fit_transform(X)
    tsne_embs.append(emb)
    tsne_scores.append(silhouette(emb))

umap_embs, umap_scores = [], []
if UMAP_AVAILABLE:
    print("Running UMAP (3 fits, ~10-30 s the first time)...")
    for nn in UMAP_PARAMS:
        emb = UMAP(n_components=2, random_state=42, n_neighbors=nn).fit_transform(X)
        umap_embs.append(emb)
        umap_scores.append(silhouette(emb))
else:
    print("UMAP not installed -> pip install umap-learn  (Kaggle: !pip install umap-learn)")

# ---------------------------------------------------------------------------
# FIGURE 1 - 2x3 COMPARISON GRID
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for i, (perp, emb, score) in enumerate(zip(TSNE_PARAMS, tsne_embs, tsne_scores)):
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=df["final_grade"],
                    hue_order=grade_order, palette="viridis",
                    s=22, alpha=0.7, legend=(i == 0), ax=axes[0, i])
    axes[0, i].set_title(f"t-SNE perplexity={perp}\nsilhouette={score:.3f}",
                         fontsize=11)
    axes[0, i].set_xlabel("dim 1")
    axes[0, i].set_ylabel("dim 2")

if UMAP_AVAILABLE:
    for i, (nn, emb, score) in enumerate(zip(UMAP_PARAMS, umap_embs, umap_scores)):
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=df["final_grade"],
                        hue_order=grade_order, palette="viridis",
                        s=22, alpha=0.7, legend=(i == 0), ax=axes[1, i])
        axes[1, i].set_title(f"UMAP n_neighbors={nn}\nsilhouette={score:.3f}",
                             fontsize=11)
        axes[1, i].set_xlabel("dim 1")
        axes[1, i].set_ylabel("dim 2")
else:
    for i, nn in enumerate(UMAP_PARAMS):
        axes[1, i].text(0.5, 0.5, "UMAP not installed", ha="center",
                        va="center", transform=axes[1, i].transAxes, fontsize=12)
        axes[1, i].set_title(f"UMAP n_neighbors={nn}", fontsize=11)
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

fig.suptitle("t-SNE vs UMAP Across Parameters (colored by final grade)",
             fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q10_comparison.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# FIGURE 2 - PARAMETER STUDY: silhouette vs parameter value
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(TSNE_PARAMS, tsne_scores, "o-", color="#2196F3", linewidth=2,
        markersize=8, label="t-SNE (perplexity)")
if UMAP_AVAILABLE:
    ax.plot(UMAP_PARAMS, umap_scores, "s-", color="#FF5722", linewidth=2,
            markersize=8, label="UMAP (n_neighbors)")
ax.set_xlabel("Parameter value")
ax.set_ylabel("Silhouette score (grade separation)")
ax.set_title("Cluster Separation vs Hyper-parameter", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q10_parameter_study.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
print("\nSILHOUETTE SCORES (higher = grades better separated)")
print("  t-SNE:", dict(zip(TSNE_PARAMS, np.round(tsne_scores, 3))))
if UMAP_AVAILABLE:
    print("  UMAP :", dict(zip(UMAP_PARAMS, np.round(umap_scores, 3))))

print("\nPARAMETER EFFECTS")
print("- t-SNE perplexity balances LOCAL vs GLOBAL structure: small values")
print("  (5) make tiny islands; large values (50) blur groups together.")
print("- UMAP n_neighbors does the same job: small = local detail,")
print("  large = more of the big-picture geometry preserved.")
print("\nt-SNE vs UMAP")
print("- t-SNE: excellent local clusters, but distances BETWEEN clusters and")
print("  their relative sizes are essentially meaningless.")
print("- UMAP: preserves more global structure, usually faster on large data,")
print("  and keeps more of the continuum between groups.")
print("\nLIMITATIONS - read before publishing any embedding")
print("- These maps are for VISUALIZATION only; never cluster or classify on them.")
print("- Different seeds/parameters give different pictures of the SAME data.")
print("- Silhouette above uses grade labels only as a reference for comparison -")
print("  it does NOT prove students naturally form exactly 7 clusters.")
print("- If UMAP is missing: pip install umap-learn  (Kaggle: !pip install umap-learn)")
