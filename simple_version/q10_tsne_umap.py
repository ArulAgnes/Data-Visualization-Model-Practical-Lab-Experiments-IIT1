"""
Question 10: t-SNE and UMAP
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program projects 11 features into 2 dimensions with t-SNE and
(with a graceful fallback) UMAP to reveal student performance clusters.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# try to import UMAP - it is an optional package
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# --- paths are anchored to THIS file so the program runs from any folder ---
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "..", "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(BASE, "..", "outputs", "simple")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")

# same 11 numerical features as Question 9
features = ["attendance_percentage", "study_hours_per_week", "lab_marks",
            "assignment_marks", "internal_marks", "previous_semester_cgpa",
            "online_assessment_score", "library_visits_per_month",
            "coding_practice_hours_per_week", "support_sessions_attended",
            "final_exam_marks"]
X_scaled = StandardScaler().fit_transform(df[features])   # scale first!

grade_order = ["O", "A+", "A", "B+", "B", "C", "F"]

# ---------------- t-SNE ------------------------------------------------------
print("Running t-SNE (about 5-10 seconds)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
emb_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
sns.scatterplot(x=emb_tsne[:, 0], y=emb_tsne[:, 1], hue=df["final_grade"],
                hue_order=grade_order, palette="viridis", alpha=0.75, s=45)
plt.title("t-SNE: Student Performance Clusters", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Grade", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q10_tsne.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- UMAP (with fallback) --------------------------------------
if UMAP_AVAILABLE:
    print("Running UMAP (about 5-15 seconds)...")
    emb_umap = UMAP(n_components=2, random_state=42, n_neighbors=15).fit_transform(X_scaled)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x=emb_umap[:, 0], y=emb_umap[:, 1], hue=df["final_grade"],
                    hue_order=grade_order, palette="viridis", alpha=0.75, s=45)
    plt.title("UMAP: Student Performance Clusters", fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Grade", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q10_umap.png"), dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close()
else:
    print("UMAP not installed. Install with: pip install umap-learn")
    print("For Kaggle, run in a cell: !pip install umap-learn")
    # draw a placeholder image so the output folder stays complete
    plt.figure(figsize=(8, 5))
    plt.text(0.5, 0.55, "UMAP not available", ha="center", fontsize=20,
             fontweight="bold", transform=plt.gca().transAxes)
    plt.text(0.5, 0.35, "Install with:  pip install umap-learn",
             ha="center", fontsize=13, transform=plt.gca().transAxes)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "q10_umap.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ---------------- INTERPRETATION --------------------------------------------
print("\nINTERPRETATION")
print("- t-SNE groups similar students together in 2D")
print("- Students with similar grades tend to cluster together")
print("- WARNING: distances between clusters in t-SNE are NOT meaningful")
print("- WARNING: cluster sizes in t-SNE do NOT represent real group sizes")
print("- UMAP generally preserves global structure better than t-SNE")
print("- Both methods are for VISUALIZATION only - never for clustering decisions")
