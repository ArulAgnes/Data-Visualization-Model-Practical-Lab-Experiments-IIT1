"""
Question 9: PCA Visualization
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program reduces 11 numerical features to 2 principal components so
all 1200 students can be drawn on one scatter plot, colored by grade.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- paths are anchored to THIS file so the program runs from any folder ---
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "..", "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(BASE, "..", "outputs", "simple")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")

# Step 1: only REAL numerical features (never student_id!)
features = ["attendance_percentage", "study_hours_per_week", "lab_marks",
            "assignment_marks", "internal_marks", "previous_semester_cgpa",
            "online_assessment_score", "library_visits_per_month",
            "coding_practice_hours_per_week", "support_sessions_attended",
            "final_exam_marks"]
X = df[features]
print("Missing values in features:", int(X.isnull().sum().sum()))

# Step 2: standardize - PCA is scale sensitive!
print("Why standardize? A 0-100 column would dominate a 0-10 column.")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)          # each feature -> mean 0, std 1

# Step 3: project every student down to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

var1, var2 = pca.explained_variance_ratio_   # share of variance per component
print(f"PC1 explained variance: {var1 * 100:.1f}%")
print(f"PC2 explained variance: {var2 * 100:.1f}%")
print(f"Cumulative:            {(var1 + var2) * 100:.1f}%")

# ---------------- CHART 1: PCA SCATTER --------------------------------------
grade_order = ["O", "A+", "A", "B+", "B", "C", "F"]
plt.figure(figsize=(9, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df["final_grade"],
                hue_order=grade_order, palette="viridis", alpha=0.75, s=45)
plt.title("PCA: Students in 2D Space (colored by Grade)", fontsize=14)
plt.xlabel(f"Principal Component 1 ({var1 * 100:.1f}% variance)")
plt.ylabel(f"Principal Component 2 ({var2 * 100:.1f}% variance)")
plt.legend(title="Grade", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q09_pca_scatter.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- CHART 2: VARIANCE BAR CHART -------------------------------
plt.figure(figsize=(7, 4))
plt.bar(["PC1", "PC2"], [var1 * 100, var2 * 100],
        color=["#2196F3", "#4CAF50"], edgecolor="black")
plt.title("Explained Variance per Component", fontsize=14)
plt.ylabel("Variance Explained (%)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q09_variance.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- INTERPRETATION --------------------------------------------
print("\nINTERPRETATION")
print(f"- PC1 explains {var1 * 100:.1f}% and PC2 explains {var2 * 100:.1f}% "
      f"of total variance")
print(f"- Together they capture {(var1 + var2) * 100:.1f}% - "
      "the rest of the information is lost in 2D")
print("- Students with grade O tend to sit on the HIGH side of PC1")
print("- PCA finds the directions where the data varies the most")
print("- 11 dimensions -> 2 dimensions so we can SEE the whole dataset at once")
