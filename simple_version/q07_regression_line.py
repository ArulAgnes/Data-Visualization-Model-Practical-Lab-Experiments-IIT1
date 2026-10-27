"""
Question 7: Linear Regression
Simple Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

This program fits a simple linear regression of final marks on study
hours, draws the fit line with a 95% confidence band, and checks the
residuals to judge the quality of the fit.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# --- paths are anchored to THIS file so the program runs from any folder ---
BASE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(BASE, "..", "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(BASE, "..", "outputs", "simple")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")

# Step 1: predictor X must be a 2-D table for sklearn -> use double brackets
X = df[["study_hours_per_week"]]
y = df["final_exam_marks"]

# Step 2: create the model and learn (fit) the best straight line
model = LinearRegression()
model.fit(X, y)

# Step 3: read the learned parameters
slope = model.coef_[0]                 # change in marks per extra study hour
intercept = model.intercept_           # predicted marks at 0 study hours
r2 = model.score(X, y)                 # R-squared = variance explained
y_pred = model.predict(X)              # predicted marks for every student

# ---------------- CHART 1: SCATTER + REGRESSION LINE ------------------------
plt.figure(figsize=(8, 5))
# regplot draws the points, the fit line AND a 95% confidence band
sns.regplot(x="study_hours_per_week", y="final_exam_marks", data=df,
            ci=95, line_kws={"color": "#FF5722", "linewidth": 2},
            scatter_kws={"alpha": 0.4, "color": "#2196F3"})
plt.title("Linear Regression: Study Hours -> Final Marks", fontsize=14)
plt.xlabel("Study Hours per Week")
plt.ylabel("Final Exam Marks")
# text box with the model summary
box = f"Slope = {slope:.2f}\nIntercept = {intercept:.1f}\nR2 = {r2:.3f}"
plt.text(0.05, 0.95, box, transform=plt.gca().transAxes, va="top",
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q07_regression.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- CHART 2: RESIDUAL PLOT ------------------------------------
residuals = y - y_pred                 # how far real marks fall from prediction
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color="#4CAF50")
plt.axhline(0, color="red", linestyle="--", linewidth=2)   # perfect = on this line
plt.title("Residual Plot", fontsize=14)
plt.xlabel("Predicted Marks")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "q07_residuals.png"), dpi=150, bbox_inches="tight")
# plt.show()
plt.close()

# ---------------- INTERPRETATION --------------------------------------------
print("INTERPRETATION")
print(f"- Slope: {slope:.2f} - for each additional study hour, marks increase "
      f"by {slope:.2f} points on average")
print(f"- Intercept: {intercept:.1f} - predicted marks for zero study hours "
      "(extrapolation, not observed)")
print(f"- R-squared: {r2:.3f} - study hours explain {r2 * 100:.1f}% of the "
      "variation in marks")
print(f"- Residual spread: std = {residuals.std():.1f} marks")
print("- Limitation: this does NOT prove studying CAUSES higher marks "
      "(correlation is not causation)")
print("- Residuals should be randomly scattered around zero for a good fit; "
      "any pattern means a straight line is not enough.")
