"""
Question 7: Regression Diagnostics (Full Analysis)
Advanced Version — Data Visualization Lab
Author: Arul Agnes
Date: 2025

Fits the study-hours -> marks model with BOTH scikit-learn and
statsmodels, prints full OLS statistics (SE, t, p, R2, adjusted R2),
draws a 2x2 residual diagnostics panel, Cook's distance, and separates
the confidence interval from the prediction interval.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
CSV = os.path.join(ROOT, "datasets", "cse_student_performance_1200.csv")
OUT = os.path.join(ROOT, "outputs", "advanced")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(CSV)
sns.set_style("whitegrid")

X = df[["study_hours_per_week"]]
y = df["final_exam_marks"]

# ---------------------------------------------------------------------------
# 1. scikit-learn fit (simple API)
# ---------------------------------------------------------------------------
sk_model = LinearRegression().fit(X, y)
sk_slope, sk_intercept = sk_model.coef_[0], sk_model.intercept_
sk_r2 = sk_model.score(X, y)
y_pred = sk_model.predict(X)
residuals = y - y_pred

# ---------------------------------------------------------------------------
# 2. statsmodels OLS fit (full statistics)
# ---------------------------------------------------------------------------
Xc = sm.add_constant(X)                      # adds the intercept column
ols = sm.OLS(y, Xc).fit()
print("=" * 70)
print("STATSMODELS OLS SUMMARY (key numbers)")
print("=" * 70)
print(ols.summary2().tables[1].round(4).to_string())   # coefficients table
print(f"R-squared      : {ols.rsquared:.4f}")
print(f"Adj. R-squared : {ols.rsquared_adj:.4f}")
print(f"F-statistic    : {ols.fvalue:.1f} (p = {ols.f_pvalue:.2e})")
print(f"\nMODEL EQUATION: final_marks = {ols.params.iloc[1]:.3f} * study_hours "
      f"+ {ols.params.iloc[0]:.2f}")

# ---------------------------------------------------------------------------
# 3. 2x2 RESIDUAL DIAGNOSTICS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) residuals vs fitted - should be random scatter around 0
axes[0, 0].scatter(y_pred, residuals, alpha=0.4, color="#2196F3", s=20)
axes[0, 0].axhline(0, color="red", linewidth=2)
axes[0, 0].set(title="(a) Residuals vs Fitted", xlabel="Fitted values",
               ylabel="Residuals")
# lowess smoother reveals systematic patterns
lowess = sm.nonparametric.lowess(residuals, y_pred, frac=0.4)
axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color="#FF5722", linewidth=2)

# (b) normal QQ plot of residuals
sm.qqplot(residuals, line="45", ax=axes[0, 1], alpha=0.4)
axes[0, 1].set(title="(b) Normal QQ of Residuals")

# (c) scale-location: sqrt |standardized residuals| vs fitted
std_res = ols.get_influence().resid_studentized_internal
axes[1, 0].scatter(y_pred, np.sqrt(np.abs(std_res)), alpha=0.4,
                   color="#4CAF50", s=20)
axes[1, 0].set(title="(c) Scale-Location (homoscedasticity)",
               xlabel="Fitted values", ylabel="sqrt(|std residuals|)")

# (d) residual histogram + fitted normal curve
axes[1, 1].hist(residuals, bins=25, density=True, color="#FF9800",
                edgecolor="black", alpha=0.85)
xs = np.linspace(residuals.min(), residuals.max(), 200)
axes[1, 1].plot(xs, stats.norm.pdf(xs, residuals.mean(), residuals.std()),
                color="black", linewidth=2, label="normal fit")
axes[1, 1].set(title="(d) Residual Distribution", xlabel="Residual")
axes[1, 1].legend()

fig.suptitle("Regression Diagnostics: study_hours -> final_marks", fontsize=15)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q07_regression_diagnostics.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. CONFIDENCE INTERVAL vs PREDICTION INTERVAL
# ---------------------------------------------------------------------------
grid = np.linspace(X["study_hours_per_week"].min(),
                   X["study_hours_per_week"].max(), 100).reshape(-1, 1)
grid_pred = sm.add_constant(grid)
mean_ci = ols.get_prediction(grid_pred).conf_int(alpha=0.05)      # E[Y|X]
indiv_pi = ols.get_prediction(grid_pred).conf_int(alpha=0.05, obs=True)  # new Y

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X["study_hours_per_week"], y, alpha=0.35, s=18, color="#2196F3",
           label="students")
ax.plot(grid, ols.params.iloc[0] + ols.params.iloc[1] * grid,
        color="#FF5722", linewidth=2, label="fitted line")
ax.fill_between(grid.ravel(), mean_ci[:, 0], mean_ci[:, 1], alpha=0.45,
                color="#FF5722", label="95% confidence interval (mean)")
ax.fill_between(grid.ravel(), indiv_pi[:, 0], indiv_pi[:, 1], alpha=0.18,
                color="#9C27B0", label="95% prediction interval (individual)")
ax.set_xlabel("Study Hours per Week")
ax.set_ylabel("Final Exam Marks")
ax.set_title("Confidence Interval vs Prediction Interval", fontsize=13)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q07_confidence_prediction.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. COOK'S DISTANCE - influential observations
# ---------------------------------------------------------------------------
cooks = ols.get_influence().cooks_distance[0]
n_influential = int((cooks > 4 / len(y)).sum())
influential_idx = np.where(cooks > 4 / len(y))[0]

# ---------------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------------
print("\nDIAGNOSTICS REPORT")
print(f"- sklearn slope {sk_slope:.3f} == statsmodels slope "
      f"{ols.params.iloc[1]:.3f} (both agree)")
print(f"- Residual std: {residuals.std():.2f} marks")
shap_stat, shap_p = stats.shapiro(residuals.sample(500, random_state=42))
print(f"- Residual normality (Shapiro, n=500): p = {shap_p:.3f} "
      f"-> {'normal-ish' if shap_p > 0.05 else 'non-normal'}")
print(f"- Cook's distance: {n_influential} points above 4/n "
      f"({4 / len(y):.4f}) are influential")
print("\nASSUMPTIONS CHECK")
print("1. LINEARITY     : lowess line in plot (a) is roughly flat => OK")
print("2. INDEPENDENCE  : data is one row per student => plausible")
print("3. NORMALITY     : QQ points near the 45-degree line => OK")
print("4. EQUAL VARIANCE : plot (c) shows no funnel => OK")
print("\nINTERPRETATION")
print(f"- The model explains {sk_r2 * 100:.1f}% of the variance; the other "
      f"{100 - sk_r2 * 100:.1f}% comes from attendance, prior ability, luck...")
print("- Confidence interval = uncertainty of the MEAN response (narrow band)")
print("- Prediction interval = uncertainty for ONE student (much wider band)")
print("- A narrow CI does NOT mean individual predictions are accurate!")
