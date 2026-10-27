# EXPERIMENT 7: Regression with confidence interval and residual analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

df = pd.read_csv('cse_student_performance_1200.csv')
sns.set_style("whitegrid")
x, y = df['study_hours_per_day'].values, df['final_exam_score'].values

slope, intercept, r, p, se = stats.linregress(x, y)
y_pred = slope * x + intercept
resid = y - y_pred
print(f"Equation : Final Score = {slope:.3f} x Study Hours + {intercept:.3f}")
print(f"r = {r:.4f} | R2 = {r**2:.4f} | p-value = {p:.2e} | slope SE = {se:.4f}")
print(f"Residuals: mean = {resid.mean():.4f}, std = {resid.std():.4f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1) Regression + 95% CI
sns.regplot(x='study_hours_per_day', y='final_exam_score', data=df, ax=axes[0, 0], ci=95,
            scatter_kws={'alpha': 0.3, 's': 20, 'color': '#42A5F5'},
            line_kws={'color': 'red', 'linewidth': 2})
axes[0, 0].text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}\nr = {r:.3f}\nR² = {r**2:.3f}',
                transform=axes[0, 0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
axes[0, 0].set_title('Study Hours vs Final Exam Score\n(red line = fit, shaded = 95% CI)', fontweight='bold')
axes[0, 0].set_xlabel('Study Hours Per Day'); axes[0, 0].set_ylabel('Final Exam Score')

# 2) Residuals vs fitted
axes[0, 1].scatter(y_pred, resid, alpha=0.3, s=20, color='#7E57C2')
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title('Residuals vs Fitted Values', fontweight='bold')
axes[0, 1].set_xlabel('Fitted Values'); axes[0, 1].set_ylabel('Residuals')

# 3) Residual histogram + normal curve
axes[1, 0].hist(resid, bins=30, density=True, color='#26A69A', edgecolor='black', alpha=0.8, label='Residuals')
xs = np.linspace(resid.min(), resid.max(), 100)
axes[1, 0].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), 'r-', linewidth=2, label='Normal curve')
axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
axes[1, 0].set_xlabel('Residual'); axes[1, 0].set_ylabel('Density'); axes[1, 0].legend()

# 4) Q-Q plot
stats.probplot(resid, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot of Residuals', fontweight='bold')

plt.tight_layout(); plt.savefig('exp7_regression_analysis.png', dpi=150, bbox_inches='tight'); plt.show()
print(f"\nStudy hours explain about {r**2 * 100:.1f}% of the variance in final exam score.")
