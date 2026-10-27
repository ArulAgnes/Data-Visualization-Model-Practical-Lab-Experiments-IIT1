# Exp 7: regression line, confidence interval, residuals
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
df = pd.read_csv('cse_student_performance_1200.csv')
x, y = df['study_hours_per_day'], df['final_exam_score']
res = stats.linregress(x, y)
resid = y - (res.slope * x + res.intercept)
print(f"y = {res.slope:.2f}x + {res.intercept:.2f} | R2 = {res.rvalue**2:.3f} | p = {res.pvalue:.2e}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
sns.regplot(x=x, y=y, ci=95, ax=ax[0], scatter_kws={'alpha': .3}, line_kws={'color': 'red'})
ax[0].set_title('Regression with 95% CI')
ax[1].scatter(res.slope * x + res.intercept, resid, alpha=.3); ax[1].axhline(0, color='red', linestyle='--')
ax[1].set(title='Residuals vs Fitted', xlabel='Fitted', ylabel='Residual')
stats.probplot(resid, plot=ax[2]); ax[2].set_title('Q-Q Plot')

plt.tight_layout(); plt.savefig('exp7_regression_analysis.png', dpi=150); plt.show()
