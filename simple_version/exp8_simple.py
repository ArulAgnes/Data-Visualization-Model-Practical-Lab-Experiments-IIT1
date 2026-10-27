# Exp 8: rolling average + uncertainty band
import pandas as pd, numpy as np, matplotlib.pyplot as plt
df = pd.read_csv('cse_student_performance_1200.csv', parse_dates=['assessment_date'])
d = df.groupby('assessment_date')['final_exam_score'].agg(['mean', 'std', 'count'])
d['ci'] = 1.96 * d['std'] / np.sqrt(d['count'])       # 95% confidence margin
d['rolling'] = d['mean'].rolling(5, center=True).mean()

plt.figure(figsize=(12, 6))
plt.plot(d.index, d['mean'], 'o-', alpha=.5, label='Daily mean')
plt.plot(d.index, d['rolling'], linewidth=3, color='navy', label='5-point rolling average')
plt.fill_between(d.index, d['mean'] - d['ci'], d['mean'] + d['ci'], alpha=.2, label='95% CI')
plt.title('Final Score Over Time with Rolling Average and Uncertainty')
plt.xlabel('Date'); plt.ylabel('Average Final Score'); plt.legend()
plt.tight_layout(); plt.savefig('exp8_rolling_avg_uncertainty.png', dpi=150); plt.show()
print('Highest variability date:', d['std'].idxmax().date())
