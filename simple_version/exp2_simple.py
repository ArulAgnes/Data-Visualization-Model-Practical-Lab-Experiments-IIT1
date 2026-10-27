# Exp 2: time-series line chart + heatmap
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('cse_student_performance_1200.csv', parse_dates=['assessment_date'])
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

daily = df.groupby('assessment_date')['final_exam_score'].mean()
daily.plot(ax=ax[0], marker='o', color='royalblue')
ax[0].set(title='Average Final Score Over Time (2024)', xlabel='Date', ylabel='Average Score')

df['month'] = df['assessment_date'].dt.strftime('%m-%b')
sns.heatmap(df.pivot_table('final_exam_score', 'semester', 'month'), annot=True, fmt='.0f', cmap='YlOrRd', ax=ax[1])
ax[1].set(title='Average Final Score: Semester vs Month', xlabel='Month', ylabel='Semester')

plt.tight_layout(); plt.savefig('exp2_timeseries_visualization.png', dpi=150); plt.show()
