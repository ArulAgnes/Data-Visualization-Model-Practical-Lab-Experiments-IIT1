# EXPERIMENT 8: Rolling average + uncertainty bands
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

df = pd.read_csv('cse_student_performance_1200.csv')
df['assessment_date'] = pd.to_datetime(df['assessment_date'])
sns.set_style("whitegrid")

daily = df.groupby('assessment_date')['final_exam_score'].agg(['mean', 'std', 'count']).reset_index()
daily.columns = ['date', 'mean_score', 'std_score', 'count']
daily = daily.sort_values('date').reset_index(drop=True)
daily['se'] = daily['std_score'] / np.sqrt(daily['count'])
daily['ci_lower'] = daily['mean_score'] - 1.96 * daily['se']
daily['ci_upper'] = daily['mean_score'] + 1.96 * daily['se']
window = 5
daily['rolling_mean'] = daily['mean_score'].rolling(window, center=True).mean()
daily['rolling_std'] = daily['mean_score'].rolling(window, center=True).std()

fig, axes = plt.subplots(2, 1, figsize=(16, 12))
ax1 = axes[0]
ax1.plot(daily['date'], daily['mean_score'], marker='o', markersize=4, linewidth=1, alpha=0.5,
         color='#90CAF9', label='Daily Mean Score')
ax1.plot(daily['date'], daily['rolling_mean'], linewidth=2.5, color='#1565C0', label=f'{window}-Point Rolling Average')
ax1.fill_between(daily['date'], daily['ci_lower'], daily['ci_upper'], alpha=0.2, color='#42A5F5',
                 label='95% Confidence Interval')
v = daily['rolling_mean'].notna()
ax1.fill_between(daily.loc[v, 'date'], daily.loc[v, 'rolling_mean'] - daily.loc[v, 'rolling_std'],
                 daily.loc[v, 'rolling_mean'] + daily.loc[v, 'rolling_std'], alpha=0.15, color='orange',
                 label='Rolling ±1 Std Dev')
ax1.xaxis.set_major_locator(mdates.MonthLocator()); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax1.set_title('Average Final Exam Score Over Time with Rolling Average and Uncertainty', fontsize=14, fontweight='bold')
ax1.set_xlabel('Assessment Date'); ax1.set_ylabel('Average Final Exam Score'); ax1.legend(loc='lower right')

ax2 = axes[1]
ax2.bar(daily['date'], daily['std_score'], width=8, color='#EF5350', alpha=0.6, edgecolor='black',
        linewidth=0.3, label='Std Dev of Scores')
twin = ax2.twinx()
twin.plot(daily['date'], daily['count'], marker='s', markersize=4, color='#2E7D32', label='Number of Assessments')
twin.set_ylabel('Count of Assessments', color='#2E7D32')
ax2.xaxis.set_major_locator(mdates.MonthLocator()); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.set_title('Score Variability and Assessment Volume Over Time', fontsize=14, fontweight='bold')
ax2.set_xlabel('Assessment Date'); ax2.set_ylabel('Standard Deviation of Scores', color='#EF5350')
l1, n1 = ax2.get_legend_handles_labels(); l2, n2 = twin.get_legend_handles_labels()
ax2.legend(l1 + l2, n1 + n2, loc='upper right')

plt.tight_layout(); plt.savefig('exp8_rolling_avg_uncertainty.png', dpi=150, bbox_inches='tight'); plt.show()

hi, lo = daily.loc[daily['std_score'].idxmax()], daily.loc[daily['std_score'].idxmin()]
print("Assessment dates:", len(daily), "| rolling window:", window)
print(f"Highest variability: {hi['date'].date()}  std={hi['std_score']:.2f}  n={int(hi['count'])}")
print(f"Lowest  variability: {lo['date'].date()}  std={lo['std_score']:.2f}  n={int(lo['count'])}")
print("Wider CI = fewer assessments = less reliable mean.")
