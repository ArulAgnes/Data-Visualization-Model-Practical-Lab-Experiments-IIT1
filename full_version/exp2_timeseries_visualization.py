# EXPERIMENT 2: Time-series line chart + temporal heatmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

df = pd.read_csv('cse_student_performance_1200.csv')
df['assessment_date'] = pd.to_datetime(df['assessment_date'])
sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# 1) Line chart with linear trend
daily = df.groupby('assessment_date')['final_exam_score'].mean().reset_index().sort_values('assessment_date')
axes[0].plot(daily['assessment_date'], daily['final_exam_score'], marker='o', markersize=5,
             linewidth=2, color='#1976D2', label='Average Final Exam Score', zorder=3)
x_num = mdates.date2num(daily['assessment_date'])
p = np.poly1d(np.polyfit(x_num, daily['final_exam_score'], 1))
axes[0].plot(daily['assessment_date'], p(x_num), '--', color='red', linewidth=1.5, label='Linear Trend')
axes[0].xaxis.set_major_locator(mdates.MonthLocator())
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
axes[0].set_title('Average Final Exam Score Over Time (2024)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Assessment Date'); axes[0].set_ylabel('Average Final Exam Score')
axes[0].legend(loc='upper left')

# 2) Heatmap: semester x month
df['month'] = df['assessment_date'].dt.month
pivot = df.pivot_table(values='final_exam_score', index='semester', columns='month', aggfunc='mean')
pivot.columns = [pd.Timestamp(2024, m, 1).strftime('%b') for m in pivot.columns]
sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5, linecolor='white',
            ax=axes[1], cbar_kws={'label': 'Average Final Exam Score'})
axes[1].set_title('Average Final Exam Score by Semester and Month', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Assessment Month'); axes[1].set_ylabel('Semester')

plt.tight_layout()
plt.savefig('exp2_timeseries_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Period: {df['assessment_date'].min().date()} to {df['assessment_date'].max().date()}")
print("Distinct assessment dates:", df['assessment_date'].nunique())
print("\nMonthly average scores:")
for m, s in df.groupby('month')['final_exam_score'].mean().items():
    print(f"  {pd.Timestamp(2024, m, 1).strftime('%B'):<10}: {s:.2f}")
print("\nNote: one year of data -> cannot claim seasonality.")
