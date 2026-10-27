# Exp 6: scatter plots, pair plot, correlation heatmap
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('cse_student_performance_1200.csv')

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
sns.scatterplot(x='study_hours_per_day', y='final_exam_score', hue='performance_category', data=df, ax=ax[0])
sns.scatterplot(x='attendance_percentage', y='final_exam_score', data=df, ax=ax[1])
sns.scatterplot(x='internal_exam_1', y='internal_exam_2', data=df, ax=ax[2])
plt.tight_layout(); plt.savefig('exp6_scatter_plots.png', dpi=150); plt.show()

sns.pairplot(df[['study_hours_per_day', 'attendance_percentage', 'assignment_score', 'final_exam_score',
                 'performance_category']], hue='performance_category', height=2.2)
plt.savefig('exp6_pairplot.png', dpi=150); plt.show()

cols = ['attendance_percentage', 'study_hours_per_day', 'assignment_score', 'internal_exam_1',
        'internal_exam_2', 'lab_score', 'final_exam_score', 'overall_score']
plt.figure(figsize=(10, 8))
sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Correlation Heatmap'); plt.tight_layout()
plt.savefig('exp6_correlation_heatmap.png', dpi=150); plt.show()
