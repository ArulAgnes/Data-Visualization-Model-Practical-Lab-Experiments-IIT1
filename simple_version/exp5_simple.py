# Exp 5: EDA - histogram+density, box plot, violin plot
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('cse_student_performance_1200.csv')
cols = ['attendance_percentage', 'study_hours_per_day', 'assignment_score',
        'internal_exam_1', 'internal_exam_2', 'final_exam_score']

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, c in zip(ax.flat, cols): sns.histplot(df[c], kde=True, ax=a); a.set_title(c)
plt.tight_layout(); plt.savefig('exp5_histograms_density.png', dpi=150); plt.show()

df[cols].plot(kind='box', subplots=True, layout=(2, 3), figsize=(15, 8))
plt.tight_layout(); plt.savefig('exp5_boxplots.png', dpi=150); plt.show()

fig, ax = plt.subplots(2, 3, figsize=(15, 8))
order = ['Poor', 'Average', 'Good', 'Excellent']
for a, c in zip(ax.flat, cols): sns.violinplot(x='performance_category', y=c, data=df, order=order, ax=a, cut=0); a.set_title(c)
plt.tight_layout(); plt.savefig('exp5_violinplots.png', dpi=150); plt.show()

print(df[cols].describe().T.join(df[cols].skew().rename('skew')).round(2))
