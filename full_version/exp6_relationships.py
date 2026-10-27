# EXPERIMENT 6: Scatter plots, pair plot, correlation heatmap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cse_student_performance_1200.csv')
sns.set_style("whitegrid")
order = ['Poor', 'Average', 'Good', 'Excellent']
pal = {'Poor': '#EF5350', 'Average': '#FFA726', 'Good': '#66BB6A', 'Excellent': '#42A5F5'}
rel_vars = ['attendance_percentage', 'study_hours_per_day', 'assignment_score', 'internal_exam_1',
            'internal_exam_2', 'lab_score', 'final_exam_score', 'overall_score']

# Figure 1: scatter plots
fig1, ax = plt.subplots(2, 2, figsize=(14, 12))
sns.scatterplot(x='study_hours_per_day', y='final_exam_score', data=df, hue='performance_category',
                hue_order=order, palette=pal, alpha=0.5, s=30, ax=ax[0, 0])
ax[0, 0].set_title('Study Hours vs Final Exam Score', fontweight='bold')
sns.scatterplot(x='attendance_percentage', y='final_exam_score', data=df, hue='performance_category',
                hue_order=order, palette=pal, alpha=0.5, s=30, ax=ax[0, 1])
ax[0, 1].set_title('Attendance vs Final Exam Score', fontweight='bold')
sns.scatterplot(x='internal_exam_1', y='internal_exam_2', data=df, alpha=0.5, s=30, color='#7E57C2', ax=ax[1, 0])
ax[1, 0].plot([0, 100], [0, 100], 'r--', label='IE1 = IE2'); ax[1, 0].legend()
ax[1, 0].set_title('Internal Exam 1 vs Internal Exam 2', fontweight='bold')
sns.scatterplot(x='assignment_score', y='lab_score', data=df, alpha=0.5, s=30, color='#26A69A', ax=ax[1, 1])
ax[1, 1].set_title('Assignment Score vs Lab Score', fontweight='bold')
plt.tight_layout(); plt.savefig('exp6_scatter_plots.png', dpi=150, bbox_inches='tight'); plt.show()

# Figure 2: pair plot
g = sns.pairplot(df[['study_hours_per_day', 'attendance_percentage', 'assignment_score',
                     'final_exam_score', 'performance_category']],
                 hue='performance_category', hue_order=order, palette=pal, diag_kind='kde',
                 plot_kws={'alpha': 0.4, 's': 20}, height=2.5)
g.figure.suptitle('Pair Plot of Key Academic Variables', fontweight='bold', y=1.02)
plt.savefig('exp6_pairplot.png', dpi=150, bbox_inches='tight'); plt.show()

# Figure 3: correlation heatmap
corr = df[rel_vars].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig3, ax3 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax3, cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
ax3.set_title('Correlation Heatmap of Academic Variables', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('exp6_correlation_heatmap.png', dpi=150, bbox_inches='tight'); plt.show()

print("Strongest correlations (|r| > 0.5):")
for i in range(len(rel_vars)):
    for j in range(i + 1, len(rel_vars)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            print(f"  {rel_vars[i]} <-> {rel_vars[j]}: r = {r:.3f} "
                  f"({'strong' if abs(r) > 0.7 else 'moderate'} {'positive' if r > 0 else 'negative'})")
print("\nNOTE: correlation does NOT prove causation.")
