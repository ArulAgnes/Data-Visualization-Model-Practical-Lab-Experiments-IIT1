# Exp 9: PCA
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv('cse_student_performance_1200.csv')
cols = ['attendance_percentage', 'study_hours_per_day', 'assignment_score', 'internal_exam_1', 'internal_exam_2',
        'lab_score', 'previous_semester_gpa', 'library_visits_per_month', 'coding_problems_solved', 'final_exam_score']
pca = PCA().fit(StandardScaler().fit_transform(df[cols]))
pcs = pca.transform(StandardScaler().fit_transform(df[cols]))
var = pca.explained_variance_ratio_ * 100
print('Explained variance %:', var.round(1), '\nCumulative %:', var.cumsum().round(1))

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].bar(range(1, 11), var); ax[0].plot(range(1, 11), var.cumsum(), 'ro-')
ax[0].set(title='Scree Plot', xlabel='Component', ylabel='Variance %')
for cat in ['Poor', 'Average', 'Good', 'Excellent']:
    m = df['performance_category'] == cat
    ax[1].scatter(pcs[m, 0], pcs[m, 1], label=cat, alpha=.5, s=20)
ax[1].set(title='PC1 vs PC2', xlabel=f'PC1 ({var[0]:.1f}%)', ylabel=f'PC2 ({var[1]:.1f}%)'); ax[1].legend()
plt.tight_layout(); plt.savefig('exp9_pca_analysis.png', dpi=150); plt.show()
