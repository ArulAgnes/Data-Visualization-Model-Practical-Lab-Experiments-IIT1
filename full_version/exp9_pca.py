# EXPERIMENT 9: Principal Component Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('cse_student_performance_1200.csv')
sns.set_style("whitegrid")
features = ['attendance_percentage', 'study_hours_per_day', 'assignment_score', 'internal_exam_1',
            'internal_exam_2', 'lab_score', 'previous_semester_gpa', 'library_visits_per_month',
            'coding_problems_solved', 'final_exam_score']
perf = df['performance_category'].values

X_scaled = StandardScaler().fit_transform(df[features])
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
ev = pca.explained_variance_ratio_
cum = np.cumsum(ev)

print(f"{'Component':<10}{'Variance %':>12}{'Cumulative %':>15}")
for i, (a, b) in enumerate(zip(ev, cum), 1):
    print(f"PC{i:<8}{a * 100:>12.2f}{b * 100:>15.2f}")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
comps = range(1, len(ev) + 1)
axes[0].bar(comps, ev * 100, color='#42A5F5', edgecolor='black', alpha=0.8, label='Individual')
axes[0].plot(comps, cum * 100, 'ro-', label='Cumulative')
axes[0].axhline(80, color='green', linestyle='--', label='80% threshold')
axes[0].set_title('Scree Plot - Explained Variance', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_xticks(list(comps)); axes[0].legend()

pal = {'Poor': '#EF5350', 'Average': '#FFA726', 'Good': '#66BB6A', 'Excellent': '#42A5F5'}
for cat in ['Poor', 'Average', 'Good', 'Excellent']:
    m = perf == cat
    axes[1].scatter(X_pca[m, 0], X_pca[m, 1], c=pal[cat], label=cat, alpha=0.5, s=25)
axes[1].set_title(f'PCA - First Two Components ({(ev[0] + ev[1]) * 100:.1f}% variance)', fontsize=12, fontweight='bold')
axes[1].set_xlabel(f'PC1 ({ev[0] * 100:.1f}%)'); axes[1].set_ylabel(f'PC2 ({ev[1] * 100:.1f}%)')
axes[1].legend(title='Performance')

load = pca.components_[:2].T
for i, f in enumerate(features):
    axes[2].arrow(0, 0, load[i, 0], load[i, 1], head_width=0.03, head_length=0.02, fc='#5C6BC0', ec='#5C6BC0', alpha=0.7)
    axes[2].text(load[i, 0] * 1.1, load[i, 1] * 1.1, f.replace('_', '\n'), fontsize=7, ha='center', va='center')
axes[2].add_patch(plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--'))
axes[2].set_xlim(-1.2, 1.2); axes[2].set_ylim(-1.2, 1.2); axes[2].set_aspect('equal')
axes[2].set_title('PCA Loadings', fontsize=13, fontweight='bold')
axes[2].set_xlabel('PC1'); axes[2].set_ylabel('PC2')

plt.tight_layout(); plt.savefig('exp9_pca_analysis.png', dpi=150, bbox_inches='tight'); plt.show()
