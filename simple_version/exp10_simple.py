# Exp 10: t-SNE and UMAP   (pip install umap-learn)
import pandas as pd, matplotlib.pyplot as plt, umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
df = pd.read_csv('cse_student_performance_1200.csv')
cols = ['attendance_percentage', 'study_hours_per_day', 'assignment_score', 'internal_exam_1', 'internal_exam_2',
        'lab_score', 'previous_semester_gpa', 'library_visits_per_month', 'coding_problems_solved', 'final_exam_score']
X = StandardScaler().fit_transform(df[cols])

def plot(ax, emb, title):
    for cat in ['Poor', 'Average', 'Good', 'Excellent']:
        m = (df['performance_category'] == cat).values
        ax.scatter(emb[m, 0], emb[m, 1], label=cat, alpha=.5, s=20)
    ax.set_title(title); ax.legend()

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
plot(ax[0], TSNE(perplexity=30, init='pca', random_state=42).fit_transform(X), 't-SNE (perplexity=30)')
plot(ax[1], umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X), 'UMAP')
plt.tight_layout(); plt.savefig('exp10_tsne_umap_comparison.png', dpi=150); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for a, p in zip(ax, [5, 30, 100]):
    plot(a, TSNE(perplexity=p, init='pca', random_state=42).fit_transform(X), f't-SNE perplexity={p}')
plt.tight_layout(); plt.savefig('exp10_tsne_perplexity_comparison.png', dpi=150); plt.show()
print("Note: distances and cluster sizes in 2D are not reliable.")
