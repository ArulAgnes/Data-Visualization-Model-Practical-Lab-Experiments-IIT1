# EXPERIMENT 10: t-SNE and UMAP  (pip install scikit-learn umap-learn)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

df = pd.read_csv('cse_student_performance_1200.csv')
sns.set_style("whitegrid")
features = ['attendance_percentage', 'study_hours_per_day', 'assignment_score', 'internal_exam_1',
            'internal_exam_2', 'lab_score', 'previous_semester_gpa', 'library_visits_per_month',
            'coding_problems_solved', 'final_exam_score']
X = StandardScaler().fit_transform(df[features])
perf = df['performance_category'].values
order = ['Poor', 'Average', 'Good', 'Excellent']
pal = {'Poor': '#EF5350', 'Average': '#FFA726', 'Good': '#66BB6A', 'Excellent': '#42A5F5'}

def draw(ax, emb, title, xl, yl, legend=True):
    for cat in order:
        m = perf == cat
        ax.scatter(emb[m, 0], emb[m, 1], c=pal[cat], label=cat, alpha=0.5, s=25)
    ax.set_title(title, fontsize=13, fontweight='bold'); ax.set_xlabel(xl); ax.set_ylabel(yl)
    if legend: ax.legend(title='Performance')

print("Running t-SNE ...")
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, max_iter=1000, random_state=42, init='pca')
X_tsne = tsne.fit_transform(X)
print(f"t-SNE KL divergence: {tsne.kl_divergence_:.4f}")
print("Running UMAP ...")
X_umap = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
draw(axes[0], X_tsne, 't-SNE (perplexity=30)', 't-SNE 1', 't-SNE 2')
draw(axes[1], X_umap, 'UMAP (n_neighbors=15, min_dist=0.1)', 'UMAP 1', 'UMAP 2')
plt.tight_layout(); plt.savefig('exp10_tsne_umap_comparison.png', dpi=150, bbox_inches='tight'); plt.show()

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
for ax, perp in zip(axes2, [5, 30, 100]):
    print("t-SNE perplexity", perp)
    emb = TSNE(n_components=2, perplexity=perp, max_iter=1000, random_state=42, init='pca').fit_transform(X)
    draw(ax, emb, f't-SNE (perplexity={perp})', 'Dim 1', 'Dim 2', legend=(perp == 100))
fig2.suptitle('Effect of Perplexity on t-SNE', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('exp10_tsne_perplexity_comparison.png', dpi=150, bbox_inches='tight'); plt.show()

print("\nLimitations: distances/cluster sizes/gaps in 2D are NOT reliable; these are visualization tools, not clustering.")
