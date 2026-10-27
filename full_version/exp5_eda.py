# EXPERIMENT 5: Exploratory Data Analysis (histogram+KDE, box plot, violin plot)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cse_student_performance_1200.csv')
sns.set_style("whitegrid")
eda_vars = ['attendance_percentage', 'study_hours_per_day', 'assignment_score',
            'internal_exam_1', 'internal_exam_2', 'final_exam_score']
colors = ['#5C6BC0', '#26A69A', '#FFA726', '#EF5350', '#AB47BC', '#42A5F5']

# Figure 1: histograms + density
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10)); axes1 = axes1.flatten()
for i, var in enumerate(eda_vars):
    ax = axes1[i]
    sns.histplot(df[var], bins=30, kde=True, ax=ax, color=colors[i], edgecolor='black',
                 alpha=0.7, stat='density')
    ax.axvline(df[var].mean(), color='red', linestyle='--', label=f'Mean={df[var].mean():.1f}')
    ax.axvline(df[var].median(), color='green', linestyle=':', label=f'Median={df[var].median():.1f}')
    ax.set_title(var.replace('_', ' ').title(), fontweight='bold'); ax.set_xlabel(''); ax.legend(fontsize=8)
fig1.suptitle('Histograms with Density Overlay', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('exp5_histograms_density.png', dpi=150, bbox_inches='tight'); plt.show()

# Figure 2: box plots
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10)); axes2 = axes2.flatten()
for i, var in enumerate(eda_vars):
    ax = axes2[i]
    sns.boxplot(y=df[var], ax=ax, color=colors[i], width=0.4,
                flierprops=dict(marker='o', markersize=4, alpha=0.5, markerfacecolor='red'))
    q1, q3 = df[var].quantile(0.25), df[var].quantile(0.75)
    ax.text(0.65, 0.95, f'Q1={q1:.1f}\nQ3={q3:.1f}\nIQR={q3 - q1:.1f}', transform=ax.transAxes,
            fontsize=8, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title(var.replace('_', ' ').title(), fontweight='bold'); ax.set_ylabel('Value')
fig2.suptitle('Box Plots - Distribution and Outliers', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('exp5_boxplots.png', dpi=150, bbox_inches='tight'); plt.show()

# Figure 3: violin plots by performance category
order = ['Poor', 'Average', 'Good', 'Excellent']
pal = {'Poor': '#EF5350', 'Average': '#FFA726', 'Good': '#66BB6A', 'Excellent': '#42A5F5'}
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10)); axes3 = axes3.flatten()
for i, var in enumerate(eda_vars):
    sns.violinplot(x='performance_category', y=var, hue='performance_category', data=df,
                   ax=axes3[i], order=order, palette=pal, inner='box', cut=0, legend=False)
    axes3[i].set_title(var.replace('_', ' ').title(), fontweight='bold')
    axes3[i].set_xlabel('Performance Category'); axes3[i].set_ylabel('')
fig3.suptitle('Violin Plots by Performance Category', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig('exp5_violinplots.png', dpi=150, bbox_inches='tight'); plt.show()

# Statistical summary
summary = df[eda_vars].describe().T
summary['skewness'] = df[eda_vars].skew()
summary['kurtosis'] = df[eda_vars].kurtosis()
out = {}
for v in eda_vars:
    q1, q3 = df[v].quantile(0.25), df[v].quantile(0.75); iqr = q3 - q1
    out[v] = ((df[v] < q1 - 1.5 * iqr) | (df[v] > q3 + 1.5 * iqr)).sum()
summary['outliers'] = pd.Series(out)
print(summary.round(2).to_string())
