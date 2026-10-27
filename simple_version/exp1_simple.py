# Exp 1: bar chart, histogram, box plot
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('cse_student_performance_1200.csv')
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

df.groupby('year')['final_exam_score'].mean().plot.bar(ax=ax[0], color='skyblue', edgecolor='black')
ax[0].set(title='Average Final Score by Year', xlabel='Year', ylabel='Average Score', ylim=(0, 100))

ax[1].hist(df['final_exam_score'], bins=25, color='steelblue', edgecolor='black')
ax[1].axvline(df['final_exam_score'].mean(), color='red', linestyle='--', label='Mean')
ax[1].set(title='Final Score Distribution', xlabel='Final Score', ylabel='Frequency'); ax[1].legend()

sns.boxplot(x='section', y='final_exam_score', data=df, ax=ax[2], order=['A', 'B', 'C'])
ax[2].set(title='Final Score by Section', xlabel='Section', ylabel='Final Score')

plt.tight_layout(); plt.savefig('exp1_basic_visualizations.png', dpi=150); plt.show()
print(df.groupby('year')['final_exam_score'].mean().round(2))
