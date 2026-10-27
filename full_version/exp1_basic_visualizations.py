# EXPERIMENT 1: Bar Chart, Histogram, Box Plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

df = pd.read_csv('cse_student_performance_1200.csv')
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1) Bar chart: average final score by year
year_avg = df.groupby('year')['final_exam_score'].mean().reset_index()
colors_bar = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
bars = axes[0].bar(year_avg['year'].astype(str), year_avg['final_exam_score'],
                   color=colors_bar, edgecolor='black', width=0.6)
for bar, val in zip(bars, year_avg['final_exam_score']):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')
axes[0].set_title('Average Final Exam Score by Year', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Year of Study'); axes[0].set_ylabel('Average Final Exam Score')
axes[0].set_ylim(0, 100)
axes[0].legend(handles=[Patch(facecolor=c, edgecolor='black', label=f'Year {y}')
                        for c, y in zip(colors_bar, [1, 2, 3, 4])], loc='upper right')

# 2) Histogram: distribution of final scores
axes[1].hist(df['final_exam_score'], bins=25, color='#42A5F5', edgecolor='black',
             alpha=0.85, label='Final Exam Scores')
mean_s, med_s = df['final_exam_score'].mean(), df['final_exam_score'].median()
axes[1].axvline(mean_s, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_s:.1f}')
axes[1].axvline(med_s, color='green', linestyle=':', linewidth=2, label=f'Median = {med_s:.1f}')
axes[1].set_title('Distribution of Final Exam Scores', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Final Exam Score'); axes[1].set_ylabel('Frequency'); axes[1].legend()

# 3) Box plot: final score by section
sec_colors = {'A': '#66BB6A', 'B': '#FFA726', 'C': '#AB47BC'}
sns.boxplot(x='section', y='final_exam_score', hue='section', data=df, ax=axes[2],
            palette=sec_colors, width=0.5, legend=False, order=['A', 'B', 'C'])
axes[2].set_title('Final Exam Score by Section', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Section'); axes[2].set_ylabel('Final Exam Score')
axes[2].legend(handles=[Patch(facecolor=sec_colors[s], label=f'Section {s}') for s in 'ABC'],
               loc='lower right')

plt.tight_layout()
plt.savefig('exp1_basic_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nAverage Final Exam Score by Year:\n", year_avg.to_string(index=False))
print(f"\nOverall - Mean: {mean_s:.2f}, Median: {med_s:.2f}, Std: {df['final_exam_score'].std():.2f}")
print("\nScore by Section:\n", df.groupby('section')['final_exam_score'].describe().round(2))
