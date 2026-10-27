# EXPERIMENT 4: Compare graphical encodings (bar vs bubble vs redesigned grouped bar)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cse_student_performance_1200.csv')
subj = df.groupby('subject').agg(avg_score=('final_exam_score', 'mean'),
                                 avg_attendance=('attendance_percentage', 'mean'),
                                 count=('final_exam_score', 'count')).reset_index()
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# 1) Bar chart (position + length)
bars = axes[0].barh(subj['subject'], subj['avg_score'], color=sns.color_palette("Set2", len(subj)),
                    edgecolor='black', height=0.6)
for b, v in zip(bars, subj['avg_score']):
    axes[0].text(v + 0.5, b.get_y() + b.get_height() / 2, f'{v:.1f}', va='center', fontweight='bold')
axes[0].set_title('Encoding 1: Bar Chart\n(Position + Length)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Average Final Exam Score'); axes[0].set_ylabel('Subject'); axes[0].set_xlim(0, 100)

# 2) Bubble chart (position + size)
axes[1].scatter(subj['avg_score'], subj['avg_attendance'], s=subj['count'] * 3,
                c=range(len(subj)), cmap='Set2', alpha=0.7, edgecolors='black')
for _, r in subj.iterrows():
    axes[1].annotate(r['subject'], (r['avg_score'], r['avg_attendance']), fontsize=8,
                     ha='center', xytext=(0, 10), textcoords='offset points')
axes[1].set_title('Encoding 2: Bubble Chart\n(Position + Size)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Average Final Exam Score'); axes[1].set_ylabel('Average Attendance %')

# 3) Redesigned: grouped bar chart
x, w = np.arange(len(subj)), 0.35
b1 = axes[2].bar(x - w / 2, subj['avg_score'], w, label='Avg Final Score', color='#42A5F5', edgecolor='black')
b2 = axes[2].bar(x + w / 2, subj['avg_attendance'], w, label='Avg Attendance %', color='#66BB6A', edgecolor='black')
for bar in list(b1) + list(b2):
    axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')
axes[2].set_title('Redesigned: Grouped Bar Chart', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Subject'); axes[2].set_ylabel('Average Value')
axes[2].set_xticks(x); axes[2].set_xticklabels(subj['subject'], rotation=30, ha='right', fontsize=9)
axes[2].set_ylim(0, 100); axes[2].legend()

plt.tight_layout()
plt.savefig('exp4_compare_encodings.png', dpi=150, bbox_inches='tight')
plt.show()

print(subj.round(2).to_string(index=False))
print("""
Bar chart   : length is judged accurately -> best for comparing one metric.
Bubble chart: more variables, but area is judged poorly and labels overlap.
Grouped bar : keeps accurate length encoding for two variables.
""")
