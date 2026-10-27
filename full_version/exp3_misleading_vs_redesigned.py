# EXPERIMENT 3: Misleading chart vs redesigned chart
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

df = pd.read_csv('cse_student_performance_1200.csv')
year_avg = df.groupby('year')['final_exam_score'].mean()
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# ---- MISLEADING chart (intentionally bad) ----
ax1 = axes[0]
ax1.bar(['Yr-1!!!', 'Yr-2!!!', 'Yr-3!!!', 'Yr-4!!!'], year_avg.values,
        color=['#FF0000', '#00FF00', '#0000FF', '#FFFF00'], width=0.9)
ax1.set_ylim(year_avg.min() - 2, year_avg.max() + 2)          # truncated axis
ax1.set_title('HUGE Difference in Scores!!!', fontsize=16, fontweight='bold', color='red')
ax1.tick_params(axis='x', rotation=60, labelsize=14)
ax1.tick_params(axis='y', labelsize=8)
ax1.grid(True, linewidth=2, color='black', alpha=0.5)
ax1.annotate('WORST!', xy=(0, year_avg.values[0]), fontsize=18, color='red', fontweight='bold',
             xytext=(0.5, year_avg.values[0] + 1), arrowprops=dict(arrowstyle='->', color='red', lw=3))

# ---- REDESIGNED chart ----
ax2 = axes[1]
good_colors = ['#5C6BC0', '#42A5F5', '#26A69A', '#FFA726']
bars = ax2.bar(['Year 1', 'Year 2', 'Year 3', 'Year 4'], year_avg.values,
               color=good_colors, edgecolor='black', width=0.55)
ax2.set_ylim(0, 100)
ax2.set_title('Average Final Exam Score by Year of Study', fontsize=14, fontweight='bold')
ax2.set_xlabel('Year of Study'); ax2.set_ylabel('Average Final Exam Score')
for bar, val in zip(bars, year_avg.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{val:.1f}',
             ha='center', fontsize=11, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)
ax2.legend(handles=[Patch(facecolor=c, edgecolor='black', label=f'Year {y}')
                    for c, y in zip(good_colors, [1, 2, 3, 4])], loc='upper right')

plt.tight_layout()
plt.savefig('exp3_misleading_vs_redesigned.png', dpi=150, bbox_inches='tight')
plt.show()

print("""
FOUR+ DESIGN ISSUES AND FIXES
1. Truncated y-axis      -> redesigned axis starts at 0 (honest proportions)
2. Clashing neon colours -> accessible, consistent palette
3. Heavy grid + 'WORST!' -> light grid, neutral value labels
4. No axis labels, biased title -> descriptive title and axis labels
5. Cluttered 60-degree tick labels -> clean horizontal 'Year 1' labels
""")
print("Actual year averages:\n", year_avg.round(2).to_string())
