# Exp 3: misleading chart vs good chart
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('cse_student_performance_1200.csv')
avg = df.groupby('year')['final_exam_score'].mean()
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# BAD: truncated axis, clashing colours, no labels, biased title
ax[0].bar(avg.index.astype(str), avg.values, color=['red', 'lime', 'blue', 'yellow'])
ax[0].set_ylim(avg.min() - 2, avg.max() + 2)
ax[0].set_title('HUGE Difference in Scores!!!', color='red')

# GOOD: axis from 0, calm colours, labels, neutral title
ax[1].bar(avg.index.astype(str), avg.values, color='steelblue', edgecolor='black')
ax[1].set(ylim=(0, 100), title='Average Final Score by Year', xlabel='Year', ylabel='Average Score')
for i, v in enumerate(avg.values): ax[1].text(i, v + 1, f'{v:.1f}', ha='center')

plt.tight_layout(); plt.savefig('exp3_misleading_vs_redesigned.png', dpi=150); plt.show()
print("Issues: 1) truncated y-axis 2) clashing colours 3) no axis labels 4) biased title")
