# Exp 4: bar vs bubble vs grouped bar
import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv('cse_student_performance_1200.csv')
s = df.groupby('subject').agg(score=('final_exam_score', 'mean'), att=('attendance_percentage', 'mean'),
                              n=('student_id', 'count'))
fig, ax = plt.subplots(1, 3, figsize=(20, 6))

s['score'].plot.barh(ax=ax[0], color='skyblue', edgecolor='black')
ax[0].set(title='Bar Chart (length)', xlabel='Average Final Score')

ax[1].scatter(s['score'], s['att'], s=s['n'] * 2, alpha=0.6, edgecolors='black')
for name, r in s.iterrows(): ax[1].annotate(name, (r['score'], r['att']), fontsize=8, ha='center')
ax[1].set(title='Bubble Chart (position + size)', xlabel='Average Final Score', ylabel='Average Attendance')

s[['score', 'att']].plot.bar(ax=ax[2], edgecolor='black')
ax[2].set(title='Redesign: Grouped Bar', ylabel='Average Value', ylim=(0, 100)); ax[2].tick_params(axis='x', rotation=30)

plt.tight_layout(); plt.savefig('exp4_compare_encodings.png', dpi=150); plt.show()
