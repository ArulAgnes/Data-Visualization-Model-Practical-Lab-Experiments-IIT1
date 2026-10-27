# ALL 10 EXPERIMENTS - FULL VERSION (run: python all_experiments_full.py)
# Each section is the complete code of one experiment file.

# ================= 00_generate_dataset =================
def run_00_generate_dataset():
    # Generates cse_student_performance_1200.csv (SYNTHETIC data, educational use only)
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    subjects = ['Programming', 'Data Structures', 'DBMS',
                'Computer Networks', 'Operating Systems', 'Artificial Intelligence']
    sections = ['A', 'B', 'C']
    genders = ['Male', 'Female']
    NUM_STUDENTS = 200
    
    student_ids = [f"CSE_{str(i).zfill(4)}" for i in range(1, NUM_STUDENTS + 1)]
    student_years = np.random.choice([1, 2, 3, 4], size=NUM_STUDENTS, p=[0.28, 0.28, 0.24, 0.20])
    student_sections = np.random.choice(sections, size=NUM_STUDENTS)
    student_genders = np.random.choice(genders, size=NUM_STUDENTS, p=[0.55, 0.45])
    student_semesters = np.array([np.random.choice([2 * y - 1, 2 * y]) for y in student_years])
    
    # Hidden "ability" drives realistic correlations
    student_ability = np.random.beta(5, 3, size=NUM_STUDENTS) * 100
    student_prev_gpa = np.clip(student_ability / 10 * 0.7 + np.random.normal(0, 0.8, NUM_STUDENTS), 0, 10).round(2)
    student_base_study = np.clip(student_ability / 100 * 6 + np.random.normal(0, 1.2, NUM_STUDENTS), 0.5, 10).round(1)
    student_base_attendance = np.clip(student_ability * 0.7 + 20 + np.random.normal(0, 8, NUM_STUDENTS), 15, 100).round(1)
    student_library = np.clip((student_ability / 100 * 15 + np.random.normal(0, 4, NUM_STUDENTS)).astype(int), 0, 30)
    student_coding = np.clip((student_ability / 100 * 120 + np.random.normal(0, 25, NUM_STUDENTS)).astype(int), 0, 200)
    student_extra = np.clip(8 - student_base_study * 0.5 + np.random.normal(0, 2, NUM_STUDENTS), 0, 15).round(1)
    
    rows = []
    dates_pool = pd.date_range('2024-01-15', '2024-12-10', periods=30)
    
    for i in range(NUM_STUDENTS):
        subj_order = np.random.permutation(subjects)
        chosen_dates = sorted(np.random.choice(dates_pool, size=6, replace=False))
        for j in range(6):
            ability = student_ability[i]
            noise = np.random.normal(0, 6)
            attendance = np.clip(student_base_attendance[i] + np.random.normal(0, 5), 10, 100)
            study = np.clip(student_base_study[i] + np.random.normal(0, 0.6), 0.5, 10)
            assign = np.clip(ability * 0.5 + study * 4 + np.random.normal(5, 8), 0, 100)
            ie1 = np.clip(ability * 0.55 + attendance * 0.15 + np.random.normal(0, 9), 0, 100)
            ie2 = np.clip(ie1 * 0.6 + ability * 0.2 + np.random.normal(5, 8), 0, 100)
            lab = np.clip(ability * 0.4 + student_coding[i] * 0.15 + np.random.normal(10, 7), 0, 100)
            final = np.clip(ability * 0.4 + study * 3 + attendance * 0.1 +
                            student_prev_gpa[i] * 1.5 + np.random.normal(0, 7) + noise, 0, 100)
    
            attendance, study, assign, ie1, ie2, lab, final = [round(v, 1) for v in
                                                                (attendance, study, assign, ie1, ie2, lab, final)]
            overall = round(0.10 * attendance + 0.10 * assign + 0.15 * ie1 +
                            0.15 * ie2 + 0.10 * lab + 0.40 * final, 2)
    
            result = 'Pass' if (overall >= 40 and attendance >= 50) else 'Fail'
            if overall >= 80: perf = 'Excellent'
            elif overall >= 60: perf = 'Good'
            elif overall >= 40: perf = 'Average'
            else: perf = 'Poor'
    
            rows.append({
                'student_id': student_ids[i], 'department': 'CSE',
                'year': int(student_years[i]), 'section': student_sections[i],
                'gender': student_genders[i], 'semester': int(student_semesters[i]),
                'subject': subj_order[j], 'assessment_date': chosen_dates[j],
                'attendance_percentage': attendance, 'study_hours_per_day': study,
                'assignment_score': assign, 'internal_exam_1': ie1, 'internal_exam_2': ie2,
                'lab_score': lab, 'previous_semester_gpa': student_prev_gpa[i],
                'library_visits_per_month': int(student_library[i]),
                'coding_problems_solved': int(student_coding[i]),
                'extracurricular_hours_per_week': student_extra[i],
                'final_exam_score': final, 'overall_score': overall,
                'result_status': result, 'performance_category': perf})
    
    df = pd.DataFrame(rows)
    assert df.shape[0] == 1200
    df['assessment_date'] = pd.to_datetime(df['assessment_date'])
    
    print("=" * 60); print("DATASET VALIDATION REPORT"); print("=" * 60)
    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())
    print("Unique students:", df['student_id'].nunique())
    print("Date range:", df['assessment_date'].min(), "to", df['assessment_date'].max())
    for col in ['year', 'section', 'gender', 'subject', 'result_status', 'performance_category']:
        print(f"  {col}: {df[col].value_counts().to_dict()}")
    
    df.to_csv('cse_student_performance_1200.csv', index=False)
    print("\nSaved cse_student_performance_1200.csv  |", len(df), "rows,", len(df.columns), "columns")

# ================= exp1_basic_visualizations =================
def run_exp1_basic_visualizations():
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

# ================= exp2_timeseries_visualization =================
def run_exp2_timeseries_visualization():
    # EXPERIMENT 2: Time-series line chart + temporal heatmap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    
    df = pd.read_csv('cse_student_performance_1200.csv')
    df['assessment_date'] = pd.to_datetime(df['assessment_date'])
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1) Line chart with linear trend
    daily = df.groupby('assessment_date')['final_exam_score'].mean().reset_index().sort_values('assessment_date')
    axes[0].plot(daily['assessment_date'], daily['final_exam_score'], marker='o', markersize=5,
                 linewidth=2, color='#1976D2', label='Average Final Exam Score', zorder=3)
    x_num = mdates.date2num(daily['assessment_date'])
    p = np.poly1d(np.polyfit(x_num, daily['final_exam_score'], 1))
    axes[0].plot(daily['assessment_date'], p(x_num), '--', color='red', linewidth=1.5, label='Linear Trend')
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    axes[0].set_title('Average Final Exam Score Over Time (2024)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Assessment Date'); axes[0].set_ylabel('Average Final Exam Score')
    axes[0].legend(loc='upper left')
    
    # 2) Heatmap: semester x month
    df['month'] = df['assessment_date'].dt.month
    pivot = df.pivot_table(values='final_exam_score', index='semester', columns='month', aggfunc='mean')
    pivot.columns = [pd.Timestamp(2024, m, 1).strftime('%b') for m in pivot.columns]
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5, linecolor='white',
                ax=axes[1], cbar_kws={'label': 'Average Final Exam Score'})
    axes[1].set_title('Average Final Exam Score by Semester and Month', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Assessment Month'); axes[1].set_ylabel('Semester')
    
    plt.tight_layout()
    plt.savefig('exp2_timeseries_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Period: {df['assessment_date'].min().date()} to {df['assessment_date'].max().date()}")
    print("Distinct assessment dates:", df['assessment_date'].nunique())
    print("\nMonthly average scores:")
    for m, s in df.groupby('month')['final_exam_score'].mean().items():
        print(f"  {pd.Timestamp(2024, m, 1).strftime('%B'):<10}: {s:.2f}")
    print("\nNote: one year of data -> cannot claim seasonality.")

# ================= exp3_misleading_vs_redesigned =================
def run_exp3_misleading_vs_redesigned():
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

# ================= exp4_compare_encodings =================
def run_exp4_compare_encodings():
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

# ================= exp5_eda =================
def run_exp5_eda():
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

# ================= exp6_relationships =================
def run_exp6_relationships():
    # EXPERIMENT 6: Scatter plots, pair plot, correlation heatmap
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = pd.read_csv('cse_student_performance_1200.csv')
    sns.set_style("whitegrid")
    order = ['Poor', 'Average', 'Good', 'Excellent']
    pal = {'Poor': '#EF5350', 'Average': '#FFA726', 'Good': '#66BB6A', 'Excellent': '#42A5F5'}
    rel_vars = ['attendance_percentage', 'study_hours_per_day', 'assignment_score', 'internal_exam_1',
                'internal_exam_2', 'lab_score', 'final_exam_score', 'overall_score']
    
    # Figure 1: scatter plots
    fig1, ax = plt.subplots(2, 2, figsize=(14, 12))
    sns.scatterplot(x='study_hours_per_day', y='final_exam_score', data=df, hue='performance_category',
                    hue_order=order, palette=pal, alpha=0.5, s=30, ax=ax[0, 0])
    ax[0, 0].set_title('Study Hours vs Final Exam Score', fontweight='bold')
    sns.scatterplot(x='attendance_percentage', y='final_exam_score', data=df, hue='performance_category',
                    hue_order=order, palette=pal, alpha=0.5, s=30, ax=ax[0, 1])
    ax[0, 1].set_title('Attendance vs Final Exam Score', fontweight='bold')
    sns.scatterplot(x='internal_exam_1', y='internal_exam_2', data=df, alpha=0.5, s=30, color='#7E57C2', ax=ax[1, 0])
    ax[1, 0].plot([0, 100], [0, 100], 'r--', label='IE1 = IE2'); ax[1, 0].legend()
    ax[1, 0].set_title('Internal Exam 1 vs Internal Exam 2', fontweight='bold')
    sns.scatterplot(x='assignment_score', y='lab_score', data=df, alpha=0.5, s=30, color='#26A69A', ax=ax[1, 1])
    ax[1, 1].set_title('Assignment Score vs Lab Score', fontweight='bold')
    plt.tight_layout(); plt.savefig('exp6_scatter_plots.png', dpi=150, bbox_inches='tight'); plt.show()
    
    # Figure 2: pair plot
    g = sns.pairplot(df[['study_hours_per_day', 'attendance_percentage', 'assignment_score',
                         'final_exam_score', 'performance_category']],
                     hue='performance_category', hue_order=order, palette=pal, diag_kind='kde',
                     plot_kws={'alpha': 0.4, 's': 20}, height=2.5)
    g.figure.suptitle('Pair Plot of Key Academic Variables', fontweight='bold', y=1.02)
    plt.savefig('exp6_pairplot.png', dpi=150, bbox_inches='tight'); plt.show()
    
    # Figure 3: correlation heatmap
    corr = df[rel_vars].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax3, cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
    ax3.set_title('Correlation Heatmap of Academic Variables', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig('exp6_correlation_heatmap.png', dpi=150, bbox_inches='tight'); plt.show()
    
    print("Strongest correlations (|r| > 0.5):")
    for i in range(len(rel_vars)):
        for j in range(i + 1, len(rel_vars)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                print(f"  {rel_vars[i]} <-> {rel_vars[j]}: r = {r:.3f} "
                      f"({'strong' if abs(r) > 0.7 else 'moderate'} {'positive' if r > 0 else 'negative'})")
    print("\nNOTE: correlation does NOT prove causation.")

# ================= exp7_regression =================
def run_exp7_regression():
    # EXPERIMENT 7: Regression with confidence interval and residual analysis
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    df = pd.read_csv('cse_student_performance_1200.csv')
    sns.set_style("whitegrid")
    x, y = df['study_hours_per_day'].values, df['final_exam_score'].values
    
    slope, intercept, r, p, se = stats.linregress(x, y)
    y_pred = slope * x + intercept
    resid = y - y_pred
    print(f"Equation : Final Score = {slope:.3f} x Study Hours + {intercept:.3f}")
    print(f"r = {r:.4f} | R2 = {r**2:.4f} | p-value = {p:.2e} | slope SE = {se:.4f}")
    print(f"Residuals: mean = {resid.mean():.4f}, std = {resid.std():.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1) Regression + 95% CI
    sns.regplot(x='study_hours_per_day', y='final_exam_score', data=df, ax=axes[0, 0], ci=95,
                scatter_kws={'alpha': 0.3, 's': 20, 'color': '#42A5F5'},
                line_kws={'color': 'red', 'linewidth': 2})
    axes[0, 0].text(0.05, 0.95, f'y = {slope:.2f}x + {intercept:.2f}\nr = {r:.3f}\nR² = {r**2:.3f}',
                    transform=axes[0, 0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[0, 0].set_title('Study Hours vs Final Exam Score\n(red line = fit, shaded = 95% CI)', fontweight='bold')
    axes[0, 0].set_xlabel('Study Hours Per Day'); axes[0, 0].set_ylabel('Final Exam Score')
    
    # 2) Residuals vs fitted
    axes[0, 1].scatter(y_pred, resid, alpha=0.3, s=20, color='#7E57C2')
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_title('Residuals vs Fitted Values', fontweight='bold')
    axes[0, 1].set_xlabel('Fitted Values'); axes[0, 1].set_ylabel('Residuals')
    
    # 3) Residual histogram + normal curve
    axes[1, 0].hist(resid, bins=30, density=True, color='#26A69A', edgecolor='black', alpha=0.8, label='Residuals')
    xs = np.linspace(resid.min(), resid.max(), 100)
    axes[1, 0].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), 'r-', linewidth=2, label='Normal curve')
    axes[1, 0].set_title('Distribution of Residuals', fontweight='bold')
    axes[1, 0].set_xlabel('Residual'); axes[1, 0].set_ylabel('Density'); axes[1, 0].legend()
    
    # 4) Q-Q plot
    stats.probplot(resid, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals', fontweight='bold')
    
    plt.tight_layout(); plt.savefig('exp7_regression_analysis.png', dpi=150, bbox_inches='tight'); plt.show()
    print(f"\nStudy hours explain about {r**2 * 100:.1f}% of the variance in final exam score.")

# ================= exp8_rolling_uncertainty =================
def run_exp8_rolling_uncertainty():
    # EXPERIMENT 8: Rolling average + uncertainty bands
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    
    df = pd.read_csv('cse_student_performance_1200.csv')
    df['assessment_date'] = pd.to_datetime(df['assessment_date'])
    sns.set_style("whitegrid")
    
    daily = df.groupby('assessment_date')['final_exam_score'].agg(['mean', 'std', 'count']).reset_index()
    daily.columns = ['date', 'mean_score', 'std_score', 'count']
    daily = daily.sort_values('date').reset_index(drop=True)
    daily['se'] = daily['std_score'] / np.sqrt(daily['count'])
    daily['ci_lower'] = daily['mean_score'] - 1.96 * daily['se']
    daily['ci_upper'] = daily['mean_score'] + 1.96 * daily['se']
    window = 5
    daily['rolling_mean'] = daily['mean_score'].rolling(window, center=True).mean()
    daily['rolling_std'] = daily['mean_score'].rolling(window, center=True).std()
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax1 = axes[0]
    ax1.plot(daily['date'], daily['mean_score'], marker='o', markersize=4, linewidth=1, alpha=0.5,
             color='#90CAF9', label='Daily Mean Score')
    ax1.plot(daily['date'], daily['rolling_mean'], linewidth=2.5, color='#1565C0', label=f'{window}-Point Rolling Average')
    ax1.fill_between(daily['date'], daily['ci_lower'], daily['ci_upper'], alpha=0.2, color='#42A5F5',
                     label='95% Confidence Interval')
    v = daily['rolling_mean'].notna()
    ax1.fill_between(daily.loc[v, 'date'], daily.loc[v, 'rolling_mean'] - daily.loc[v, 'rolling_std'],
                     daily.loc[v, 'rolling_mean'] + daily.loc[v, 'rolling_std'], alpha=0.15, color='orange',
                     label='Rolling ±1 Std Dev')
    ax1.xaxis.set_major_locator(mdates.MonthLocator()); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_title('Average Final Exam Score Over Time with Rolling Average and Uncertainty', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Assessment Date'); ax1.set_ylabel('Average Final Exam Score'); ax1.legend(loc='lower right')
    
    ax2 = axes[1]
    ax2.bar(daily['date'], daily['std_score'], width=8, color='#EF5350', alpha=0.6, edgecolor='black',
            linewidth=0.3, label='Std Dev of Scores')
    twin = ax2.twinx()
    twin.plot(daily['date'], daily['count'], marker='s', markersize=4, color='#2E7D32', label='Number of Assessments')
    twin.set_ylabel('Count of Assessments', color='#2E7D32')
    ax2.xaxis.set_major_locator(mdates.MonthLocator()); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.set_title('Score Variability and Assessment Volume Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Assessment Date'); ax2.set_ylabel('Standard Deviation of Scores', color='#EF5350')
    l1, n1 = ax2.get_legend_handles_labels(); l2, n2 = twin.get_legend_handles_labels()
    ax2.legend(l1 + l2, n1 + n2, loc='upper right')
    
    plt.tight_layout(); plt.savefig('exp8_rolling_avg_uncertainty.png', dpi=150, bbox_inches='tight'); plt.show()
    
    hi, lo = daily.loc[daily['std_score'].idxmax()], daily.loc[daily['std_score'].idxmin()]
    print("Assessment dates:", len(daily), "| rolling window:", window)
    print(f"Highest variability: {hi['date'].date()}  std={hi['std_score']:.2f}  n={int(hi['count'])}")
    print(f"Lowest  variability: {lo['date'].date()}  std={lo['std_score']:.2f}  n={int(lo['count'])}")
    print("Wider CI = fewer assessments = less reliable mean.")

# ================= exp9_pca =================
def run_exp9_pca():
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

# ================= exp10_tsne_umap =================
def run_exp10_tsne_umap():
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

if __name__ == '__main__':
    run_00_generate_dataset()
    run_exp1_basic_visualizations()
    run_exp2_timeseries_visualization()
    run_exp3_misleading_vs_redesigned()
    run_exp4_compare_encodings()
    run_exp5_eda()
    run_exp6_relationships()
    run_exp7_regression()
    run_exp8_rolling_uncertainty()
    run_exp9_pca()
    run_exp10_tsne_umap()
