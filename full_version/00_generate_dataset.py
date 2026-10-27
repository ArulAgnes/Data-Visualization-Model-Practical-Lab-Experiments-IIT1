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
