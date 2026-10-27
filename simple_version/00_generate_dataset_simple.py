# Simple dataset generator -> cse_student_performance_1200.csv (synthetic data)
import numpy as np, pandas as pd
np.random.seed(42)
N = 200                                    # 200 students x 6 subjects = 1200 rows
subjects = ['Programming','Data Structures','DBMS','Computer Networks','Operating Systems','Artificial Intelligence']
ability = np.random.beta(5, 3, N) * 100    # hidden skill: makes scores realistic & correlated

students = pd.DataFrame({
    'student_id': [f'CSE_{i:04d}' for i in range(1, N + 1)],
    'department': 'CSE',
    'year': np.random.choice([1,2,3,4], N, p=[.28,.28,.24,.20]),
    'section': np.random.choice(['A','B','C'], N),
    'gender': np.random.choice(['Male','Female'], N, p=[.55,.45]),
    'ability': ability})
students['semester'] = students['year'] * 2 - np.random.randint(0, 2, N)
students['previous_semester_gpa'] = np.clip(ability/10*.7 + np.random.normal(0,.8,N), 0, 10).round(2)
students['base_study'] = np.clip(ability/100*6 + np.random.normal(0,1.2,N), .5, 10)
students['base_att'] = np.clip(ability*.7 + 20 + np.random.normal(0,8,N), 15, 100)
students['library_visits_per_month'] = np.clip(ability/100*15 + np.random.normal(0,4,N), 0, 30).astype(int)
students['coding_problems_solved'] = np.clip(ability/100*120 + np.random.normal(0,25,N), 0, 200).astype(int)
students['extracurricular_hours_per_week'] = np.clip(8 - students['base_study']*.5 + np.random.normal(0,2,N), 0, 15).round(1)

df = students.loc[students.index.repeat(6)].reset_index(drop=True)       # 6 rows per student
n = len(df)
df['subject'] = np.tile(subjects, N)
df['assessment_date'] = pd.to_datetime('2024-01-15') + pd.to_timedelta(np.random.randint(0, 330, n), unit='D')
c = lambda v: np.clip(v, 0, 100).round(1)
df['attendance_percentage'] = c(df['base_att'] + np.random.normal(0,5,n))
df['study_hours_per_day'] = np.clip(df['base_study'] + np.random.normal(0,.6,n), .5, 10).round(1)
df['assignment_score'] = c(df['ability']*.5 + df['study_hours_per_day']*4 + np.random.normal(5,8,n))
df['internal_exam_1'] = c(df['ability']*.55 + df['attendance_percentage']*.15 + np.random.normal(0,9,n))
df['internal_exam_2'] = c(df['internal_exam_1']*.6 + df['ability']*.2 + np.random.normal(5,8,n))
df['lab_score'] = c(df['ability']*.4 + df['coding_problems_solved']*.15 + np.random.normal(10,7,n))
df['final_exam_score'] = c(df['ability']*.4 + df['study_hours_per_day']*3 + df['attendance_percentage']*.1
                           + df['previous_semester_gpa']*1.5 + np.random.normal(0,9,n))
df['overall_score'] = (.10*df['attendance_percentage'] + .10*df['assignment_score'] + .15*df['internal_exam_1']
                       + .15*df['internal_exam_2'] + .10*df['lab_score'] + .40*df['final_exam_score']).round(2)
df['result_status'] = np.where((df['overall_score'] >= 40) & (df['attendance_percentage'] >= 50), 'Pass', 'Fail')
df['performance_category'] = pd.cut(df['overall_score'], [-1,40,60,80,101], right=False,
                                    labels=['Poor','Average','Good','Excellent']).astype(str)

cols = ['student_id','department','year','section','gender','semester','subject','assessment_date',
        'attendance_percentage','study_hours_per_day','assignment_score','internal_exam_1','internal_exam_2',
        'lab_score','previous_semester_gpa','library_visits_per_month','coding_problems_solved',
        'extracurricular_hours_per_week','final_exam_score','overall_score','result_status','performance_category']
df[cols].to_csv('cse_student_performance_1200.csv', index=False)
print(df[cols].shape, 'saved. Missing values:', df[cols].isnull().sum().sum())
