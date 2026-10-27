"""
scripts/generate_dataset.py
Generate the SYNTHETIC CSE student performance dataset (1200 rows x 20 columns).

Scenario: Velammal Engineering College, Dept of CSE - Academic Performance
Monitoring System, Even Semester 2024-2025. 1200 students tracked across
attendance, study habits, lab work, assignments, internals, coding practice,
library usage and final exams.

THIS IS 100% SYNTHETIC / FICTIONAL DATA created for a college lab exercise.
No real student data is used anywhere.

Run:  python scripts/generate_dataset.py
Out:  datasets/cse_student_performance_1200.csv
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 0. SETTINGS
# ---------------------------------------------------------------------------
SEED = 42                      # fixed seed => same dataset every run
N = 1200                       # number of students
np.random.seed(SEED)           # makes ALL random numbers reproducible

# repo root = parent of the scripts/ folder that contains this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "datasets")
OUT_PATH = os.path.join(OUT_DIR, "cse_student_performance_1200.csv")


def zscore(values):
    """Standardize an array so it has mean 0 and standard deviation 1.
    We need z-scores so every feature can be mixed on the same scale."""
    return (values - values.mean()) / values.std()


def grade_from_marks(marks):
    """Convert a final exam mark (0-100) into the letter grade."""
    if marks >= 90:
        return "O"
    if marks >= 80:
        return "A+"
    if marks >= 70:
        return "A"
    if marks >= 60:
        return "B+"
    if marks >= 50:
        return "B"
    if marks >= 40:
        return "C"
    return "F"


# ---------------------------------------------------------------------------
# 1. IDENTIFIERS: student_id, student_name, department, semester, section, gender
# ---------------------------------------------------------------------------

# 1a. student_id -> CSE2024_0001 ... CSE2024_1200 (all unique by construction)
student_id = [f"CSE2024_{i:04d}" for i in range(1, N + 1)]

# 1b. gender -> exact counts: Male 55%, Female 43%, Other 2%
gender = np.array(["Male"] * 660 + ["Female"] * 516 + ["Other"] * 24)
np.random.shuffle(gender)                      # mix the order randomly

# 1c. fictional name pools (generic Indian first/last names, NO real persons)
MALE_FIRST = [
    "Arun", "Karthik", "Rahul", "Vijay", "Suresh", "Rajesh", "Prakash",
    "Sathish", "Manoj", "Bharath", "Saran", "Naveen", "Vignesh", "Hari",
    "Ashwin", "Kiran", "Deepak", "Ramesh", "Senthil", "Anand", "Balaji",
    "Murugan", "Jegan", "Prabhu", "Siva", "Vetri", "Yogesh", "Dinesh",
    "Gokul", "Ajith", "Vishnu", "Sriram", "Hariharan", "Muthu", "Chandru",
    "Raghav", "Siddharth", "Aravind", "Kavin", "Pranav", "Rohit", "Sanjay",
    "Thiru", "Vishal", "Kuralarasan",
]
FEMALE_FIRST = [
    "Priya", "Deepa", "Divya", "Kavitha", "Lakshmi", "Meena", "Nithya",
    "Ramya", "Sneha", "Swetha", "Bhavani", "Varsha", "Anitha", "Gayathri",
    "Harini", "Keerthana", "Malathi", "Nandhini", "Oviya", "Poornima",
    "Rekha", "Sangeetha", "Thamarai", "Uthra", "Vidya", "Yazhini",
    "Aishwarya", "Brindha", "Chitra", "Dharani", "Easwari", "Fathima",
    "Gomathi", "Indira", "Janani", "Kanimozhi", "Lavanya", "Maha", "Nila",
    "Pavithra", "Reshma", "Sindhu", "Tamilselvi", "Uma", "Vennila",
]
LAST_NAMES = [
    "Kumar", "Sharma", "Rajan", "Krishnan", "Iyer", "Nair", "Reddy",
    "Pillai", "Gounder", "Murugan", "Selvam", "Balaji", "Subramanian",
    "Venkatesh", "Chandran", "Ramanathan", "Srinivasan", "Mani", "Palani",
    "Eswaran", "Thangam", "Kumaresan", "Anand", "Joshi", "Verma", "Das",
    "Bose", "Ghosh", "Mehta", "Shah", "Rao", "Naik", "Hegde", "Patil",
    "Kulkarni", "Deshmukh", "Bhat", "Gowda", "Prasad", "Sundaram",
    "Anbu", "Chelvan", "Durai", "Elango", "Fernandes", "Gunasekar",
    "Hariharan", "Indran", "Jeyaram", "Kannan", "Logesh", "Manikandan",
    "Nithyanand", "Parthiban", "Ravichandran", "Sakthivel", "Thirumalai",
    "Umesh", "Vairam", "Zaki",
]

# build 1200 UNIQUE names (loop until we find an unused combination)
student_name = []
used_names = set()
for g in gender:
    pool = MALE_FIRST if g == "Male" else FEMALE_FIRST
    if g == "Other":
        pool = MALE_FIRST + FEMALE_FIRST
    while True:
        full = f"{np.random.choice(pool)} {np.random.choice(LAST_NAMES)}"
        if full not in used_names:
            used_names.add(full)
            student_name.append(full)
            break

# 1d. department -> exact counts CSE 900 / CSE-AI 200 / CSE-DS 100
department = np.array(["CSE"] * 900 + ["CSE-AI"] * 200 + ["CSE-DS"] * 100)
np.random.shuffle(department)

# 1e. semester -> exact counts 2:400, 4:350, 6:300, 8:150 (attrition pattern)
semester = np.array([2] * 400 + [4] * 350 + [6] * 300 + [8] * 150)
np.random.shuffle(semester)

# 1f. section -> A/B/C/D spread roughly evenly inside every semester
section = np.random.choice(["A", "B", "C", "D"], size=N)

# ---------------------------------------------------------------------------
# 2. LATENT "ABILITY" - one hidden number that links related features
#    Students who study more also attend more, score better etc.
#    Beta(5,3) is slightly right-skewed: most students are average-to-good.
# ---------------------------------------------------------------------------
ability = np.random.beta(5, 3, size=N)
az = zscore(ability)                       # standardized ability (mean 0, std 1)

# ---------------------------------------------------------------------------
# 3. NUMERICAL FEATURES (each clipped to its allowed range)
# ---------------------------------------------------------------------------

# attendance: Normal(78, 12) clipped to [40, 100]
attendance = np.clip(78 + 9 * az + np.random.normal(0, 8, N), 40, 100).round(1)

# study hours: log-normal => right-skewed (most study 8-15 h, few study 35+)
study_hours = np.clip(
    np.exp(2.35 + 0.35 * az + np.random.normal(0, 0.5, N)), 2, 45
).round(1)

# coding practice hours: also right-skewed, range [0, 30]
coding_hours = np.clip(
    np.exp(1.6 + 0.6 * az + np.random.normal(0, 0.65, N)), 0, 30
).round(1)

# lab marks: left-skewed Beta(5,2)*50 -> most score 30-45, few below 15
#            boosted by coding practice (their correlation target is 0.5-0.65)
lab_base = 50 * np.random.beta(5, 2, size=N)
lab_marks = np.clip(
    lab_base + 1.1 * (coding_hours - coding_hours.mean()) + 3 * az
    + np.random.normal(0, 2.5, N), 0, 50
).round(1)

# assignment marks: left-skewed, range [0, 30], most score 20-28
assignment_marks = np.clip(
    30 * np.random.beta(6, 1.8, size=N) + 1.5 * az + np.random.normal(0, 1.5, N),
    0, 30,
).round(1)

# internal marks: Normal(28, 7) clipped to [0, 40]
internal_marks = np.clip(28 + 4 * az + np.random.normal(0, 6, N), 0, 40).round(1)

# previous semester CGPA: Normal(7.5, 1.2) clipped to [4, 10]
prev_cgpa = np.clip(7.5 + 0.5 * az + np.random.normal(0, 1.1, N), 4.0, 10.0).round(2)

# online assessment: Normal(62, 18) clipped to [0, 100]
online_score = np.clip(62 + 6 * az + np.random.normal(0, 17, N), 0, 100).round(1)

# library visits: Poisson(lambda) with lambda depending on study effort
#                => weak positive correlation with study hours (target 0.2-0.35)
z_study = zscore(study_hours)
lib_lambda = np.clip(6 + 0.7 * az + 0.6 * z_study, 2, 14)
library_visits = np.clip(np.random.poisson(lib_lambda), 0, 25)

# support sessions: Poisson(lambda) with HIGHER lambda for LOW ability students
#                  => struggling students ask for help (negative r with marks)
sup_lambda = np.clip(3 - 1.4 * az, 0.5, 9)
support_sessions = np.clip(np.random.poisson(sup_lambda), 0, 10)

# ---------------------------------------------------------------------------
# 4. ASSESSMENT DATES - weekly Monday assessments over one semester (+4 extra)
#    2025-01-06 ... 2025-04-28 (17 Mondays + 4 extra days = 20 dates). The
#    forced dates guarantee every month has >=5 distinct dates (check #23).
# ---------------------------------------------------------------------------
mondays = pd.date_range("2025-01-06", "2025-04-28", freq="7D").values  # 16 Mondays
extra_days = pd.to_datetime(
    ["2025-01-31", "2025-02-28", "2025-04-04", "2025-04-11"]
).values
date_pool = np.concatenate([mondays, extra_days])          # 17 Mondays + 4 extra
# Mondays are common (90% of students), extra days are rare (10%)
date_p = np.array([0.9 / len(mondays)] * len(mondays)
                  + [0.1 / len(extra_days)] * len(extra_days))
assessment_date = np.random.choice(date_pool, size=N, p=date_p)
assessment_date[:len(date_pool)] = date_pool               # force every date used
np.random.shuffle(assessment_date)

# ---------------------------------------------------------------------------
# 5. FINAL EXAM MARKS - the primary dependent variable
#    final = 58 + 15 * (w1*z_attendance + w2*z_study + ... + noise)
#    The z-scores put every feature on one scale; the weights decide how much
#    each feature influences the exam result. We auto-tune the weights so the
#    correlations match the targets of the lab specification.
# ---------------------------------------------------------------------------
FEATURES = [
    "attendance", "study", "lab", "assignment",
    "internal", "cgpa", "online", "coding",
]
RAW = {
    "attendance": attendance,
    "study": study_hours,
    "lab": lab_marks,
    "assignment": assignment_marks,
    "internal": internal_marks,
    "cgpa": prev_cgpa,
    "online": online_score,
    "coding": coding_hours,
}
# target correlation of each feature with the final marks
TARGET_R = {
    "attendance": 0.53,   # spec target 0.45 - 0.60 (0.53 -> ~0.49 after outliers)
    "study": 0.60,        # spec target 0.55 - 0.70
    "lab": 0.22,
    "assignment": 0.18,
    "internal": 0.20,
    "cgpa": 0.44,         # spec target 0.40 - 0.55
    "online": 0.10,
    "coding": 0.12,
}
Z = {name: zscore(RAW[name]) for name in FEATURES}
Z_NOISE = zscore(np.random.normal(0, 1, N))   # pure random exam-day luck
NOISE_W = 0.30

# auto-calibration: adjust each weight until the real correlation hits target
weights = dict(TARGET_R)
for _ in range(80):
    signal = NOISE_W * Z_NOISE
    for name in FEATURES:
        signal = signal + weights[name] * Z[name]
    final = 58 + 15 * signal / signal.std()      # mean 58, std 15, then clip
    for name in FEATURES:
        actual_r = np.corrcoef(RAW[name], final)[0, 1]
        if actual_r != 0:
            # damped update (power 0.7) keeps the iteration stable
            weights[name] *= (TARGET_R[name] / actual_r) ** 0.7
final_exam_marks = np.clip(final, 0, 100).round(1)

# ---------------------------------------------------------------------------
# 6. OUTLIER INJECTION - 24 deliberate "interesting" students (2% of data)
#    These make EDA and scatter plots realistic and discussion-worthy.
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
chosen = rng.permutation(N)          # one shuffled index list -> disjoint groups
pos = 0


def take(k):
    """Take the next k unused student indexes."""
    global pos
    idx = chosen[pos:pos + k]
    pos += k
    return idx


# 6a. 6 students: very high attendance (>95) but low marks (<35) - present, not engaged
idx = take(6)
attendance[idx] = rng.uniform(96, 99.5, 6).round(1)
final_exam_marks[idx] = rng.uniform(26, 34, 6).round(1)

# 6b. 6 students: low attendance (<55) but high marks (>75) - self-learners
idx = take(6)
attendance[idx] = rng.uniform(46, 54, 6).round(1)
final_exam_marks[idx] = rng.uniform(76, 90, 6).round(1)

# 6c. 4 students: extremely high study hours (>35) but mediocre marks (45-55)
idx = take(4)
study_hours[idx] = rng.uniform(36, 44, 4).round(1)
final_exam_marks[idx] = rng.uniform(45, 55, 4).round(1)

# 6d. 4 students: very high coding hours (>25) but low lab marks (<15)
idx = take(4)
coding_hours[idx] = rng.uniform(26, 30, 4).round(1)
lab_marks[idx] = rng.uniform(5, 14, 4).round(1)

# 6e. 4 students: many support sessions (>8) AND high marks (>80) - proactive
idx = take(4)
support_sessions[idx] = rng.integers(9, 11, 4)
final_exam_marks[idx] = rng.uniform(81, 92, 4).round(1)

# ---------------------------------------------------------------------------
# 7. DERIVED COLUMNS - final_grade and pass_status (100% consistent)
# ---------------------------------------------------------------------------
final_grade = [grade_from_marks(m) for m in final_exam_marks]
pass_status = ["Fail" if g == "F" else "Pass" for g in final_grade]

# ---------------------------------------------------------------------------
# 8. ASSEMBLE THE TABLE (column order exactly as specified)
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "student_id": student_id,
    "student_name": student_name,
    "department": department,
    "semester": semester,
    "section": section,
    "gender": gender,
    "attendance_percentage": attendance,
    "study_hours_per_week": study_hours,
    "lab_marks": lab_marks,
    "assignment_marks": assignment_marks,
    "internal_marks": internal_marks,
    "previous_semester_cgpa": prev_cgpa,
    "online_assessment_score": online_score,
    "library_visits_per_month": library_visits,
    "coding_practice_hours_per_week": coding_hours,
    "support_sessions_attended": support_sessions,
    "assessment_date": pd.to_datetime(assessment_date).strftime("%Y-%m-%d"),
    "final_exam_marks": final_exam_marks,
    "final_grade": final_grade,
    "pass_status": pass_status,
})

# ---------------------------------------------------------------------------
# 9. VALIDATION - 25 automated checks
# ---------------------------------------------------------------------------
print("=" * 70)
print("DATASET VALIDATION REPORT (25 checks)")
print("=" * 70)
results = []


def check(label, condition, detail=""):
    """Run one validation check and remember PASS/FAIL."""
    status = "PASS" if condition else "FAIL"
    results.append((label, status, detail))
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))


check("1. Exactly 1200 rows", len(df) == 1200, f"rows={len(df)}")
check("2. Exactly 20 columns", len(df.columns) == 20, f"cols={len(df.columns)}")
check("3. Zero duplicate rows", int(df.duplicated().sum()) == 0)
check("4. Zero duplicate student_ids", int(df["student_id"].duplicated().sum()) == 0)
check("5. Zero missing values", int(df.isnull().sum().sum()) == 0)
check("6. attendance in [40,100]", df["attendance_percentage"].between(40, 100).all())
check("7. study_hours in [2,45]", df["study_hours_per_week"].between(2, 45).all())
check("8. lab_marks in [0,50]", df["lab_marks"].between(0, 50).all())
check("9. assignment_marks in [0,30]", df["assignment_marks"].between(0, 30).all())
check("10. internal_marks in [0,40]", df["internal_marks"].between(0, 40).all())
check("11. prev_cgpa in [4,10]", df["previous_semester_cgpa"].between(4, 10).all())
check("12. online_score in [0,100]", df["online_assessment_score"].between(0, 100).all())
check("13. library_visits in [0,25]", df["library_visits_per_month"].between(0, 25).all())
check("14. coding_hours in [0,30]", df["coding_practice_hours_per_week"].between(0, 30).all())
check("15. support_sessions in [0,10]", df["support_sessions_attended"].between(0, 10).all())

dates = pd.to_datetime(df["assessment_date"])
dates_ok = ((dates >= "2025-01-06") & (dates <= "2025-04-28")).all()
check("16. dates within 2025-01-06..2025-04-28",
      bool(dates_ok),
      f"min={dates.min().date()} max={dates.max().date()}")
check("17. final_marks in [0,100]", df["final_exam_marks"].between(0, 100).all())
check("18. grades in allowed set",
      set(df["final_grade"]) <= {"O", "A+", "A", "B+", "B", "C", "F"},
      f"found={sorted(set(df['final_grade']))}")
check("19. pass_status in {Pass,Fail}",
      set(df["pass_status"]) <= {"Pass", "Fail"})

recomputed = df["final_exam_marks"].apply(grade_from_marks)
check("20. grade matches marks on every row", bool((recomputed == df["final_grade"]).all()))
expected_pass = np.where(df["final_grade"] == "F", "Fail", "Pass")
check("21. pass_status matches final_grade",
      bool((expected_pass == df["pass_status"]).all()))

r_study_final = df["study_hours_per_week"].corr(df["final_exam_marks"])
check("22. corr(study, final) inside [0.4, 0.8]",
      0.4 <= r_study_final <= 0.8, f"r={r_study_final:.3f}")

distinct_per_month = dates.groupby(dates.dt.to_period("M")).nunique()
check("23. >=5 distinct assessment dates per month",
      bool((distinct_per_month >= 5).all()),
      ", ".join(f"{m}:{c}" for m, c in distinct_per_month.items()))

pass_rate = (df["pass_status"] == "Pass").mean()
check("24. reasonable grade distribution (pass rate 60-98%)",
      0.60 <= pass_rate <= 0.98, f"pass_rate={pass_rate:.1%}")

print("-" * 70)
n_fail = sum(1 for _, s, _ in results if s == "FAIL")
print(f"  [PASS] 25. Complete validation report printed (failures: {n_fail})")
print("-" * 70)

# correlation structure report (soft targets from spec section 1.3)
print("\nCORRELATION STRUCTURE (actual vs target):")
pairs = [
    ("study_hours_per_week", "final_exam_marks", 0.55, 0.70),
    ("attendance_percentage", "final_exam_marks", 0.45, 0.60),
    ("coding_practice_hours_per_week", "lab_marks", 0.50, 0.65),
    ("previous_semester_cgpa", "final_exam_marks", 0.40, 0.55),
    ("library_visits_per_month", "study_hours_per_week", 0.20, 0.35),
    ("support_sessions_attended", "final_exam_marks", -0.30, -0.15),
]
for a, b, lo, hi in pairs:
    r = df[a].corr(df[b])
    mark = "OK" if lo <= r <= hi else "OUT OF RANGE"
    print(f"  {a:34s} <-> {b:20s} r={r:+.3f}  target [{lo:+.2f},{hi:+.2f}]  {mark}")

# ---------------------------------------------------------------------------
# 10. SAVE + SUMMARY
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes.to_string())
print("\nFirst 10 rows:\n", df.head(10).to_string(index=False))
print("\nDescribe():\n", df.describe().round(2).to_string())
for col in ["department", "semester", "section", "gender", "final_grade", "pass_status"]:
    print(f"\nvalue_counts({col}):\n{df[col].value_counts().to_string()}")

print("\n" + "=" * 70)
if n_fail == 0:
    print("DATASET GENERATION COMPLETE - ALL CHECKS PASSED")
else:
    print(f"DATASET GENERATION FINISHED WITH {n_fail} FAILED CHECK(S):")
    for label, status, detail in results:
        if status == "FAIL":
            print(f"  FAIL: {label} {detail}")
print("=" * 70)
print(f"Saved -> {OUT_PATH}")
