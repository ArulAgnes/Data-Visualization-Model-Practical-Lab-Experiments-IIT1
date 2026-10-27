# 📘 Kaggle Notebook — Instructions

**Notebook file:** `data_visualization_complete_lab.ipynb`
**Online URL:** https://www.kaggle.com/code/arulmariaagnes/data-v
**Dataset:** `arulmariaagnes/cse-student-performance-1200` (1,200 × 20, synthetic)

---

## 1. Open the notebook

Go to **https://www.kaggle.com/code/arulmariaagnes/data-v/edit**
(don't have it yet? see §6 — upload the local `.ipynb`).

## 2. Add the CSV as input

1. Look at the **right sidebar** → click **+ Add Data** (or "Add Input")
2. Search for `cse-student-performance-1200` (owner: `arulmariaagnes`)
3. Click it — it appears under **Input data**

The loader cell already knows the Kaggle path:

```
/kaggle/input/cse-student-performance-1200/cse_student_performance_1200.csv
```

> The cell also tries 5 other paths and finally a `kagglehub` download, so it
> finds the file in almost any setup. If it prints **"CSV not found"**, you
> attached the wrong dataset — re-check §2.

## 3. Run all cells

- **Kaggle menu → Run All**, or
- `Ctrl/Cmd + Enter` runs one cell, `Shift + Enter` runs and advances

Expected order:
| Section | What you should see |
|---|---|
| Setup | "Loaded 1200 rows and 20 columns" |
| Q1–Q10 | 23 charts rendered inline + printed interpretation |
| Summary | "23 figures saved … All 10 experiments completed successfully!" |

First run takes **2–4 minutes** (t-SNE/UMAP + `!pip install umap-learn`).

## 4. View outputs

- Charts appear **inline** under each code cell
- PNG files are also saved in `/kaggle/working/` → open the **Output** panel
  (right side) → click a file to preview or download all as a zip

## 5. Save a version

**Save Version → Save & Run All (Commit)** — Kaggle re-runs everything on its
servers and stores a permanent snapshot. Wait for the green "success" badge,
then refresh; the version appears under the **Versions** tab.

## 6. Upload the notebook manually (if it isn't on Kaggle yet)

1. Go to https://www.kaggle.com/code/arulmariaagnes/data-v/edit
2. **File → Import Notebook → Upload** → choose
   `kaggle/data_visualization_complete_lab.ipynb` from this repository
3. **+ Add Data** → attach dataset `cse-student-performance-1200`
4. **Run All**, then **Save Version → Save & Run All (Commit)**
5. Verify all 23 charts render

## 7. Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: umap` | The first cell already runs `!pip install umap-learn --quiet` — run that cell first, or add `!pip install <package>` in a new cell |
| `FileNotFoundError` / "CSV not found" | §2 — attach the dataset; check the printed loader paths |
| `FileNotFoundError` still after attach | Print `os.listdir('/kaggle/input')` in a new cell and use the folder name you see |
| Plots not showing | Make sure you ran the imports cell; `%matplotlib inline` is default on Kaggle |
| UMAP cell slow | Normal on first run (numba compile); later versions are faster |
| Want to restart clean | **Kernel → Restart & Clear Output**, then Run All |

## 8. What NOT to do

- ❌ Don't delete the path-fallback loader cell — it's what makes the notebook portable
- ❌ Don't paste credentials/API keys anywhere in the notebook
- ✅ Do keep the "SYNTHETIC DATA" note — the dataset is fictional by design
