# 🎓 Student Exam Score Predictor

> A machine learning project that predicts students' exam scores using linear regression, trained on 19 behavioral, demographic, and academic features from a dataset of 6,607 students.

---

## 📌 Problem Statement

Predict a student's `Exam_Score` (continuous, 0–100 range) from 19 input features spanning study habits, family background, and school environment.

**Success benchmark:** R² > 0.75

---

## 📂 Dataset

| Property | Value |
|---|---|
| Source | `StudentPerformanceFactors.csv` |
| Size | 6,607 rows × 20 columns |
| Target | `Exam_Score` |
| Features | 19 (6 numerical, 13 categorical) |

### Feature Groups

| Group | Features |
|---|---|
| Study habits / effort | `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity` |
| Motivation / disposition | `Motivation_Level`, `Peer_Influence` |
| Family background | `Parental_Involvement`, `Access_to_Resources`, `Family_Income`, `Parental_Education_Level`, `Distance_from_Home`, `Extracurricular_Activities` |
| School / access | `Teacher_Quality`, `School_Type`, `Internet_Access`, `Learning_Disabilities`, `Gender` |

### Preprocessing Steps
- Missing values in 3 categorical columns imputed with the **mode**
- All 13 categorical columns **ordinal/binary encoded** to integers
- Continuous features **winsorized to IQR whiskers** (outliers kept, not dropped)
- **No feature scaling** — interpretable coefficients were the priority

---

## 🔍 Key EDA Insights

- **Attendance** is the strongest single predictor (r ≈ 0.58)
- **Hours_Studied** is next (r ≈ 0.44), followed by **Tutoring_Sessions** (r ≈ 0.22)
- **No dangerous multicollinearity** — max inter-feature correlation ≈ 0.15
- **Parental_Involvement** and **Motivation_Level** are the strongest categorical predictors
- Relationships are roughly **linear** — polynomial terms added negligible value

---

## 🧪 Models Compared

All models trained on the same 80/20 test split (`random_state=42`).

| Option | Features | Test R² | CV R² (5-fold) |
|---|---|---|---|
| Baseline | `Hours_Studied` only | 0.232 | — |
| Polynomial deg 2 | `Hours_Studied` (quadratic) | 0.233 | — |
| Polynomial deg 3 | `Hours_Studied` (cubic) | 0.233 | — |
| Polynomial deg 4 | `Hours_Studied` (quartic) | 0.233 | — |
| Option 2 | + Attendance, Previous_Scores, Tutoring | 0.640 | — |
| Option 3 | + Sleep_Hours, Physical_Activity | 0.640 | — |
| Option 4 | + Motivation_Level, Parental_Involvement | 0.683 | — |
| Option 5 | Internet_Access, School_Type, Gender | 0.623 | — |
| Option 6 | + Sleep_Hours, Internet_Access | 0.683 | — |
| **Option 7** ✅ | **All 19 features** | **0.769** | **0.726 ± 0.069** |

---

## 🏆 Best Model — Option 7

**Linear Regression on all 19 features**

| Metric | Train | Test |
|---|---|---|
| R² | 0.716 | **0.769** |
| MAE | — | 0.46 |
| RMSE | — | 1.81 |

**Why Option 7 wins:**
- Meets and exceeds the R² > 0.75 benchmark
- Outperforms the best 5-feature model (Option 4/6, R² = 0.683) by **+8.6%**
- Outperforms the single-feature baseline (R² = 0.232) by **+53.7%**
- 5-fold CV R² = **0.726 ± 0.069** confirms robustness across folds

---

## 🗂️ Repo Structure

```
task1/
├── student_performance_analysis.ipynb   # Full EDA, training, and evaluation notebook
├── StudentPerformanceFactors.csv        # Raw dataset (6,607 × 20)
├── app/
│   ├── app.py                           # Standalone Gradio inference app
│   └── requirements.txt                 # Pinned dependencies for the Space
└── models/
    ├── best_model.pkl                   # Serialized Option-7 LinearRegression (joblib)
    └── encode_maps.pkl                  # Encoding map shared by training & inference
```

---

## ⚙️ Run It Locally

```bash
git clone https://github.com/M-Amine-HM/student-score-predictor
cd student-score-predictor/task1

# Install pinned dependencies (scikit-learn pinned to ==1.9.0)
python -m pip install -r app/requirements.txt

# Launch the Gradio app
python app/app.py
```

Then open the local URL printed in your terminal (typically `http://127.0.0.1:7860/`).

> ⚠️ The model was trained with **scikit-learn 1.9.0**, which is pinned in `requirements.txt` to ensure `best_model.pkl` and `encode_maps.pkl` load correctly.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | EDA visualizations |
| `scikit-learn` | Model training and evaluation |
| `joblib` | Model serialization |
| `gradio` | Interactive demo UI |

---

## 👤 Author

- **Author:** Mohamed ElAmine Haj Mohamed
- **GitHub:** [github.com/M-Amine-HM](https://github.com/M-Amine-HM)
- **Portfolio:** [aminehm-portfolio.vercel.app](https://aminehm-portfolio.vercel.app)
- **License:** MIT