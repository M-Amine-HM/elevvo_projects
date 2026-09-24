# ML Internship Portfolio — 10 Tasks, One System

A 10-task machine learning portfolio completed as part of the **Elevvo Machine Learning
Internship**, progressing from foundations to production. Every task ships the same
three deliverables: a **trained model**, a **standalone app you can run yourself**, and
a **written breakdown** of the approach, results, and the honest tradeoffs behind both.
Tasks were not picked at random — each one deliberately adds a capability the previous
ones didn't require (imbalance, temporal data, detection, serving), so the repo reads
as a curriculum rather than a collection.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4.1-E22527?logo=xgboost&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)
![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics%20YOLOv8-8.3.22-00C7F2?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.26-FF5B00?logo=gradio&logoColor=white)

---

## About this internship

This repository contains all 10 tasks completed for the **Elevvo Machine Learning
Internship**, spanning Foundations, Applied, and Industry-level problem sets as
defined by the internship's task brief. Each task was built independently.


## The ten tasks at a glance

Every number below is taken from the task's own README / notebook output. Where a
result was corrected during an audit (data leakage, improper splitting, tautological
metrics), the corrected, verified number is shown — see
[§7 — Honest engineering notes](#7--honest-engineering-notes).

| # | Task | Level | Techniques | Key Result | Folder |
|---|---|---|---|---|---|
| 1 | Student Score Prediction | Foundations | Linear regression, EDA, feature selection via 5-fold CV | **R² = 0.769** on held-out test (target > 0.75), CV R² 0.726 ± 0.069 | [task1-Student Score-Prediction](task1-Student%20Score-Prediction/) |
| 2 | Customer Segmentation | Foundations | K-Means, DBSCAN, silhouette / Davies–Bouldin | **Silhouette 0.5547** at k=5 (chosen programmatically); K-Means vs DBSCAN essentially tied | [task2-customer-segmentation](task2-customer-segmentation/) |
| 3 | Forest Cover Type Classification | Applied | Logistic Regression, Random Forest, XGBoost, class balancing, RandomizedSearchCV | **Macro F1 0.9272, Cohen's Kappa 0.9261** (95.4% acc, 7-class, 103.1× imbalance) | [task3-Forest-Cover-Type-Classification](task3-Forest-Cover-Type-Classification/) |
| 4 | Loan Approval Prediction | Applied | Decision Tree vs Logistic Regression, 5-fold CV selection, SMOTE | **Weighted F1 0.9953**, minority-class recall **0.9876** | [task4-loan-approval-prediction](task4-loan-approval-prediction/) |
| 5 | Movie Recommendation System | Applied | User-based / item-based CF, SVD (k=30), Precision@K | **SVD Precision@5 0.2963**, wins 5/5 random splits; all methods beat the 0.166 popularity baseline | [task5-movie-recommendation-system](task5-movie-recommendation-system/) |
| 6 | Music Genre Classification | Applied | Tabular XGBoost, CNN from scratch, MobileNetV2 transfer learning, track-grouped splits | **Test macro F1 0.7016** (XGBoost); MobileNetV2 0.6163 — all beat the 0.10 random floor | [task6-music-genre-classification](task6-music-genre-classification/) |
| 7 | Sales Forecasting | Applied | XGBoost, lag/rolling features, TimeSeriesSplit, chronological holdout | **MAE 1,360.85** (19.0% below the seasonal-persistence floor), RMSE 2,860.95 | [task7-sales-forecasting](task7-sales-forecasting/) |
| 8 | Traffic Sign Recognition (GTSDB) | Applied | YOLOv8n/s, 43→4 superclass remap, ONNX export, measured FPS | **mAP@0.5 0.1605** (≈3× the 43-class baseline) at **8.5 FPS** on laptop CPU | [task8-traffic-sign-recognition](task8-traffic-sign-recognition/) |
| 9 | Industrial Predictive Maintenance | Industry | Class-weighted Random Forest, validation-tuned decision threshold | Test **FDR 0.1887 / Precision 0.8113 / Recall 0.6143**, with explicit per-class detail | [task9-Industrial-Predictive-Maintenance](task9-Industrial-Predictive-Maintenance/) |
| 10 | MLOps Pipeline | Industry | FastAPI, Docker Compose, Gradio-as-HTTP-client, Pydantic validation, pytest + GitHub Actions CI | Serves Task 9 with **re-verified metrics**; deterministic 400 contract across 5 validation cases | [task10-mlops-pipeline](task10-mlops-pipeline/) |

---

## Why the tasks escalate the way they do

This is not a shuffled list — it's a deliberate ladder where each rung forces a new
kind of thinking:

1. **Foundations (1–2).** Supervised regression, then unsupervised clustering. Task 1
   establishes evaluation discipline (CV-based feature selection, honest residual
   analysis); Task 2 adds the first metric-judgment call — two clustering algorithms
   that turn out to be *tied*, and the decision to ship the more business-ready one.
2. **Classification under imbalance (3–4).** Task 3 faces 7 classes with 103.1×
   imbalance and learns that accuracy lies — hence Macro F1 and Cohen's Kappa. Task 4
   pushes the same principle into a cost-relevant binary setting with per-class
   P/R/F1 and a SMOTE comparison that found SMOTE *didn't help*.
3. **Structured + unstructured range (5–7).** Recommendation systems (sparse 98%
   matrices, Precision@K), audio/multimodal with transfer learning and group-aware
   data hygiene, and time-series with causal feature engineering and a chronological
   (not random) split.
4. **Real-time perception (8).** Object detection where accuracy is only half the
   contract — the other half is FPS, measured, not assumed.
5. **Industry (9–10).** Cost-sensitive predictive maintenance judged by False
   Discovery Rate with an honest per-class breakdown, then the same model lifted out
   of the notebook and shipped as a production-shaped FastAPI + Docker service that
   re-verifies the original numbers end to end.

The range is intentional: by task 10 the repo has covered structured and unstructured
data, balanced and heavily imbalanced targets, offline and real-time constraints, and
single-script apps through containerized services.

---

## How every task is built

Nine of the ten tasks follow the same skeleton, so once you've run one you can run any
of them:

```
task-N/
├── <notebook>.ipynb          # Full EDA → training → evaluation → artifact export
├── app.py                    # Standalone interface (loads ./models/, zero notebook dependency)
├── requirements.txt          # Pinned dependencies (versions match the trained artifacts)
├── models/                   # Committed model artifact(s) + scaler/encoder/params
├── assets/ (or screenshots/) # Result figures and UI screenshots
└── README.md                  # Problem, approach, verified results, how to run
```

Each task is **independently runnable** — `app.py` loads the committed artifact from
`models/`, so you never need to re-run a notebook to use a model. The notebooks are
the *evidence* (they reproduce every reported number) and `app.py` maps each
notebook's final interactive demo into a standalone script.

**Task 10 is architecturally different on purpose.** It has no modeling notebook. It
takes Task 9's shipped artifact and wraps it in a FastAPI service (`main.py` loads
the model once at import time), with a Gradio frontend that is a *thin HTTP client*
of the API (no model loaded client-side), orchestrated by Docker Compose with
healthcheck-gated startup. It also carries the only automated test suite in the repo
(`tests/test_api.py`, run in CI).

---

## Repo structure

```
elevvo_projects/
├── README.md
├── task1-Student Score-Prediction/
├── task2-customer-segmentation/
├── task3-Forest-Cover-Type-Classification/
├── task4-loan-approval-prediction/
├── task5-movie-recommendation-system/
├── task6-music-genre-classification/
├── task7-sales-forecasting/
├── task8-traffic-sign-recognition/
├── task9-Industrial-Predictive-Maintenance/
└── task10-mlops-pipeline/
```

---

## How to run any task

**Tasks 1–9** (all modeling tasks):

```bash
cd <task-folder>
pip install -r requirements.txt
python app.py
```

Then open the printed local URL (typically `http://127.0.0.1:7860`). The same
commands also work from Jupyter for the notebooks.

**Task 10** (served version):

```bash
cd task10-mlops-pipeline
docker-compose up --build
```

- API + Swagger docs: `http://localhost:8000/docs`
- Gradio frontend: `http://localhost:7860`

Or, without Docker, `pip install -r requirements.txt && pytest tests/` runs the API
test suite directly against the FastAPI TestClient.

> **Pin-version warning.** Task artifacts are joblib/pickle files trained with
> specific library versions. Each `requirements.txt` pins those versions (e.g.
> scikit-learn, XGBoost) — do not upgrade them or the `.pkl`/`.joblib` artifacts may
> not load.

---

## Honest engineering notes

A few decisions and limitations worth knowing up front — they're real, and they're
deliberate.

**Small-sample limitations are stated where they exist.** Task 9's two rarest failure
modes (Random Failure, test support 4; Tool Wear Failure, test support 9) have
near-zero recall at the deployed threshold. That is a data-volume fact, not a tuning
mistake, which is exactly why its README reports the per-class table alongside the
aggregate — quoting FDR alone without that detail would overstate the model. Task 8's
GTSDB dataset has ~14 signs/class, a fraction of a production detection set, so its
absolute mAP (0.1605) is low by design; the value of that task is the pipeline and
the honest reporting, and the README says so rather than dressing it up.

**Metric choices were reasoned, not defaulted.** Macro F1 (over accuracy) is used
wherever classes are imbalanced — Task 3's 103.1× multi-class split and Task 6's
audio task both make a single accuracy number misleading. Task 7 uses a
*chronological* holdout (cutoff 2012-07-01) instead of a random split because
training on the future to predict the past would be vacuous, and reports improvement
against naive seasonal persistence rather than against another model. Task 4 notes
explicitly that its weighted F1 (0.9953) sits numerically close to accuracy because
62/38 is only mildly imbalanced — the per-class row is the real signal.

**Audit fixes are the source of several headline numbers.** Task 3's scaler
originally fit on the full dataset before splitting; after fixing the leak,
XGBoost's balanced re-run scored *lower* than the original unweighted run, and the
README reports the corrected comparison rather than the flattering one. Task 6
originally validated on row-level splits where 3-second slices of the same track
straddled train and validation, inflating validation accuracy to 0.9043; switching to
track-grouped splits dropped it to a far more honest 0.7416. Task 7's
cross-validation once double-counted the holdout and was corrected to score strictly
on pre-cutoff data. Task 9 *retracted* a RUL bonus because the metric was a tautology
(250 − tool wear is a linear function of an input). Wherever a number changed, the
changed one is the one in this README.

**Model size and reproducibility shaped decisions.** Task 3's tuned Random Forest
(200 trees) was trimmed to 40 trees (81.5 MB) so the artifact fits under GitHub's
100 MB file limit, at a cost of just 0.0044 Macro F1 (0.9272 → 0.9228). Every task
pins library versions in `requirements.txt` so committed artifacts load against the
versions they were trained with; Task 10 pins scikit-learn 1.9.0 in its Docker image
to make unpickling exact and warning-free, and re-verifies the wrapped model's
numbers before serving them.

---

## Portfolio & links

- **Internship:** Elevvo Machine Learning Internship
- **Portfolio site:** [Portfolio](https://aminehm-portfolio.vercel.app/)
- **About the author:** Mohamed ElAmine Haj Mohamed — [GitHub](https://github.com/M-Amine-HM)
- **Live interfaces:** every task's demo runs locally via its `app.py` (Tasks 1–9) or
  via `docker-compose up --build` for the served Task 10 version. Each `README.md`
  links screenshots of the actual interface.