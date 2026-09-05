# 🎬 Task 5: Movie Recommendation System

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Gradio](https://img.shields.io/badge/Gradio-6.26-FF5B00?logo=gradio&logoColor=white)](https://www.gradio.app/)

A collaborative-filtering pipeline that recommends movies a user will like by comparing their
taste against users who rate movies similarly. Built on the **MovieLens** dataset, the project
compares **user-based CF**, **item-based CF**, and **SVD matrix factorization** on **precision@K** —
with SVD coming out on top (0.296 at K=5). A live Gradio demo lets you enter any user ID and get a
ranked list of unseen, highly-scored movies.

---

## 📌 Project Overview

| | |
|---|---|
| **Problem** | Recommend movies a user hasn't seen, based on similar users' tastes and latent preference patterns |
| **Dataset** | [MovieLens (GroupLens)](https://www.kaggle.com/datasets/abhikjha/movielens-100k) — `ml-latest-small`: 100,836 ratings, 610 users, 9,724 movies, ratings on a 0.5–5.0 half-point scale |
| **Matrix** | 610 users × 2,269 movies (after popularity filter) — 98%+ sparse |
| **Best Method** | SVD matrix factorization — **Precision@5 = 0.2963** |
| **Bonus** | Item-based CF (`item_sim_matrix`) · SVD matrix factorization (`TruncatedSVD`) |

---

## 📁 Project Structure

```
task5-movie-recommendation-system/
├── dataset/                  ← Raw MovieLens files (ratings.csv, movies.csv, links.csv, tags.csv)
├── notebook.ipynb            ← Main notebook (11 sections, fixed internship architecture)
├── app.py                    ← Standalone Gradio app (loads ./models/, zero notebook deps)
├── requirements.txt          ← Pinned dependencies
├── models/
│   ├── user_item_matrix.pkl  ← Training matrix (users × movies, 0 = unrated)
│   ├── user_sim_matrix.pkl   ← User–user cosine similarity (primary method)
│   ├── item_sim_matrix.pkl   ← Movie–movie cosine similarity (bonus 1)
│   ├── svd_model.pkl         ← Fitted TruncatedSVD, k=30 (bonus 2)
│   ├── user_id_map.pkl       ← userId ↔ row index
│   ├── movie_id_map.pkl      ← movieId ↔ column index
│   └── movies_info.pkl       ← movieId → title/genres (display)
├── assets/
│   ├── results.png           ← Precision@K for the primary (user-based) method
│   └── precision_comparison.png  ← Three-method comparison chart
└── README.md                 ← This file
```

---

## Problem

Given a user's rating history, rank **unseen movies** by how likely they are to be liked — using
the implicit signal of *which users rate movies alike* — and score the ranking with **Precision@K**
against held-out high-rated movies.

## Dataset

**MovieLens (ml-latest-small)** — [GroupLens](https://grouplens.org/datasets/movielens/) (same
family as the task's recommended MovieLens 100K).

| Property | Value |
|---|---|
| Ratings | 100,836 (`userId, movieId, rating, timestamp`) |
| Users / Movies | 610 / 9,724 |
| Rating scale | 0.5 – 5.0 in **half-point** increments (this variant) |
| Sparsity | 98.3% of matrix cells empty |
| Most-rated movie | Toy Story (1995) |

Popularity filter: movies rated by **< 10 users** are dropped (2,269 kept) so all three
algorithms stay tractable and the comparison stays fair — near-zero-signal movies would add
memory blow-up for item–item/SVD with no recommender value.

## Approach

- **Preprocessing / matrix construction:** pivot `ratings.csv` → user-item matrix (rows = users,
  cols = movies). Missing cells stay **NaN** for Pearson, and a **0-filled copy** is used for
  cosine (safe: no real 0 ratings exist, so `0` unambiguously means *"no opinion"*).
- **Similarity method:** **cosine similarity** between users as the primary metric (Pearson
  correlation on co-rated items is demonstrated too, but in a 98%-sparse matrix co-rated pairs
  are too rare to be reliable). Top `NEIGHBORS = 50` similar users form the neighborhood.
- **Recommendation logic (user-based):** score each unseen movie as the **similarity-weighted
  *sum*** of neighbors' ratings — a sum rewards *consensus* (many similar users liking a movie),
  whereas a weighted *average* lets one obscure neighbor's 5.0 outrank a movie 50 similar users
  love. This single choice moved Precision@5 from ~0.006 → 0.273 on the same held-out split.
  Rank and return the top-K.

## Results

Held-out evaluation: for each of 600 eligible users, 20% of their **high-rated (≥ 4.0)** movies
are hidden; a recommendation counts as a hit if it appears in that hidden set.

**Primary method — User-based CF (cosine, k=50):**

| Metric | Score |
|---|---|
| Precision@5  | **0.2730** ± 0.2619 |
| Precision@10 | **0.2132** ± 0.2077 |
| Precision@20 | **0.1625** ± 0.1594 |

![results](assets/results.png)

**All three methods** (same split, same Precision@K protocol):

| K | User-based CF | Item-based CF | SVD (k=30) |
|---|---|---|---|
| 5  | 0.2730 | 0.2020 | **0.2963** |
| 10 | 0.2132 | 0.1613 | **0.2303** |
| 20 | 0.1625 | 0.1276 | **0.1692** |

![precision comparison](assets/precision_comparison.png)

> **Takeaway:** SVD wins at every K — it compresses the sparse interaction matrix into 30 latent
> factors that capture global structure (genre blocks, popularity bias) that raw user-item
> similarity cannot see. User-based CF is second and is strongest at smaller K; item-based CF
> trails because the movie–movie similarity signal is even sparser.

## Interface

Interactive **Gradio** demo — enter a **User ID**, pick a method, and set the list size; the app
returns a ranked table of movie titles with scores. It loads the saved artifacts from `./models/`,
so `app.py` behaves identically to the notebook's final section.

![gradio demo](assets/gradio_demo.png)

## Bonus work

- [x] **Item-based collaborative filtering** — movie–movie cosine similarity; scores candidates
  by similarity to the user's top-rated movies. Precision@5 = **0.2020**.
- [x] **Matrix factorization (SVD)** — `TruncatedSVD` (k=30) decomposes the sparse training
  matrix into latent factors; predictions reconstructed via `inverse_transform(transform(X))`.
  Precision@5 = **0.2963** — the winner.

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Then open the printed local URL (default `http://127.0.0.1:7860`).

### Run the notebook
1. Open `notebook.ipynb` in Jupyter / VS Code.
2. Ensure `dataset/ratings.csv` and `dataset/movies.csv` sit next to the notebook (they do).
3. Run all cells (**Kernel → Restart & Run All**). The final section launches the same interface
   with `demo.launch(share=True)` — use the public link for a screenshot.

### Libraries used

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data loading, matrix construction, numerics |
| `scipy` | Sparse matrix input for SVD |
| `scikit-learn` | `cosine_similarity`, `TruncatedSVD` |
| `matplotlib` / `seaborn` | EDA plots and Precision@K charts |
| `joblib` | Artifact persistence (`models/*.pkl`) |
| `gradio` | Live prediction interface |