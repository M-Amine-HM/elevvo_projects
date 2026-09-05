"""Standalone Gradio app for Task 5 — Movie Recommendation System.

Loads the recommender artifacts saved by the training notebook (notebook.ipynb) from
./models/ and exposes the exact same Gradio interface as the notebook's final section.
Fully self-contained — no notebook dependency.

Run standalone locally from the task5 folder with:
    python app.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ---------------------------------------------------------------------------
# Load artifacts (mirrors notebook step 10)
# ---------------------------------------------------------------------------
user_item = joblib.load(os.path.join(MODELS_DIR, "user_item_matrix.pkl"))
user_sim = joblib.load(os.path.join(MODELS_DIR, "user_sim_matrix.pkl"))
item_sim = joblib.load(os.path.join(MODELS_DIR, "item_sim_matrix.pkl"))
svd_model = joblib.load(os.path.join(MODELS_DIR, "svd_model.pkl"))
user_map = joblib.load(os.path.join(MODELS_DIR, "user_id_map.pkl"))
movie_map = joblib.load(os.path.join(MODELS_DIR, "movie_id_map.pkl"))
movies_info = joblib.load(os.path.join(MODELS_DIR, "movies_info.pkl"))

HIGH_RATING_THRESHOLD = 4.0
TOP_INPUTS = 5      # user's top-rated movies that seed item-based scores
NEIGHBORS = 50      # similar users used by user-based scores


# ---------------------------------------------------------------------------
# Scoring + recommendation logic (mirrors notebook steps 5-7 and 9)
# ---------------------------------------------------------------------------
def user_based_scores(user_idx, k=NEIGHBORS):
    """Score every movie as the similarity-weighted SUM of neighbors' ratings."""
    sims = user_sim[user_idx].copy()
    sims[user_idx] = 0.0
    neighbors = np.argsort(-sims)[:k]
    neighbors = neighbors[sims[neighbors] > 0]
    if len(neighbors) == 0:
        return np.zeros(user_item.shape[1])
    w = sims[neighbors]
    return (user_item[neighbors] * w[:, None]).sum(axis=0)


def item_based_scores(user_idx, n_inputs=TOP_INPUTS):
    """Score movies by similarity to the user's top-rated movies."""
    row = user_item[user_idx]
    liked = np.where(row >= HIGH_RATING_THRESHOLD)[0]
    if len(liked) == 0:
        return np.zeros(user_item.shape[1])
    liked = liked[np.argsort(-row[liked])[:n_inputs]]
    weights = row[liked]
    return (weights[:, None] * item_sim[liked]).sum(axis=0) / weights.sum()


def svd_scores(user_idx):
    """Reconstruct predicted ratings from the fitted SVD latent factors."""
    scores = svd_model.inverse_transform(svd_model.transform(user_item[user_idx:user_idx + 1]))[0]
    return np.nan_to_num(scores, nan=0.0)


def recommend(user_idx, scores, top_n):
    scores = scores.copy()
    scores[user_item[user_idx] > 0] = -np.inf  # exclude movies already seen
    return np.argsort(-scores)[:top_n]


# ---------------------------------------------------------------------------
# Gradio interface (mirrors notebook step 11)
# ---------------------------------------------------------------------------
def recommend_for_app(user_id, method="User-based CF", top_n=10):
    top_n = int(top_n)
    user_id = int(user_id)
    if user_id not in user_map["id2idx"]:
        return pd.DataFrame({
            "movieId": ["-"],
            "title": [f"User {user_id} not found. Try a userId in "
                      f"[{user_map['idx2id'][0]}, {user_map['idx2id'][-1]}]."],
            "score": [np.nan],
        })
    ui = user_map["id2idx"][user_id]
    if method == "Item-based CF":
        scores = item_based_scores(ui)
    elif method == "SVD":
        scores = svd_scores(ui)
    else:
        scores = user_based_scores(ui)

    idx = recommend(ui, scores, top_n)
    return pd.DataFrame({
        "movieId": [movie_map["idx2id"][mi] for mi in idx],
        "title": [movies_info[movie_map["idx2id"][mi]]["title"] for mi in idx],
        "score": [round(float(scores[mi]), 2) for mi in idx],
    })


uid_min, uid_max = int(user_map["idx2id"][0]), int(user_map["idx2id"][-1])

demo = gr.Interface(
    fn=recommend_for_app,
    inputs=[
        gr.Number(minimum=uid_min, maximum=uid_max, value=uid_min,
                  step=1, label="User ID"),
        gr.Dropdown(["User-based CF", "Item-based CF", "SVD"],
                    value="User-based CF", label="Recommendation method"),
        gr.Slider(1, 20, value=10, step=1, label="Number of recommendations (top-N)"),
    ],
    outputs=gr.Dataframe(headers=["movieId", "title", "score"],
                         label="Top-N recommended movies"),
    title="Task 5: Movie Recommendation System — Live Prediction",
    description="Enter a user ID to get top-N movie recommendations based on similar users' "
                "tastes (user-based CF), similar movies (item-based CF), or latent factors (SVD).",
)

if __name__ == "__main__":
    demo.launch(share=True)