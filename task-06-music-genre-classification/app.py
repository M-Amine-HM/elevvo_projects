"""Standalone Gradio app for Task 6 — Music Genre Classification.

Loads the trained CNN from ./models/ (saved by notebook.ipynb) and exposes an upgraded
Gradio interface (Blocks-based) for the same prediction pipeline as the notebook's final
section. Fully self-contained — no notebook dependency.

Run standalone locally from the task folder with:
    python app.py
"""

import os

import joblib
import numpy as np

import tensorflow as tf
import librosa
from PIL import Image

import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "cnn_best.keras"))
GENRES = joblib.load(os.path.join(MODELS_DIR, "class_names.pkl"))

IMG_SIZE = (128, 128)

# A small color per genre so the results feel a bit more "alive" than default gray bars.
GENRE_EMOJI = {
    "blues": "🎸",
    "classical": "🎻",
    "country": "🤠",
    "disco": "🕺",
    "hiphop": "🎤",
    "jazz": "🎷",
    "metal": "🤘",
    "pop": "🎧",
    "reggae": "🌴",
    "rock": "🎶",
}


def _emoji_for(genre: str) -> str:
    return GENRE_EMOJI.get(genre.lower(), "🎵")


def audio_to_spectrogram_image(audio_path, target_size=IMG_SIZE):
    """Convert a .wav file to a normalized 128x128 RGB mel-spectrogram image."""
    y, sr = librosa.load(audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img = (mel_norm * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L").convert("RGB").resize(target_size)
    return pil


def predict(audio_file):
    """Run inference and return everything the UI needs: headline text, confidence
    dict, and the spectrogram image so the user can see what the model "saw"."""
    if audio_file is None:
        empty = {g: 0.0 for g in GENRES}
        return (
            "### Upload a clip to get a prediction 🎶",
            empty,
            None,
        )

    pil_img = audio_to_spectrogram_image(audio_file)
    X = np.expand_dims(np.array(pil_img), axis=0)
    proba = model.predict(X, verbose=0)[0]

    pred_idx = int(np.argmax(proba))
    genre = GENRES[pred_idx]
    confidence_pct = float(proba[pred_idx]) * 100

    confidences = {g: round(float(p), 4) for g, p in zip(GENRES, proba)}

    headline = (
        f"## {_emoji_for(genre)} Predicted genre: **{genre.title()}**\n"
        f"Confidence: **{confidence_pct:.1f}%**"
    )

    return headline, confidences, pil_img


CUSTOM_CSS = """
#header-block {
    text-align: center;
    padding: 12px 0 4px 0;
}
#header-block h1 {
    font-size: 1.9rem;
    margin-bottom: 0.25rem;
}
#header-block p {
    color: var(--body-text-color-subdued);
}
#predict-btn {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
#result-md {
    padding: 8px 4px;
}
.genre-chip {
    display: inline-block;
    padding: 2px 10px;
    margin: 2px;
    border-radius: 999px;
    background: var(--block-background-fill);
    border: 1px solid var(--border-color-primary);
    font-size: 0.85rem;
}
"""

GENRE_CHIPS = " ".join(
    f'<span class="genre-chip">{_emoji_for(g)} {g.title()}</span>' for g in GENRES
)

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo"),
    css=CUSTOM_CSS,
    title="Music Genre Classifier",
) as demo:

    with gr.Column(elem_id="header-block"):
        gr.Markdown(
            "# 🎼 Music Genre Classification\n"
            "Upload a short `.wav` clip and a CNN trained on mel-spectrograms will "
            "guess the genre, with a confidence breakdown across all classes."
        )
        gr.HTML(f"<div>{GENRE_CHIPS}</div>")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            audio_in = gr.Audio(
                type="filepath",
                label="🎧 Upload audio (.wav)",
                sources=["upload", "microphone"],
            )
            with gr.Row():
                clear_btn = gr.ClearButton(value="Clear")
                predict_btn = gr.Button(
                    "🔮 Predict genre", variant="primary", elem_id="predict-btn"
                )
            gr.Markdown(
                "**Tip:** a 3–10 second clip works well — the model only looks at "
                "a single mel-spectrogram snapshot of the audio."
            )

        with gr.Column(scale=1):
            result_md = gr.Markdown(
                "### Upload a clip to get a prediction 🎶", elem_id="result-md"
            )
            confidence_out = gr.Label(
                label="Confidence across genres", num_top_classes=10)
            spectrogram_out = gr.Image(
                label="Mel-spectrogram (model input)", type="pil", height=200
            )

    clear_btn.add([audio_in, result_md, confidence_out, spectrogram_out])

    predict_btn.click(
        fn=predict,
        inputs=audio_in,
        outputs=[result_md, confidence_out, spectrogram_out],
    )
    audio_in.change(
        fn=predict,
        inputs=audio_in,
        outputs=[result_md, confidence_out, spectrogram_out],
    )

    gr.Markdown(
        "<center><sub>Task 6 · CNN trained on GTZAN-style mel-spectrograms · "
        "Model + labels loaded from <code>./models/</code></sub></center>"
    )

if __name__ == "__main__":
    demo.launch(share=True)
