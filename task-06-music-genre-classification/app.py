"""Standalone Gradio app for Task 6 — Music Genre Classification.

Loads the trained tabular XGBoost model from ./models/ (saved by notebook.ipynb) and
exposes a Gradio interface (Blocks-based). The model is fed classic GTZAN audio
features (MFCCs, chroma, spectral centroid/bandwidth, rolloff, zero-crossing rate,
harmonic/percussive energy, tempo) reproduced from the uploaded .wav in the exact
column order the model was trained on (feature_names.pkl). The mel-spectrogram image
is shown in the UI as a visual preview only — it is no longer model input.

Run standalone locally from the task folder with:
    python app.py
"""

import os

import joblib
import numpy as np

import librosa
from PIL import Image

import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

tabular_model = joblib.load(os.path.join(MODELS_DIR, "tabular_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
FEATURE_NAMES = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
GENRES = list(encoder.classes_)

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
    """Convert a .wav file to a 128x128 RGB mel-spectrogram preview image (visual only)."""
    y, sr = librosa.load(audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    img = (mel_norm * 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L").convert("RGB").resize(target_size)
    return pil


def extract_tabular_features(audio_array, sr=22050):
    """Reproduce the exact 57 GTZAN features the tabular model was trained on.

    Mirrors the extraction that produced features_3_sec.csv: per-frame means and
    variances of chroma, rms, spectral centroid/bandwidth, rolloff, zero-crossing
    rate, harmonic/percussive energy, tempo, and MFCC1-20 (mean block, then var
    block), returned in the exact order of feature_names.pkl.
    """
    n_fft = 2048
    hop_length = 512

    chroma_stft = librosa.feature.chroma_stft(
        y=audio_array, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rmse = librosa.feature.rms(
        y=audio_array, frame_length=n_fft, hop_length=hop_length)
    spec_cent = librosa.feature.spectral_centroid(
        y=audio_array, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spec_bw = librosa.feature.spectral_bandwidth(
        y=audio_array, sr=sr, n_fft=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(
        y=audio_array, sr=sr, n_fft=n_fft, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(
        audio_array, frame_length=n_fft, hop_length=hop_length)
    harmony, percussive = librosa.effects.harmonic(audio_array), librosa.effects.percussive(audio_array)
    tempo, _ = librosa.beat.beat_track(y=audio_array, sr=sr)
    mfccs = librosa.feature.mfcc(
        y=audio_array, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)

    feat_map = {}
    for name, value in (
        ("chroma_stft", chroma_stft),
        ("rms", rmse),
        ("spectral_centroid", spec_cent),
        ("spectral_bandwidth", spec_bw),
        ("rolloff", rolloff),
        ("zero_crossing_rate", zcr),
        ("harmony", harmony),
        ("perceptr", percussive),
    ):
        feat_map[f"{name}_mean"] = float(np.mean(value))
        feat_map[f"{name}_var"] = float(np.var(value))

    feat_map["tempo"] = float(np.atleast_1d(tempo)[0])

    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)
    for i in range(1, 21):
        feat_map[f"mfcc{i}_mean"] = float(mfcc_means[i - 1])
        feat_map[f"mfcc{i}_var"] = float(mfcc_vars[i - 1])

    return np.array([feat_map[name] for name in FEATURE_NAMES])


def predict(audio_file):
    """Run inference and return everything the UI needs: headline text, confidence
    dict, and the spectrogram preview image."""
    if audio_file is None:
        empty = {g: 0.0 for g in GENRES}
        return (
            "### Upload a clip to get a prediction 🎶",
            empty,
            None,
        )

    try:
        y, sr = librosa.load(audio_file, sr=22050)
        if len(y) < 22050:
            raise ValueError("Clip too short — upload at least ~1 second of audio.")
        feat_vector = extract_tabular_features(y, sr)
        X = scaler.transform(feat_vector.reshape(1, -1))
        proba = tabular_model.predict_proba(X)[0]

        pred_idx = int(np.argmax(proba))
        genre = encoder.inverse_transform([pred_idx])[0]
        confidence_pct = float(proba[pred_idx]) * 100

        confidences = {g: round(float(p), 4) for g, p in zip(GENRES, proba)}

        headline = (
            f"## {_emoji_for(genre)} Predicted genre: **{genre.title()}**\n"
            f"Confidence: **{confidence_pct:.1f}%**"
        )

        pil_img = audio_to_spectrogram_image(audio_file)
        return headline, confidences, pil_img

    except Exception as exc:
        empty = {g: 0.0 for g in GENRES}
        return (
            f"### Could not process that clip 😕\n<sub>{exc}</sub>",
            empty,
            None,
        )


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
            "Upload a short `.wav` clip and a **XGBoost** model trained on MFCC / "
            "chroma / spectral features will guess the genre, with a confidence "
            "breakdown across all classes."
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
                "**Tip:** a 3–10 second clip works well — the model reads the audio "
                "frame-by-frame statistics (MFCCs, chroma, beats…), not the thumbnail."
            )

        with gr.Column(scale=1):
            result_md = gr.Markdown(
                "### Upload a clip to get a prediction 🎶", elem_id="result-md"
            )
            confidence_out = gr.Label(
                label="Confidence across genres", num_top_classes=10)
            spectrogram_out = gr.Image(
                label="Mel-spectrogram (preview only)", type="pil", height=200
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
        "<center><sub>Task 6 · Tabular XGBoost on GTZAN audio features · "
        "Model + scaler + labels loaded from <code>./models/</code></sub></center>"
    )

if __name__ == "__main__":
    demo.launch(share=True)
