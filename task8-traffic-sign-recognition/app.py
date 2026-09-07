"""Gradio app for Task 8 — Traffic Sign Recognition.
Industrial / Brutalist theme consistent with the internship series.

Loads the trained YOLOv8 model straight from ./models/ (no notebook dependency).
Run:  python app.py
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

WEIGHTS = MODELS_DIR / "best.pt"

if not WEIGHTS.exists():
    raise FileNotFoundError(
        f"Missing weights: {WEIGHTS}.\nTrain the model first (run notebook.ipynb) "
        "or export best.pt into ./models/."
    )

model = YOLO(str(WEIGHTS))

BG = "#FFFFFF"
BLACK = "#0A0A0A"
YELLOW = "#FFD700"
RED = "#E10600"
GREY_LT = "#F2F2F2"
GREY_MD = "#CCCCCC"


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(image, conf_threshold):
    if image is None:
        return None

    results = model.predict(
        image,
        imgsz=640,
        conf=float(conf_threshold),
        verbose=False,
    )[0]

    annotated = results.plot()  # drawn boxes + class labels + confidences
    return annotated[:, :, ::-1]  # BGR -> RGB for display


# ── Bonus: live webcam inference ─────────────────────────────────────────────
def run_webcam(conf_threshold=0.30, camera=0):
    """Live detection over a local camera feed; press q to quit."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam running — press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, imgsz=640, conf=conf_threshold, verbose=False)[0]
        cv2.imshow("GTSDB webcam — Task 8", res.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# ── CSS ───────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* --- Base --- */
body, .gradio-container {
    background: #FFFFFF !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #0A0A0A !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}
.gradio-container > .main > .wrap { padding: 10px 14px !important; gap: 8px !important; }

/* --- Compact header bar --- */
.app-header {
    display: flex;
    align-items: center;
    gap: 20px;
    border: 2.5px solid #0A0A0A;
    border-bottom: 5px solid #0A0A0A;
    padding: 10px 16px;
    margin-bottom: 10px;
    background: #FFFFFF;
}
.header-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #0A0A0A;
    background: #FFD700;
    padding: 2px 8px;
    white-space: nowrap;
    border: 1.5px solid #0A0A0A;
}
.app-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 17px;
    font-weight: 700;
    color: #0A0A0A;
    margin: 0;
    white-space: nowrap;
    letter-spacing: -0.3px;
}
.app-header p {
    font-size: 11.5px;
    color: #555;
    margin: 0;
}

/* --- Section labels --- */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    color: #0A0A0A;
    background: #FFD700;
    display: block;
    padding: 2px 8px;
    margin: 8px 0 5px;
    border: 1.5px solid #0A0A0A;
    width: 100%;
}

/* --- Panels / blocks --- */
.gradio-container .block,
.gradio-container .form {
    background: #FFFFFF !important;
    border: 2px solid #0A0A0A !important;
    border-radius: 0 !important;
    box-shadow: 3px 3px 0 #0A0A0A !important;
    padding: 6px 8px !important;
    margin: 0 !important;
}

/* --- Labels --- */
label span, .gradio-container label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 9.5px !important;
    font-weight: 600 !important;
    color: #0A0A0A !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
    margin-bottom: 2px !important;
}

/* --- Image I/O --- */
.gradio-container .image-container,
.gradio-container .image-wrap {
    border: 2px solid #0A0A0A !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 #0A0A0A !important;
    background: #F2F2F2 !important;
}

/* --- Sliders --- */
input[type=range] {
    accent-color: #FFD700;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    height: 20px !important;
    padding: 0 !important;
}
.gradio-slider .wrap { border: none !important; box-shadow: none !important; }
.gradio-slider { padding: 2px 0 !important; }

/* --- Run button --- */
#run-btn, button.primary {
    background: #FFD700 !important;
    color: #0A0A0A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: 3px solid #0A0A0A !important;
    border-radius: 0 !important;
    padding: 10px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 4px 4px 0 #0A0A0A !important;
    transition: transform 0.08s ease, box-shadow 0.08s ease;
    margin-top: 6px !important;
}
#run-btn:hover, button.primary:hover {
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0 #0A0A0A !important;
}
#run-btn:active, button.primary:active {
    transform: translate(4px, 4px) !important;
    box-shadow: 0 0 0 #0A0A0A !important;
}

/* --- Divider --- */
.divider {
    height: 3px;
    background: repeating-linear-gradient(
        90deg, #0A0A0A 0px, #0A0A0A 8px,
        #FFD700 8px, #FFD700 16px
    );
    margin: 8px 0 6px;
}

footer { display: none !important; }
.gr-prose { display: none !important; }
"""


# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="Task 8: Traffic Sign Recognition — Live Prediction") as demo:

    gr.HTML("""
    <div class="app-header">
      <span class="header-eyebrow">YOLOv8N · GTSDB</span>
      <h1>Task 8 — Traffic Sign Recognition</h1>
      <p>Locate and classify traffic signs inside full, chaotic road scenes.</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT — Inputs ────────────────────────────────────────────────
        with gr.Column(scale=5, min_width=420):

            gr.HTML('<div class="section-label">SCENE INPUT</div>')
            inp = gr.Image(
                type="pil",
                label="Full-resolution scene (upload)",
                sources=["upload"],
            )

            gr.HTML('<div class="section-label">DETECTOR SETTINGS</div>')
            conf_slider = gr.Slider(
                0.05, 0.90, value=0.25, step=0.05,
                label="Confidence threshold (bonus filter)",
                info="Ignores low-quality detections below this score.",
            )

            gr.HTML('<div class="divider"></div>')
            run_btn = gr.Button(
                "▶  DETECT TRAFFIC SIGNS", elem_id="run-btn", variant="primary")

            gr.HTML("""
            <div class="section-label">INDUSTRY CONSTRAINT</div>
            <p style="font-size:11px; color:#444; margin:4px 0 0;">
            Deployed model runs real-time on an embedded GPU.
            <b>FPS is a first-class metric</b> — see notebook section 8 for measured
            throughput on the validation batch.
            </p>
            """)

        # ── RIGHT — Output ───────────────────────────────────────────────
        with gr.Column(scale=6, min_width=480):

            gr.HTML('<div class="section-label">DETECTIONS</div>')
            out = gr.Image(
                label="Signs detected (bounding boxes + class + confidence)",
            )

    run_btn.click(
        fn=predict,
        inputs=[inp, conf_slider],
        outputs=[out],
    )

    inp.upload(
        fn=predict,
        inputs=[inp, conf_slider],
        outputs=[out],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 8 — Traffic Sign Recognition")
    parser.add_argument("--webcam", action="store_true",
                        help="run live webcam inference instead of the Gradio app")
    parser.add_argument("--conf", type=float, default=0.30,
                        help="confidence threshold for webcam mode")
    parser.add_argument("--camera", type=int, default=0,
                        help="camera index for webcam mode")
    args = parser.parse_args()

    if args.webcam:
        run_webcam(conf_threshold=args.conf, camera=args.camera)
    else:
        demo.launch(share=True)