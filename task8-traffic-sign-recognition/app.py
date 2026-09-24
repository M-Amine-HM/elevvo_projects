"""Traffic Sign Recognition (GTSDB) - Gradio demo.

No-code UI for the shipped 4-superclass YOLOv8s detector
(`models/yolov8s_superclass_best.pt`). Upload a road-scene image, drag the
confidence threshold, and see detected signs + a text summary.

Run:
    python app.py
"""

from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "yolov8s_superclass_best.pt"
SAMPLE_DIR = ROOT / "dataset_remapped" / "Test" / "images"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Restore models/yolov8s_superclass_best.pt "
        "or run the notebook cells '11. Save artifacts' first."
    )

model = YOLO(str(MODEL_PATH))


def predict(image: Image.Image, conf_threshold: float):
    """Detect traffic signs in a PIL image -> (annotated PIL image, text summary)."""
    if image is None:
        return None, "Please upload an image."

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    temp_path = (ROOT / "temp_input.jpg").as_posix()
    cv2.imwrite(temp_path, img_bgr)

    results = model.predict(temp_path, conf=conf_threshold, verbose=False)[0]
    annotated = cv2.cvtColor(results.plot(), cv2.COLOR_BGR2RGB)

    detections = [
        f"\u2022 {model.names[int(box.cls)]} \u2014 {float(box.conf):.2f}"
        for box in results.boxes
    ]
    if detections:
        summary = f"Found {len(detections)} sign(s):\n" + "\n".join(detections)
    else:
        summary = "No traffic signs detected."

    return Image.fromarray(annotated), summary


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Traffic Sign Detection") as demo:
        gr.Markdown("# \U0001F6A6 Traffic Sign Detection")
        gr.Markdown(
            "Upload a road scene image and adjust the confidence threshold to detect "
            "traffic signs with a YOLOv8s model (4 superclasses)."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Input Image")
                conf_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="Confidence Threshold",
                )
                run_btn = gr.Button("Detect Signs", variant="primary")
            with gr.Column():
                output_image = gr.Image(type="pil", label="Detected Signs")
                output_text = gr.Textbox(label="Detection Summary", lines=6)

        gr.Markdown("### Try a sample image")
        example_paths = sorted(SAMPLE_DIR.glob("*.jpg"))[:4] if SAMPLE_DIR.exists() else []
        gr.Examples(
            examples=[[str(p), 0.3] for p in example_paths],
            inputs=[input_image, conf_slider],
        )

        run_btn.click(
            fn=predict,
            inputs=[input_image, conf_slider],
            outputs=[output_image, output_text],
        )
    return demo


if __name__ == "__main__":
    build_demo().launch()