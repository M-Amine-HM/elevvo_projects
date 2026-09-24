# Traffic Sign Recognition (GTSDB) - YOLOv8 Object Detection

**Task 8 of the ML/DL portfolio.** Real-time traffic-sign detection on the German
Traffic Sign Detection Benchmark ([GTSDB](https://benchmark.ini.rub.de/gtsdb_dataset.html))
built with Ultralytics **YOLOv8**.

The task's core constraint is **inference speed (FPS) alongside detection accuracy (mAP)**:
an in-car sign detector has to run live, so both are measured and reported below.

## Problem

Detect all traffic signs in a road-scene photo, classify each into a sign category, and
do it fast enough for live deployment. GTSDB is a small dataset (~14 signs/class at
43 classes), so the winning approach coarsens the label space into 4 superclasses that
mirror how drivers actually perceive signs.

## Dataset

- 900 images: **600 train / 300 test**, 43 sign classes, YOLO `class cx cy w h` format.
- Some frames contain **no sign** and are label-free by design (background frames:
  94/600 train, 65/300 test). YOLO treats them as negatives and ignores them during
  training; they are kept because a live detector must say "no sign here".
- **Data hygiene:** one corrupt source label (`Test/labels/00862.txt`) referenced
  out-of-taxonomy class `100` (valid range 0-42). It plus its image was excluded from
  `dataset/` and `dataset_remapped/`. YOLO had already been discarding it as
  "corrupt", so the effective validation set is unchanged (299 test images,
  234 labeled / 65 background). See the notebook's "Data hygiene audit" section.
- The audit also checks every label for orphan/impossible values - none remain.

## Approach

- **Baseline** - `YOLOv8n` trained on the raw 43 classes (imgsz 640, 50 epochs).
  Expected low mAP: ~14 training examples per class is far below what a 43-class
  detector needs.
- **Pivot** - remap the 43 German sign classes into **4 superclasses** that match the
  Vienna Convention groups and how drivers read signs:
  `prohibitory`, `danger`, `mandatory`, `other` (see `data_remapped.yaml` +
  notebook remap cell). Train `YOLOv8s` on this label space.
- **Experiments** (with the superclass config) - heavy augmentation @1280px, frozen
  backbone. Neither beats the plain `YOLOv8s` run.
- Validation is always on the held-out GTSDB test split (never the train split).

## Results

All numbers are computed in the notebook by re-validating the shipped checkpoints on
the committed data (cell "8. Evaluation - computed comparison"), not hard-coded:

| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| `YOLOv8n` (43 classes) | 0.0529 | 0.0400 | 0.0157 | 0.5390 |
| **`YOLOv8s` (4 superclasses)** | **0.1605** | **0.1138** | 0.1983 | 0.3633 |
| `YOLOv8s` + heavy aug (1280px) | 0.1595 | 0.0964 | 0.1471 | 0.3441 |
| `YOLOv8s` + backbone freeze | 0.1385 | 0.0942 | 0.1849 | 0.3073 |

**Winner: `YOLOv8s` on the 4-superclass config** - ~3x the baseline mAP@0.5 with 1/10
the label complexity. Absolute mAP is low by design: ~14 signs/class at 43 classes
(≈60/class at 4 classes) is a fraction of a production training set (10k+ images/class);
the task is about the pipeline + honest reporting, not SOTA numbers.

### FPS (the core constraint)

**Measured in this repo** - winner, imgsz 640, letterboxed, 299 test images,
laptop CPU (AMD Ryzen 5 4600H):

| Runtime | FPS | Latency / image |
|---|---|---|
| PyTorch | **8.5** | 117.6 ms |
| ONNX Runtime | 6.8 | 148.2 ms |

**Recorded at training time** (Kaggle GPU, Tesla T4; 43-class `YOLOv8n` only -
see notebook cell "FPS benchmark"):

| Runtime | FPS |
|---|---|
| PyTorch | 69.3 (14.4 ms/image) |
| ONNX Runtime | 12.8 |

The ONNX export / benchmark cells run in the notebook and can be re-run from this
repo; add a CUDA+TensorRT pipeline to hit 60+ FPS on the shipped model.

## Interface

Gradio demo. Upload a road image, drag a confidence threshold, see the detection +
a text summary.

```bash
python app.py
```

![Traffic Sign Detection - Gradio demo preview](assets/gradio_demo.png)

The same UI is embedded in the notebook's final section; `temp_input.jpg` is the
scratch file used for passing uploads to YOLO (overwritten each run).

## Bonus work

- **ONNX export of the winning model** (was missing) + ONNX Runtime inference benchmark.
- **FPS benchmark** (PyTorch vs ONNX), letterboxed, reported explicitly.
- **Superclass remap** (43 groups -> 4 semantic superclasses) as label-space coarsening.
- **Data-hygiene audit** cell: orphan labels, background frames, corrupt-class scans.
- Full **markdown narrative** notebook with a computed (not hard-coded) comparison table.

## Repository layout

```
sign-detection.ipynb   main deliverable - training (Kaggle GPU) recorded,
                       evaluation/FPS/ONNX/artifacts re-computed from this repo
app.py                 standalone Gradio demo
models/                shipped weights: yolov8s_superclass_best.{pt,onnx},
                       yolov8n_43class_best.pt (git-committed deliverable)
requirements.txt       pinned environment
data.yaml              local 43-class config (relative paths, no absolute path key)
data_remapped.yaml     local 4-superclass config
dataset/               original 43-class GTSDB (gitignored; see README of the dataset)
dataset_remapped/      remapped 4-superclass copy (gitignored, regenerated by notebook)
runs/                  training outputs (gitignored, regenerable on Kaggle)
```

> `dataset/`, `dataset_remapped/`, `runs/` and `*.pt`/`*.onnx` are gitignored by the
> workspace root except the `models/` deliverable, which is force-included so the
> shipped weights are part of the submission.

## How to run

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python app.py                     # launch the Gradio demo
jupyter lab sign-detection.ipynb  # open the notebook
```

To recompute the evaluation/FPS/ONNX tables, run the tagged cells in the notebook
("Data hygiene audit", "Evaluation - computed comparison", "Bonus: winner ONNX/FPS",
"Save artifacts") from the repo root as shown. Ultralytics resolves `train:`/`val:`
relative to each YAML's `path`; the notebook pins `path` to the repo root at runtime
via `local_data()`.

## References

- [GTSDB dataset](https://benchmark.ini.rub.de/gtsdb_dataset.html)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)