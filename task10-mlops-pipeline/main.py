import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas import PredictionOutput, SensorInput

# ---------------------------------------------------------------------------
# Artifact loading — done ONCE at startup (import time), never per request.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "failure_model.joblib"
ENCODER_PATH = MODELS_DIR / "feature_encoder.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"

_MISSING = [p for p in (MODEL_PATH, ENCODER_PATH, METADATA_PATH) if not p.exists()]
if _MISSING:
    raise FileNotFoundError(
        "Missing model artifacts (run in the repo root, or build the image): "
        + ", ".join(str(p) for p in _MISSING)
    )

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
metadata = joblib.load(METADATA_PATH)

FEATURE_COLUMNS = list(metadata["feature_columns"])
CLASS_NAMES = list(metadata["class_names"])
NO_FAILURE = "No failure"
NO_FAILURE_INDEX = CLASS_NAMES.index(NO_FAILURE)
DECISION_THRESHOLD = float(metadata["decision_threshold"])

app = FastAPI(
    title="Industrial Predictive Maintenance API",
    description=(
        "Wraps Task 9's shipped `failure_model.joblib` behind a typed HTTP API. "
        "Send the six raw AI4I sensor readings and receive the predicted failure "
        "type, confidence/risk, and the full 6-class probability breakdown. "
        "Task 9's four engineered features are recomputed server-side using the "
        "exact feature order stored in `metadata.joblib` — clients never need the "
        "model or encoder."
    ),
    version="1.0.0",
)


@app.exception_handler(RequestValidationError)
async def _validation_errors_are_400(request: Request, exc: RequestValidationError):
    """Map FastAPI's default 422 to a 400 so all bad inputs fail the same way."""
    return JSONResponse(
        status_code=400,
        content={
            "detail": "Request payload failed validation.",
            "errors": json.loads(json.dumps(exc.errors(), default=str)),
        },
    )


def build_features(payload: SensorInput) -> pd.DataFrame:
    """Rebuild Task 9's engineered 10-column row, preserving feature ORDER.

    `metadata['feature_columns']` is:
    Type | Air temperature | Process temperature | Rotational speed | Torque |
    Tool wear | Temperature difference | Thermal stress | Torque-speed ratio |
    Power proxy
    The encoder fit on exactly this order (numeric columns by name, one-hot
    `Type`), so the frame must match column-by-column.
    """
    air = payload.air_temperature
    process = payload.process_temperature
    speed = payload.rotational_speed
    torque = payload.torque
    wear = payload.tool_wear
    delta = process - air
    return pd.DataFrame(
        [{
            "Type": payload.product_type,
            "Air temperature": air,
            "Process temperature": process,
            "Rotational speed": speed,
            "Torque": torque,
            "Tool wear": wear,
            "Temperature difference": delta,
            "Thermal stress": delta * wear,
            "Torque-speed ratio": torque / max(speed, 1.0),
            "Power proxy": torque * speed,
        }],
        columns=FEATURE_COLUMNS,
    )


def predict(payload: SensorInput) -> PredictionOutput:
    """Replicates Task 9's `app.run_model` + frozen decision rule exactly."""
    features = build_features(payload)
    encoded = encoder.transform(features)
    probabilities = model.predict_proba(encoded)[0]

    best_index = int(np.argmax(probabilities))
    best_probability = float(probabilities[best_index])

    if best_index != NO_FAILURE_INDEX and best_probability < DECISION_THRESHOLD:
        label = NO_FAILURE
        risk = 1.0 - float(probabilities[NO_FAILURE_INDEX])
        confidence = float(probabilities[NO_FAILURE_INDEX])
    else:
        label = CLASS_NAMES[best_index]
        risk = 0.0 if label == NO_FAILURE else best_probability
        confidence = best_probability

    return PredictionOutput(
        predicted_failure_type=label,
        confidence=round(confidence, 4),
        risk=round(risk, 4),
        decision_threshold=round(DECISION_THRESHOLD, 4),
        failure_probabilities={
            name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities)
        },
    )


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "Industrial Predictive Maintenance API",
        "endpoints": {
            "predict": "POST /predict — sensor readings -> prediction",
            "health": "GET /health — liveness for Compose healthchecks",
            "docs": "/docs — interactive Swagger UI",
        },
        "model": "Task 9 failure_model.joblib (6-class Random Forest)",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "failure_model.joblib",
        "loaded": True,
        "decision_threshold": round(DECISION_THRESHOLD, 4),
    }


@app.post("/predict", response_model=PredictionOutput, name="predict")
def predict_endpoint(payload: SensorInput):
    return predict(payload)