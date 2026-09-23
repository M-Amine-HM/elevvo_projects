import json
import os

import joblib
import gradio as gr
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, 'models')

COVER_TYPE_NAMES = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine',
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas Fir',
    7: 'Krummholz',
}

CONTINUOUS_FEATURES = [
    'Elevation', 'Aspect', 'Slope',
    'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
    'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
]

WILDERNESS_FEATURES = [
    'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
]

SOIL_FEATURES = [f'Soil_Type{i}' for i in range(1, 41)]

COLUMN_NAMES = (
    CONTINUOUS_FEATURES
    + WILDERNESS_FEATURES
    + SOIL_FEATURES
    + ['Cover_Type']
)


def load_models():
    with open(os.path.join(MODELS_DIR, 'params.json'), encoding='utf-8') as f:
        params = json.load(f)

    rf = joblib.load(os.path.join(MODELS_DIR, params.get('shipped_model', 'best_rf_model.joblib')))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    return rf, scaler


loaded_rf = None
loaded_scaler = None


def predict_cover_type(
    elevation, aspect, slope,
    h_hydrology, v_hydrology,
    h_roadways, hillshade_9am, hillshade_noon, hillshade_3pm, h_fire,
    wilderness_area, soil_type,
):
    global loaded_rf, loaded_scaler
    if loaded_rf is None:
        loaded_rf, loaded_scaler = load_models()

    feature_vector = np.zeros(54, dtype=np.float64)

    continuous_values = [
        elevation, aspect, slope,
        h_hydrology, v_hydrology,
        h_roadways, hillshade_9am, hillshade_noon, hillshade_3pm, h_fire,
    ]
    feature_vector[0:10] = continuous_values
    feature_vector[10 + int(wilderness_area) - 1] = 1.0
    feature_vector[14 + int(soil_type) - 1] = 1.0

    feature_df = pd.DataFrame([feature_vector], columns=COLUMN_NAMES[:-1])
    feature_df[CONTINUOUS_FEATURES] = loaded_scaler.transform(
        feature_df[CONTINUOUS_FEATURES]
    )

    pred_class = int(loaded_rf.predict(feature_df)[0])

    proba_raw = loaded_rf.predict_proba(feature_df)[0]
    probabilities = {
        COVER_TYPE_NAMES[i + 1]: float(proba_raw[i])
        for i in range(7)
    }

    return COVER_TYPE_NAMES[pred_class], probabilities


demo = gr.Interface(
    fn=predict_cover_type,
    inputs=[
        gr.Slider(1800, 3900, value=2800, label='Elevation (m)', step=1),
        gr.Slider(0, 360, value=180, label='Aspect (degrees)', step=1),
        gr.Slider(0, 66, value=10, label='Slope (degrees)', step=1),
        gr.Slider(0, 1400, value=250, label='Horiz. Dist. to Hydrology', step=1),
        gr.Slider(-170, 601, value=0, label='Vert. Dist. to Hydrology', step=1),
        gr.Slider(0, 7000, value=3000, label='Horiz. Dist. to Roadways', step=1),
        gr.Slider(0, 255, value=220, label='Hillshade 9am', step=1),
        gr.Slider(0, 255, value=235, label='Hillshade Noon', step=1),
        gr.Slider(0, 255, value=150, label='Hillshade 3pm', step=1),
        gr.Slider(0, 7000, value=3500, label='Horiz. Dist. to Fire Points', step=1),
        gr.Dropdown([1, 2, 3, 4], value=1, label='Wilderness Area'),
        gr.Dropdown(list(range(1, 41)), value=2, label='Soil Type'),
    ],
    outputs=[
        gr.Textbox(label='Predicted Cover Type'),
        gr.Label(label='Class Probabilities'),
    ],
    title='Forest Cover Type Classifier',
    description=(
        'Enter cartographic features to predict the forest cover type '
        'in the Roosevelt National Forest of northern Colorado.'
    ),
    theme=gr.themes.Soft(),
)


if __name__ == '__main__':
    demo.launch(show_error=True)