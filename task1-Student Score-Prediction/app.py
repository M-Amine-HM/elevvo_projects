"""Standalone Gradio inference app for the Student Exam Score predictor.

Loads the trained Option-7 linear model and the single encoding map that were saved by the
training notebook (student_performance_analysis.ipynb), then exposes a Gradio UI with the
exact same widgets as the notebook's Section-12 demo.

Run standalone locally from the task1 directory with:
    python app.py
"""

import os
import warnings

import joblib
import numpy as np
import gradio as gr


def _base_dir():
    """Return the directory that contains the notebook's `models/` folder.

    The app now sits at the task root (next to `models/`), so this is simply the folder that
    holds this script. The app therefore works no matter what the current working directory is
    (e.g. when Hugging Face Spaces runs it).
    """
    return os.path.dirname(os.path.abspath(__file__))


# Load the saved artifacts (single source of truth): the fitted Option-7 LinearRegression and the
# one encode_maps dict (all 13 categorical encodings) used during training. Nothing encoding-related
# is hard-coded here, so inference always matches how the model was fitted.
model = joblib.load(os.path.join(_base_dir(), 'models', 'best_model.pkl'))
encode_maps = joblib.load(os.path.join(_base_dir(), 'models', 'encode_maps.pkl'))


def _sorted_keys(mp):
    """Return the label keys of an encode map ordered by their encoded value (ascending)."""
    return sorted(mp, key=lambda k: mp[k])


# Human-readable choices for the categorical dropdowns, derived from the loaded encode maps so the
# labels always match what the model expects. (Six numeric features stay as sliders.)
low_med_high    = _sorted_keys(encode_maps['Motivation_Level'])   # ['Low','Medium','High']
ny_choices      = _sorted_keys(encode_maps['Internet_Access'])    # ['No','Yes']
gender_choices  = _sorted_keys(encode_maps['Gender'])             # ['Female','Male']
school_choices  = _sorted_keys(encode_maps['School_Type'])        # ['Private','Public']
peer_choices    = _sorted_keys(encode_maps['Peer_Influence'])     # ['Negative','Neutral','Positive']
pedu_choices    = _sorted_keys(encode_maps['Parental_Education_Level'])  # ['High School','College','Postgraduate']
dist_choices    = _sorted_keys(encode_maps['Distance_from_Home'])  # ['Near','Moderate','Far']


def predict_score(hours_studied, attendance, parental_involvement, access_to_resources,
                  extracurricular_activities, sleep_hours, previous_scores, motivation_level,
                  internet_access, tutoring_sessions, family_income, teacher_quality,
                  school_type, peer_influence, physical_activity, learning_disabilities,
                  parental_education_level, distance_from_home, gender):
    """Build the 19-value row in the exact order the model was trained on, then predict."""
    # Collect every value in df-column order (Option 7 training order), encoding categoricals via
    # the single encode_maps dict loaded above.
    row = [
        hours_studied, attendance,
        encode_maps['Parental_Involvement'][parental_involvement],
        encode_maps['Access_to_Resources'][access_to_resources],
        encode_maps['Extracurricular_Activities'][extracurricular_activities],
        sleep_hours, previous_scores,
        encode_maps['Motivation_Level'][motivation_level],
        encode_maps['Internet_Access'][internet_access],
        tutoring_sessions,
        encode_maps['Family_Income'][family_income],
        encode_maps['Teacher_Quality'][teacher_quality],
        encode_maps['School_Type'][school_type],
        encode_maps['Peer_Influence'][peer_influence],
        physical_activity,
        encode_maps['Learning_Disabilities'][learning_disabilities],
        encode_maps['Parental_Education_Level'][parental_education_level],
        encode_maps['Distance_from_Home'][distance_from_home],
        encode_maps['Gender'][gender],
    ]
    # The model was fit on a DataFrame with column names; the 1-row array below triggers a harmless
    # "no valid feature names" UserWarning, so we suppress it (same as the training notebook).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        prediction = model.predict(np.array([row]))[0]
    # Model predicts 1D -> scalar; clip to the dataset's 0-100 scale for a sensible answer.
    return float(np.clip(prediction, 0, 100))


# Build the Demo interface with one input per feature, in the exact training order.
# NOTE: the numeric slider min/max ranges below were derived from the current dataset
# (StudentPerformanceFactors.csv) so the widgets sit on realistic values. Revisit them if a
# new/updated dataset is used, since the model was fit within these ranges.
inputs = [
    gr.Slider(1, 44, value=20, step=1, label='Hours_Studied'),
    gr.Slider(60, 100, value=80, step=1, label='Attendance'),
    gr.Dropdown(low_med_high, value='Medium', label='Parental_Involvement'),
    gr.Dropdown(low_med_high, value='Medium', label='Access_to_Resources'),
    gr.Dropdown(ny_choices, value='No', label='Extracurricular_Activities'),
    gr.Slider(4, 10, value=7, step=1, label='Sleep_Hours'),
    gr.Slider(50, 100, value=75, step=1, label='Previous_Scores'),
    gr.Dropdown(low_med_high, value='Medium', label='Motivation_Level'),
    gr.Dropdown(ny_choices, value='Yes', label='Internet_Access'),
    gr.Slider(0, 8, value=1, step=1, label='Tutoring_Sessions'),
    gr.Dropdown(low_med_high, value='Medium', label='Family_Income'),
    gr.Dropdown(low_med_high, value='Medium', label='Teacher_Quality'),
    gr.Dropdown(school_choices, value='Public', label='School_Type'),
    gr.Dropdown(peer_choices, value='Neutral', label='Peer_Influence'),
    gr.Slider(0, 6, value=3, step=1, label='Physical_Activity'),
    gr.Dropdown(ny_choices, value='No', label='Learning_Disabilities'),
    gr.Dropdown(pedu_choices, value='College', label='Parental_Education_Level'),
    gr.Dropdown(dist_choices, value='Near', label='Distance_from_Home'),
    gr.Dropdown(gender_choices, value='Male', label='Gender'),
]

# A single numeric output that shows the predicted exam score rounded to 1 decimal.
output = gr.Number(label='Predicted Exam Score', precision=1)

# Assemble the app. No share/inbrowser flags here — Hugging Face Spaces manages the server itself.
demo = gr.Interface(fn=predict_score, inputs=inputs, outputs=output,
                    title='Student Exam Score Predictor',
                    description='Enter student characteristics; the best (Option 7) model predicts '
                                'their exam score.')

if __name__ == '__main__':
    demo.launch()
