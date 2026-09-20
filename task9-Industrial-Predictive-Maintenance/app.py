import html
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "failure_model.joblib"
ENCODER_PATH = MODELS_DIR / "feature_encoder.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"

if not all(path.exists() for path in (MODEL_PATH, ENCODER_PATH, METADATA_PATH)):
    raise FileNotFoundError(
        "Saved model artifacts are missing. Run notebook.ipynb before starting app.py."
    )

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
metadata = joblib.load(METADATA_PATH)

# --------------------------------------------------------------------------
# Defaults and scenario presets
# --------------------------------------------------------------------------
# Order: air (K), process (K), speed (rpm), torque (Nm), tool wear (min), type
DEFAULTS = (298.1, 308.6, 1551, 42.8, 0, "L")

PRESETS = {
    "Normal run": (298.1, 308.6, 1551, 42.8, 0, "L"),
    "Poor cooling": (302.5, 310.9, 1300, 45.0, 100, "L"),
    "Worn tool": (300.2, 309.8, 1450, 40.0, 235, "M"),
    "Heavy load": (298.5, 309.0, 1400, 68.0, 120, "L"),
    "Overstrain": (299.0, 309.5, 1380, 60.0, 200, "L"),
}

# Reference limits from the AI4I 2020 dataset documentation.
POWER_MIN_W, POWER_MAX_W = 3500.0, 9000.0
COOLING_MIN_DELTA_K = 8.6
OVERSTRAIN_LIMITS = {"L": 11000.0, "M": 12000.0, "H": 13000.0}


# --------------------------------------------------------------------------
# Model logic (decision rule unchanged)
# --------------------------------------------------------------------------
def make_features(air_temperature, process_temperature, rotational_speed, torque, tool_wear, product_type):
    air_temperature = float(air_temperature)
    process_temperature = float(process_temperature)
    rotational_speed = float(rotational_speed)
    torque = float(torque)
    tool_wear = float(tool_wear)
    return pd.DataFrame([{
        "Type": product_type,
        "Air temperature": air_temperature,
        "Process temperature": process_temperature,
        "Rotational speed": rotational_speed,
        "Torque": torque,
        "Tool wear": tool_wear,
        "Temperature difference": process_temperature - air_temperature,
        "Thermal stress": (process_temperature - air_temperature) * tool_wear,
        "Torque-speed ratio": torque / max(rotational_speed, 1.0),
        "Power proxy": torque * rotational_speed,
    }], columns=metadata["feature_columns"])


def run_model(air_temperature, process_temperature, rotational_speed, torque, tool_wear, product_type):
    """Returns the decision dict plus the full class probability list."""
    features = make_features(
        air_temperature, process_temperature, rotational_speed, torque, tool_wear, product_type
    )
    encoded = encoder.transform(features)
    probabilities = model.predict_proba(encoded)[0]
    class_names = list(metadata["class_names"])
    best_index = int(np.argmax(probabilities))
    best_probability = float(probabilities[best_index])
    threshold = float(metadata["decision_threshold"])
    no_failure_index = class_names.index("No failure")

    if best_index != no_failure_index and best_probability < threshold:
        label = "No failure"
        risk = 1.0 - float(probabilities[no_failure_index])
        confidence = float(probabilities[no_failure_index])
    else:
        label = class_names[best_index]
        risk = 0.0 if label == "No failure" else best_probability
        confidence = best_probability

    result = {
        "predicted_failure_type": label,
        "risk": round(risk, 4),
        "confidence": round(confidence, 4),
        "decision_threshold": round(threshold, 4),
    }
    return result, class_names, probabilities, no_failure_index


# --------------------------------------------------------------------------
# Result rendering
# --------------------------------------------------------------------------
def pct(value, digits=1):
    return f"{value * 100:.{digits}f}%"


def build_indicators(air, process, speed, torque, wear, ptype):
    delta = process - air
    power = torque * speed * 2 * np.pi / 60.0
    load = wear * torque
    limit = OVERSTRAIN_LIMITS.get(ptype, 11000.0)

    def tone(bad, near=False):
        return "bad" if bad else ("near" if near else "ok")

    return [
        {
            "name": "Temperature gap",
            "value": f"{delta:.1f} K",
            "hint": f"Cooling struggles below {COOLING_MIN_DELTA_K} K",
            "tone": tone(delta < COOLING_MIN_DELTA_K, delta < COOLING_MIN_DELTA_K + 1.0),
        },
        {
            "name": "Mechanical power",
            "value": f"{power:,.0f} W",
            "hint": f"Expected range {POWER_MIN_W:,.0f} to {POWER_MAX_W:,.0f} W",
            "tone": tone(
                power < POWER_MIN_W or power > POWER_MAX_W,
                power < POWER_MIN_W * 1.1 or power > POWER_MAX_W * 0.92,
            ),
        },
        {
            "name": "Overstrain load",
            "value": f"{load:,.0f} min·Nm",
            "hint": f"Limit for type {ptype} is {limit:,.0f}",
            "tone": tone(load > limit, load > limit * 0.9),
        },
    ]


def input_warnings(air, process, speed, torque, wear):
    notes = []
    if process <= air:
        notes.append(
            "Process temperature is not above air temperature. Check the temperature sensors.")
    if speed <= 0:
        notes.append(
            "Rotational speed is zero, so the machine is not running.")
    if torque < 0 or wear < 0:
        notes.append("Torque and tool wear cannot be negative.")
    return notes


def render_result(result, class_names, probabilities, no_failure_index, indicators, warnings):
    threshold = result["decision_threshold"]
    p_no_failure = float(probabilities[no_failure_index])
    any_failure = max(0.0, min(1.0, 1.0 - p_no_failure))

    failure_probs = [
        (name, float(p)) for i, (name, p) in enumerate(zip(class_names, probabilities))
        if i != no_failure_index
    ]
    top_name, top_prob = max(failure_probs, key=lambda item: item[1])

    label = result["predicted_failure_type"]
    if label != "No failure":
        state, title = "bad", f"Failure likely: {html.escape(label)}"
        detail = (f"The model is {pct(result['confidence'])} sure, above the "
                  f"{pct(threshold)} action threshold.")
        action = "Inspect the machine before the next production run."
    elif top_prob >= 0.5 * threshold:
        state, title = "near", "Watch closely"
        detail = (f"No failure is predicted, but {html.escape(top_name)} scores "
                  f"{pct(top_prob)} against an action threshold of {pct(threshold)}.")
        action = "Keep monitoring and recheck if any reading drifts."
    else:
        state, title = "ok", "Running normally"
        detail = f"The model gives {pct(p_no_failure)} to no failure."
        action = "No action needed."

    # Gauge
    gauge_value = round(any_failure * 100, 1)
    fill_style = f"stroke-dasharray:{gauge_value} 100;" if gauge_value >= 0.5 else "display:none;"
    gauge = f"""
    <svg class="gauge" viewBox="0 0 200 116" role="img"
         aria-label="Chance of any failure {gauge_value:.0f} percent">
      <path class="g-outline" d="M20 100 A80 80 0 0 1 180 100"/>
      <path class="g-track" d="M20 100 A80 80 0 0 1 180 100" pathLength="100"/>
      <path class="g-fill" d="M20 100 A80 80 0 0 1 180 100" pathLength="100" style="{fill_style}"/>
      <text x="100" y="86" class="g-num">{gauge_value:.0f}%</text>
      <text x="100" y="108" class="g-cap">chance of any failure</text>
    </svg>"""

    # Probability bars (sorted, failure tick = decision threshold)
    rows = [(class_names[no_failure_index], p_no_failure, True)] + \
           [(n, p, False) for n, p in failure_probs]
    rows.sort(key=lambda r: r[1], reverse=True)
    bars = ""
    for name, p, is_ok in rows:
        tick = "" if is_ok else f'<i class="tick" style="left:{threshold * 100:.1f}%"></i>'
        cls = "ok" if is_ok else "bad"
        bars += f"""
        <div class="bar-row">
          <span class="bar-name">{html.escape(name)}</span>
          <span class="bar-track"><b class="bar-fill {cls}" style="width:{p * 100:.1f}%"></b>{tick}</span>
          <span class="bar-val">{pct(p)}</span>
        </div>"""

    ind = "".join(
        f"""<div class="ind {i['tone']}">
              <span class="ind-name">{i['name']}</span>
              <span class="ind-val">{i['value']}</span>
              <span class="ind-hint">{i['hint']}</span>
            </div>""" for i in indicators
    )

    warn = ""
    if warnings:
        warn = '<div class="warn">' + \
            "".join(f"<p>{html.escape(w)}</p>" for w in warnings) + "</div>"

    return f"""
    <div class="result {state}">
      {warn}
      <div class="verdict">
        <div class="verdict-text">
          <span class="badge {state}"><i></i>{title}</span>
          <p class="detail">{detail}</p>
          <p class="action">{action}</p>
        </div>
        {gauge}
      </div>
      <div class="result-grid">
        <section>
          <h4>Failure type scores</h4>
          <div class="bars">{bars}</div>
          <p class="legend"><i class="tick-key"></i><span>Decision threshold ({pct(threshold, 0)}). A failure type must reach it to raise an alert; lower scores are reported as No failure.</span></p>
        </section>
        <section>
          <h4>What the sensors imply</h4>
          <div class="inds">{ind}</div>
        </section>
      </div>
    </div>"""


def predict_failure(air_temperature, process_temperature, rotational_speed, torque, tool_wear, product_type):
    try:
        air = float(air_temperature)
        process = float(process_temperature)
        speed = float(rotational_speed)
        tq = float(torque)
        wear = float(tool_wear)
    except (TypeError, ValueError):
        return (
            '<div class="result empty"><p>Enter a number for every sensor reading to see a result.</p></div>',
            {},
        )

    result, class_names, probabilities, no_failure_index = run_model(
        air, process, speed, tq, wear, product_type
    )
    indicators = build_indicators(air, process, speed, tq, wear, product_type)
    warnings = input_warnings(air, process, speed, tq, wear)
    return (
        render_result(result, class_names, probabilities,
                      no_failure_index, indicators, warnings),
        result,
    )


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
css = """
@import url('https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ---------- Tokens + Gradio variable overrides (forced light, beats dark theme) ---------- */
.gradio-container, .gradio-container.dark, .dark .gradio-container, .dark {
  --ink: #111111 !important; --paper: #ffffff !important;
  --yellow: #ffd23f !important; --pink: #ff8fb8 !important; --blue: #5aa9ff !important; --lime: #8ef07a !important;
  --orange: #ff9f43 !important; --red: #ff4d4d !important; --mint: #d9ffd1 !important;
  color-scheme: light !important;
  --body-background-fill: #f1ecff !important;
  --body-text-color: #111 !important; --body-text-color-subdued: #333 !important;
  --background-fill-primary: #fff !important; --background-fill-secondary: #fff !important;
  --block-background-fill: transparent !important; --block-border-width: 0px !important; --block-shadow: none !important;
  --block-label-text-color: #111 !important; --block-title-text-color: #111 !important; --block-info-text-color: #333 !important;
  --block-label-background-fill: #fff !important; --block-title-background-fill: transparent !important;
  --input-background-fill: #fff !important; --input-border-color: #111 !important; --input-border-width: 2px !important;
  --input-radius: 0px !important; --input-shadow: none !important; --input-shadow-focus: 2px 2px 0 #111 !important;
  --input-border-color-focus: #111 !important; --input-text-color: #111 !important;
  --border-color-primary: #111 !important; --slider-color: #5aa9ff !important;
  --checkbox-label-background-fill: #fff !important; --checkbox-label-background-fill-hover: #fff !important;
  --checkbox-label-background-fill-selected: #ffd23f !important; --checkbox-label-text-color: #111 !important;
  --checkbox-label-text-color-selected: #111 !important; --checkbox-label-border-color: #111 !important;
  --checkbox-label-border-color-selected: #111 !important; --checkbox-label-border-width: 2px !important;
  --code-background-fill: #fff !important;
  --layout-gap: 8px !important; --block-padding: 0px !important;
  font-family: 'Space Grotesk', system-ui, sans-serif !important;
  font-size: 13px;
}
html { color-scheme: light !important; background: #f1ecff !important; }
gradio-app { background: transparent !important; }
body {
  background-color: #f1ecff !important; color: #111 !important;
  background-image: radial-gradient(rgba(17,17,17,.2) 1.2px, transparent 1.2px);
  background-size: 20px 20px;
}
footer { display: none !important; }

/* ---------- Full-width, compact page ---------- */
.gradio-container, .gradio-container > .main, .gradio-container main,
main.fillable, .gradio-container .app {
  max-width: 100% !important; width: 100% !important; background: transparent !important;
}
.gradio-container { padding: 10px clamp(10px, 1.6vw, 22px) !important; margin: 0 !important; }
.gradio-container main { padding: 0 !important; gap: 10px !important; }

/* ---------- Header (one slim bar) ---------- */
.app-header {
  display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  background: var(--yellow); border: 3px solid var(--ink); box-shadow: 5px 5px 0 var(--ink);
  padding: 8px 16px;
}
.app-header h1 {
  margin: 0; font-family: 'Archivo Black', sans-serif; font-weight: 400;
  font-size: clamp(18px, 2.4vw, 24px); line-height: 1.1; color: var(--ink);
}
.app-header p { margin: 2px 0 0; font-size: 12px; font-weight: 500; color: var(--ink); }
.sticker {
  display: inline-block; background: var(--ink); color: #fff; padding: 3px 10px;
  font-weight: 700; font-size: 11.5px; transform: rotate(-2deg); white-space: nowrap;
}

/* ---------- Two-column layout ---------- */
#main-row {
  display: grid !important; grid-template-columns: minmax(340px, 5fr) minmax(0, 6fr);
  gap: 14px; align-items: start;
}
#main-row > .column { min-width: 0 !important; width: 100% !important; flex: none !important; }
.panel {
  background: var(--paper) !important; border: 3px solid var(--ink) !important; border-radius: 0 !important;
  box-shadow: 5px 5px 0 var(--ink) !important; padding: 12px 14px !important; gap: 8px !important;
}
.panel .block { border: none !important; box-shadow: none !important; background: transparent !important; padding: 0 !important; }
.panel .form { background: transparent !important; border: none !important; box-shadow: none !important; gap: 8px 14px !important; }
.panel .prose h3, .panel h3 {
  display: inline-block; margin: 0 !important; padding: 2px 10px; background: var(--ink);
  color: #fff !important; font-family: 'Archivo Black', sans-serif; font-weight: 400; font-size: 14px;
}

/* ---------- Inputs: 2-column grid of compact controls ---------- */
#inputs-grid { gap: 0 !important; }
#inputs-grid > .form, #inputs-grid:not(:has(> .form)) {
  display: grid !important; grid-template-columns: 1fr 1fr; align-items: end;
}
#inputs-grid .block { min-width: 0 !important; }
.panel span[data-testid="block-info"] { color: var(--ink) !important; font-weight: 700 !important; font-size: 12px !important; }
.panel input[type=number] {
  border: 2px solid var(--ink) !important; border-radius: 0 !important; background: #fff !important;
  font-weight: 700 !important; font-size: 12px !important; box-shadow: 2px 2px 0 var(--ink) !important;
  padding: 2px 4px !important; height: 26px !important; width: 68px !important;
}
.panel input[type=range] {
  height: 12px !important; border: 2px solid var(--ink) !important; border-radius: 0 !important;
  background-color: #fff !important; accent-color: var(--ink);
}
.panel input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 16px; height: 22px; background: var(--yellow);
  border: 2px solid var(--ink); border-radius: 0; box-shadow: 2px 2px 0 var(--ink); cursor: pointer;
}
.panel input[type=range]::-moz-range-thumb {
  width: 12px; height: 18px; background: var(--yellow); border: 2px solid var(--ink);
  border-radius: 0; box-shadow: 2px 2px 0 var(--ink); cursor: pointer;
}
#type-radio .wrap { gap: 6px; }
#type-radio .wrap label {
  border: 2px solid var(--ink) !important; border-radius: 0 !important; background: #fff !important;
  padding: 3px 16px !important; font-weight: 700 !important; font-size: 12px !important;
  color: var(--ink) !important; cursor: pointer;
}
#type-radio .wrap label.selected { background: var(--yellow) !important; box-shadow: 2px 2px 0 var(--ink); }
#type-radio input { accent-color: var(--ink); }
.panel .raw-acc {
  border: 2px solid var(--ink) !important; box-shadow: 3px 3px 0 var(--ink) !important;
  background: #fff !important; border-radius: 0 !important;
}
.panel .raw-acc .label-wrap { font-weight: 700; font-size: 12px; color: var(--ink); padding: 4px 8px !important; }

/* ---------- Buttons ---------- */
#preset-row { display: flex !important; flex-wrap: wrap; gap: 8px; }
button.preset {
  flex: 1 1 auto; min-width: 70px !important; color: var(--ink) !important; border: 2px solid var(--ink) !important;
  border-radius: 0 !important; box-shadow: 3px 3px 0 var(--ink) !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important; font-size: 12px !important;
  padding: 4px 8px !important; height: 28px !important; min-height: 0 !important;
  transition: transform .08s ease, box-shadow .08s ease;
}
button.preset:hover { transform: translate(-1px, -1px); box-shadow: 4px 4px 0 var(--ink) !important; }
button.preset:active { transform: translate(3px, 3px); box-shadow: 0 0 0 var(--ink) !important; }
#preset-0 { background: var(--lime) !important; }
#preset-1 { background: var(--blue) !important; }
#preset-2 { background: var(--orange) !important; }
#preset-3 { background: var(--pink) !important; }
#preset-4 { background: #c9b6ff !important; }
#preset-5 { background: #fff !important; }
:focus-visible { outline: 3px solid var(--ink) !important; outline-offset: 2px; }

/* ---------- Result ---------- */
.result { display: flex; flex-direction: column; gap: 4px; color: var(--ink); font-size: 12.5px; }
.result h4 { margin: 8px 0 6px; font-family: 'Archivo Black', sans-serif; font-weight: 400; font-size: 13px; }
.verdict {
  display: flex; gap: 12px; align-items: center; justify-content: space-between; flex-wrap: wrap;
  border: 3px solid var(--ink); box-shadow: 4px 4px 0 var(--ink); padding: 10px 14px;
}
.result.ok .verdict { background: var(--lime); }
.result.near .verdict { background: var(--yellow); }
.result.bad .verdict { background: var(--pink); }
.verdict-text { flex: 1 1 200px; min-width: 0; }
.badge {
  display: inline-flex; align-items: center; gap: 8px; background: var(--ink); color: #fff;
  padding: 4px 10px; transform: rotate(-1deg);
  font-family: 'Archivo Black', sans-serif; font-size: 15px;
}
.badge i { width: 10px; height: 10px; border: 2px solid #fff; background: #fff; }
.badge.ok i { background: var(--lime); }
.badge.near i { background: var(--yellow); }
.badge.bad i { background: var(--red); }
.detail { margin: 8px 0 2px; font-size: 13px; line-height: 1.35; font-weight: 500; }
.action { margin: 0; font-size: 12px; font-weight: 700; }

.gauge { width: 150px; max-width: 100%; flex: 0 0 auto; margin: 0 auto; }
.g-outline { fill: none; stroke: var(--ink); stroke-width: 26; }
.g-track { fill: none; stroke: #fff; stroke-width: 18; }
.g-fill { fill: none; stroke-width: 18; }
.result.ok .g-fill { stroke: var(--blue); }
.result.near .g-fill { stroke: var(--orange); }
.result.bad .g-fill { stroke: var(--red); }
.g-num { text-anchor: middle; font-family: 'Archivo Black', sans-serif; font-size: 30px; fill: var(--ink); }
.g-cap { text-anchor: middle; font-size: 10.5px; font-weight: 700; fill: var(--ink); }

.result-grid { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); gap: 4px 16px; }
.bar-row {
  display: grid; grid-template-columns: minmax(90px, 150px) 1fr 44px; gap: 8px;
  align-items: center; margin: 5px 0; font-size: 12px; font-weight: 500;
}
.bar-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 700; }
.bar-track { position: relative; height: 15px; background: #fff; border: 2px solid var(--ink); }
.bar-fill { display: block; height: 100%; }
.bar-fill.ok { background: var(--lime); }
.bar-fill.bad { background: var(--pink); }
.tick { position: absolute; top: -5px; width: 3px; height: 21px; margin-left: -1px; background: var(--ink); }
.bar-val { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
.legend { display: flex; align-items: flex-start; gap: 8px; margin: 6px 0 0; font-size: 11px; line-height: 1.3; color: #222; }
.tick-key { display: inline-block; flex: 0 0 auto; width: 3px; height: 14px; background: var(--ink); }

.inds { display: grid; grid-template-columns: 1fr; gap: 8px; }
.ind {
  display: grid; grid-template-columns: 1fr auto; column-gap: 8px; padding: 6px 10px; background: var(--mint);
  border: 2px solid var(--ink); box-shadow: 3px 3px 0 var(--ink);
}
.ind.near { background: var(--yellow); }
.ind.bad { background: var(--pink); }
.ind-name { font-size: 11.5px; font-weight: 700; }
.ind-val { grid-row: 1 / span 2; grid-column: 2; align-self: center; font-family: 'Archivo Black', sans-serif; font-size: 14px; font-variant-numeric: tabular-nums; }
.ind-hint { font-size: 10.5px; color: #222; }

.warn {
  background: var(--yellow); border: 2px solid var(--ink); box-shadow: 3px 3px 0 var(--ink);
  padding: 4px 10px; margin-bottom: 8px; font-size: 12px; font-weight: 500;
}
.warn p { margin: 1px 0; }
.result.empty { padding: 12px 0; }

/* ---------- Responsive ---------- */
@media (max-width: 1100px) { .result-grid { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 960px) {
  #main-row { grid-template-columns: minmax(0, 1fr); }
  .inds { grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
}
@media (max-width: 560px) {
  #inputs-grid > .form, #inputs-grid:not(:has(> .form)) { grid-template-columns: 1fr; }
  .app-header p { display: none; }
  .bar-row { grid-template-columns: 1fr 44px; gap: 2px 8px; }
  .bar-name { grid-column: 1 / -1; }
  .bar-track { grid-column: 1; }
}
@media (min-width: 961px) and (max-height: 700px) {
  .gauge { width: 120px; }
  .app-header p { display: none; }
  .detail { margin-top: 4px; }
}
/* ---------- Force readable colors even if the dark theme is active ---------- */
.panel, .panel .prose { color: #111 !important; }
.panel input[type=number], .panel input[type=text] { color: #111 !important; background: #fff !important; }
.panel label, .panel .prose p, .panel .prose li { color: #111 !important; }
.panel .prose h3 { color: #fff !important; background: #111 !important; }
.panel .raw-acc, .panel .raw-acc .block, .panel .raw-acc pre, .panel .raw-acc .json-holder,
.panel .raw-acc .label-wrap { background: #fff !important; color: #111 !important; }

/* ---------- Inputs box: white background, black feature text ---------- */
.panel, .panel .form, .panel .column, .panel .row, .panel .block, .panel .wrap,
#inputs-grid, #inputs-grid > .form, #inputs-grid .form, #inputs-grid .block, #inputs-grid > div {
  background-color: #fff !important;
}
.panel .form, #inputs-grid .form { border: none !important; box-shadow: none !important; }
.panel [data-testid="block-info"], .panel .head label, .panel .head span, .panel label > span,
#inputs-grid label, #inputs-grid span { color: #111 !important; }
#type-radio .wrap label.selected { background-color: #ffd23f !important; }
#type-radio .wrap label { background-color: #fff !important; }

/* ---------- Border + padding for the inputs box ---------- */
#inputs-grid {
  border: 3px solid #111 !important; box-shadow: 4px 4px 0 #111 !important; border-radius: 0 !important;
  padding: 14px 16px !important; margin-top: 4px;
}
#inputs-grid .form, #inputs-grid > .form { gap: 14px 18px !important; }
.panel { padding: 16px 18px !important; }

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
with gr.Blocks(css=css, title="Machine failure risk | Predictive maintenance") as demo:
    gr.HTML("""
    <div class="app-header">
      <div>
        <h1>Machine failure risk</h1>
        <p>Adjust the sensor readings and the assessment updates live.</p>
      </div>
      <span class="sticker">AI4I 2020 dataset</span>
    </div>
    """)

    with gr.Row(equal_height=False, elem_id="main-row"):
        with gr.Column(scale=5, min_width=300, elem_classes="panel"):
            gr.Markdown("### Sensor readings")
            with gr.Row(elem_id="preset-row"):
                preset_buttons = {
                    name: gr.Button(
                        name, size="sm", elem_classes="preset", elem_id=f"preset-{i}")
                    for i, name in enumerate(PRESETS)
                }
                reset_button = gr.Button(
                    "Reset", size="sm", elem_classes="preset", elem_id="preset-5")

            with gr.Column(elem_id="inputs-grid"):
                air = gr.Slider(
                    290, 310, value=DEFAULTS[0], step=0.1, label="Air temperature (K)")
                process = gr.Slider(
                    300, 320, value=DEFAULTS[1], step=0.1, label="Process temperature (K)")
                speed = gr.Slider(
                    1000, 3000, value=DEFAULTS[2], step=1, label="Rotational speed (rpm)")
                torque = gr.Slider(
                    0, 100, value=DEFAULTS[3], step=0.1, label="Torque (Nm)")
                wear = gr.Slider(
                    0, 300, value=DEFAULTS[4], step=1, label="Tool wear (min)")
                product_type = gr.Radio(
                    ["L", "M", "H"], value=DEFAULTS[5], label="Product quality (L / M / H)",
                    elem_id="type-radio",
                )

        with gr.Column(scale=6, min_width=300, elem_classes="panel"):
            gr.Markdown("### Assessment")
            result_html = gr.HTML()
            with gr.Accordion("Raw model output", open=False, elem_classes="raw-acc"):
                raw_output = gr.JSON(
                    label="Predicted failure type / risk / confidence")

    inputs = [air, process, speed, torque, wear, product_type]
    outputs = [result_html, raw_output]

    # Live prediction: on load and whenever an input changes.
    demo.load(predict_failure, inputs=inputs,
              outputs=outputs, show_progress="hidden")
    gr.on(
        triggers=[component.change for component in inputs],
        fn=predict_failure,
        inputs=inputs,
        outputs=outputs,
        show_progress="hidden",
        trigger_mode="always_last",
    )

    # Scenario presets and reset.
    for preset_name, button in preset_buttons.items():
        button.click(lambda values=PRESETS[preset_name]: list(
            values), outputs=inputs, show_progress="hidden")
    reset_button.click(lambda: list(DEFAULTS),
                       outputs=inputs, show_progress="hidden")

if __name__ == "__main__":
    demo.launch(share=True)
