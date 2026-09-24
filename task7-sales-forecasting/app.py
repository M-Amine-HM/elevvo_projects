"""Gradio app for Task 7 — Sales Forecasting.
Industrial / Brutalist theme — compact, viewport-fit layout.
All inputs visible without scrolling via 3-column CSS grid.

Run:  python app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import gradio as gr

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["hatch.linewidth"] = 1.6

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
feat_config = joblib.load(os.path.join(MODELS_DIR, "feature_config.pkl"))
feat_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

START_DATE = pd.Timestamp(feat_config["start_date"])
STORE_LIST = list(range(1, 46))
DEPT_LIST = list(range(1, 100))

DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_FORECAST_DATE = "2012-09-21"

# Real historical weekly sales per (Store, Dept) — pre-fills the app so the
# lag/rolling features reproduce exactly what the model was trained on.
HISTORY_MAP = {}
try:
    _train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         parse_dates=["Date"])
    for (_store, _dept), _grp in _train.sort_values("Date").groupby(["Store", "Dept"]):
        HISTORY_MAP[(_store, _dept)] = (
            _grp["Date"].tolist(), _grp["Weekly_Sales"].tolist())
    del _train
except Exception:
    pass


def recent_history(store, dept, target_date, n=13):
    """The last `n` observed weekly sales for (store, dept) before target_date."""
    dates, sales = HISTORY_MAP.get((int(store), int(dept)), ([], []))
    target = pd.Timestamp(target_date)
    prior = [s for d, s in zip(dates, sales) if pd.Timestamp(d) < target]
    return prior[-n:]


def parse_history(value):
    """Parse comma-separated weekly sales (most recent LAST) into floats."""
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]

BG = "#FFFFFF"
BLACK = "#0A0A0A"
YELLOW = "#FFD700"
GREY_LT = "#F2F2F2"
GREY_MD = "#CCCCCC"


# ── Prediction ────────────────────────────────────────────────────────────────
def predict_sales(
    store, dept, is_holiday, temperature, fuel_price,
    markdown1, markdown2, markdown3, markdown4, markdown5,
    cpi, unemployment, store_type, size,
    history_text, forecast_date,
):
    store_type_code = int(store_type.split("(")[1].rstrip(")"))

    # The data are Friday-dated weekly records: snap to the Friday of the ISO
    # week so calendar features always match what training saw (day_of_week 4).
    friday = pd.Timestamp(str(forecast_date)[:10])
    iso_year, iso_week, _ = friday.isocalendar()
    friday = pd.Timestamp.fromisocalendar(iso_year, iso_week, 5)
    day_of_week = friday.dayofweek          # always Friday == 4 in this dataset
    month = friday.month
    week_of_year = friday.isocalendar().week
    year = friday.year
    days_since = (friday - START_DATE).days

    hist = parse_history(history_text)
    if len(hist) < 3:
        raise gr.Error(
            "Provide at least 3 prior weekly sales values "
            "(most recent LAST, comma-separated).")
    # Lags are the 1/2/3 most recent observed weeks; the rolling features
    # average the trailing 4/8/13 weeks (shift(1)/min_periods=1 semantics).
    lag_1, lag_2, lag_3 = hist[-1], hist[-2], hist[-3]
    rolling_4w = float(np.mean(hist[-4:]))
    rolling_8w = float(np.mean(hist[-8:]))
    rolling_13w = float(np.mean(hist[-13:]))

    feature_values = [
        store, dept, int(is_holiday), temperature, fuel_price,
        markdown1, markdown2, markdown3, markdown4, markdown5,
        cpi, unemployment, store_type_code, size,
        day_of_week, month, week_of_year, year, days_since,
        lag_1, lag_2, lag_3, rolling_4w, rolling_8w, rolling_13w,
    ]
    assert len(feature_values) == len(feat_names)
    input_df = pd.DataFrame([feature_values], columns=feat_names)
    prediction = float(model.predict(input_df)[0])

    last_week_sales = lag_1
    trend_pct = (prediction - last_week_sales) / \
        max(abs(last_week_sales), 1) * 100
    trend_symbol = "▲" if trend_pct >= 0 else "▼"
    trend_label = f"{trend_symbol} {abs(trend_pct):.1f}%  vs last week"

    # Chart
    weeks = ["W−3", "W−2", "W−1", "FORECAST"]
    values = [hist[-3], hist[-2], last_week_sales, prediction]
    hatches = ["////", "////", "////", "XXXX"]
    fcolors = [GREY_LT, GREY_LT, GREY_LT, YELLOW]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.bar(weeks, values, color=fcolors, edgecolor=BLACK,
                  hatch=hatches, linewidth=2.0, width=0.55, zorder=3)

    max_val = max(values)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max_val * 0.022,
                f"${val:,.0f}", ha="center", va="bottom",
                color=BLACK, fontsize=8.5, fontweight="bold", fontfamily="monospace")

    ax.step(range(len(weeks)), values, where="mid", color=BLACK,
            linewidth=1.3, linestyle="--", alpha=0.45, zorder=4)

    fc_x, fc_w = bars[3].get_x(), bars[3].get_width()
    ax.plot([fc_x + 0.04, fc_x + fc_w - 0.04], [prediction, prediction],
            color=YELLOW, linewidth=4, solid_capstyle="butt", zorder=5)

    ax.set_ylim(0, max_val * 1.22)
    ax.set_ylabel("WEEKLY SALES ($)", color=BLACK, fontsize=8,
                  fontweight="bold", fontfamily="monospace", labelpad=8)
    ax.tick_params(colors=BLACK, labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily("monospace")
        lbl.set_fontweight("bold")
        lbl.set_fontsize(8.5)
    for lbl in ax.get_yticklabels():
        lbl.set_fontfamily("monospace")
    for spine in ax.spines.values():
        spine.set_edgecolor(BLACK)
        spine.set_linewidth(2.0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(True, axis="y", color=GREY_MD,
            linewidth=0.8, linestyle="-", zorder=0)
    ax.set_title(f"STORE {store}  ·  DEPT {dept}  ·  WK {int(week_of_year)} / {int(year)}",
                 color=BLACK, fontsize=8.5, fontweight="bold",
                 fontfamily="monospace", pad=8, loc="left")

    hist_p = mpatches.Patch(facecolor=GREY_LT, edgecolor=BLACK, hatch="////",
                            label="Historical", linewidth=1.5)
    fore_p = mpatches.Patch(facecolor=YELLOW,  edgecolor=BLACK, hatch="XXXX",
                            label="Forecast",   linewidth=1.5)
    ax.legend(handles=[hist_p, fore_p], loc="upper left", frameon=True,
              framealpha=1.0, edgecolor=BLACK, facecolor=BG, fontsize=7.5,
              prop={"family": "monospace", "weight": "bold"})

    plt.tight_layout(pad=0.9)
    return prediction, trend_label, fig


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

/* --- 3-col input grid --- */
.input-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 5px 8px;
    margin-bottom: 5px;
}
.input-grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 5px 8px;
    margin-bottom: 5px;
}
.input-grid-5 {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr 1fr;
    gap: 5px 8px;
    margin-bottom: 5px;
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

/* --- Inputs --- */
input[type="number"],
select,
textarea,
.gradio-container input,
.gradio-container select {
    background: #F2F2F2 !important;
    border: 2px solid #0A0A0A !important;
    border-radius: 0 !important;
    color: #0A0A0A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 4px 7px !important;
    box-shadow: none !important;
    height: 32px !important;
}
input[type="number"]:focus, select:focus {
    background: #FFFBE6 !important;
    outline: 2.5px solid #FFD700 !important;
    outline-offset: 0 !important;
    box-shadow: none !important;
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

/* --- Checkbox --- */
input[type="checkbox"] {
    accent-color: #FFD700;
    width: 15px; height: 15px;
    border: 2px solid #0A0A0A;
    border-radius: 0;
}
/* Align checkbox row compactly */
.checkbox-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 9.5px;
    font-weight: 600;
    text-transform: uppercase;
}

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

/* --- Output number --- */
.gradio-container .gr-number input {
    font-size: 28px !important;
    font-weight: 700 !important;
    height: auto !important;
    padding: 10px !important;
    background: #FFFBE6 !important;
    border: 2.5px solid #0A0A0A !important;
    text-align: center !important;
}

/* --- Trend textbox --- */
.gr-textbox textarea {
    background: #F2F2F2 !important;
    border: 2px solid #0A0A0A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #0A0A0A !important;
    padding: 6px 10px !important;
    box-shadow: none !important;
    resize: none !important;
    border-radius: 0 !important;
    text-align: center !important;
}

/* --- Plot --- */
.gr-plot, .gradio-container .gr-plot {
    border: 2px solid #0A0A0A !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 #0A0A0A !important;
    background: #FFFFFF !important;
}

/* --- Dashed divider --- */
.divider {
    height: 3px;
    background: repeating-linear-gradient(
        90deg, #0A0A0A 0px, #0A0A0A 8px,
        #FFD700 8px, #FFD700 16px
    );
    margin: 8px 0 6px;
}

/* --- Gap control for rows/cols --- */
.gradio-container .gap { gap: 6px !important; }
.gradio-container .row { gap: 6px !important; }

footer { display: none !important; }
.gr-prose { display: none !important; }
"""


# ── Interface ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS, title="Sales Forecast — Brutalist") as demo:

    # ── Compact header bar ─────────────────────────────────────────────────
    gr.HTML("""
    <div class="app-header">
      <span class="header-eyebrow">XGBOOST · WALMART</span>
      <h1>WEEKLY SALES FORECASTING</h1>
      <p>Department-level revenue prediction from store conditions and sales history.</p>
    </div>
    """)

    with gr.Row(equal_height=True):

        # ── LEFT — Inputs (compact grid) ───────────────────────────────────
        with gr.Column(scale=5, min_width=420):

            # STORE IDENTITY — 3 cols: store, dept, type | size, holiday (span)
            gr.HTML('<div class="section-label">STORE IDENTITY</div>')
            with gr.Row():
                store = gr.Dropdown(STORE_LIST, value=1,
                                    label="Store №", scale=1)
                dept = gr.Dropdown(DEPT_LIST,  value=1,
                                   label="Dept №",  scale=1)
                store_type = gr.Dropdown(["A (0)", "B (1)", "C (2)"],
                                         value="A (0)", label="Type", scale=1)
                size = gr.Number(value=150000, label="Size (sq ft)", scale=2)
                is_holiday = gr.Checkbox(label="Holiday", scale=1)

            # MARKET CONDITIONS — 4 across
            gr.HTML('<div class="section-label">MARKET CONDITIONS</div>')
            with gr.Row():
                temperature = gr.Number(value=60.0,  label="Temp (°F)")
                fuel_price = gr.Number(value=3.50,  label="Fuel ($)")
                cpi = gr.Number(value=220.0, label="CPI")
                unemployment = gr.Number(value=7.5,   label="Unempl. (%)")

            # MARKDOWNS — 5 across
            gr.HTML('<div class="section-label">PROMOTIONS / MARKDOWN</div>')
            with gr.Row():
                md1 = gr.Number(value=0, label="MD 1")
                md2 = gr.Number(value=0, label="MD 2")
                md3 = gr.Number(value=0, label="MD 3")
                md4 = gr.Number(value=0, label="MD 4")
                md5 = gr.Number(value=0, label="MD 5")

            gr.HTML('<div class="divider"></div>')
            run_btn = gr.Button("▶  RUN FORECAST",
                                elem_id="run-btn", variant="primary")

        # ── RIGHT — Outputs ────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=340):

            # SALES HISTORY — last 13 weeks, most recent LAST
            gr.HTML('<div class="section-label">SALES HISTORY</div>')
            history_text = gr.Textbox(
                value=", ".join(
                    f"{v:.0f}" for v in recent_history(1, 1, DEFAULT_FORECAST_DATE)),
                label="Last 13 Weekly Sales ($) — most recent LAST, "
                      "comma-separated",
                lines=3)

            # FORECAST PERIOD — single date; snapped to that ISO week's Friday
            gr.HTML('<div class="section-label">FORECAST PERIOD</div>')
            with gr.Row():
                forecast_date = gr.DateTime(
                    value=DEFAULT_FORECAST_DATE, include_time=False,
                    type="string", label="Forecast Week (Fri)")

            gr.HTML('<div class="section-label">RESULT</div>')
            prediction_out = gr.Number(label="Forecasted Weekly Sales ($)")
            trend_out = gr.Textbox(
                label="Week-over-week change", interactive=False)

            gr.HTML('<div class="section-label">SALES TREND</div>')
            chart_out = gr.Plot(label="", show_label=False)

    run_btn.click(
        fn=predict_sales,
        inputs=[
            store, dept, is_holiday, temperature, fuel_price,
            md1, md2, md3, md4, md5,
            cpi, unemployment, store_type, size,
            history_text, forecast_date,
        ],
        outputs=[prediction_out, trend_out, chart_out],
    )

if __name__ == "__main__":
    demo.launch(share=True)
