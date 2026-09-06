# Task 7: Sales Forecasting

## Problem
Predict the next week's sales for each Walmart store and department based on historical weekly sales, store characteristics, and external economic factors.

## Dataset
[Walmart Sales Forecast (Kaggle)](https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast) — 421,570 weekly sales records across 45 stores and 81 departments (Feb 2010 – Oct 2012), plus store metadata, external features (temperature, fuel price, markdowns, CPI, unemployment), and holiday flags.

## Approach
- Feature engineering: Calendar features (day/month/week/year), 1/2/3-week lag values, 4/8/13-week rolling averages, day-since-start trend, store type encoding.
- Model(s) tried: Linear Regression (baseline), XGBoost Regressor (final).
- Why this model: XGBoost captures non-linear feature interactions and is robust to unscaled/mixed data, giving a large improvement over the linear baseline.

## Results
| Metric | Score |
|---|---|
| MAE | 1,360.85 |
| RMSE | 2,860.95 |

![results](assets/results.png)

## Interface
![gradio demo](assets/gradio-demo.png)

## Bonus work
- [x] Seasonal decomposition (statsmodels) of weekly sales into trend, seasonal, residual components.
- [x] Time-aware validation with TimeSeriesSplit (5 folds) on XGBoost.

## How to run
```bash
pip install -r requirements.txt
python app.py
```

*Note: MAE/RMSE are from the trained XGBoost model on the chronological test split (cutoff 2012-07-01). Run `notebook.ipynb` to regenerate exact scores.*
