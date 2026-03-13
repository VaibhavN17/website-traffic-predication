# 📈 Website Traffic Prediction & Best Time Slot Recommendation System

## Overview
A complete Machine Learning system to predict website traffic and recommend optimal publishing time slots using historical traffic data.

---

## 🏗️ Architecture

```
traffic_system/
├── app.py                # Main Streamlit dashboard
├── data_generator.py     # Synthetic data generation & CSV loading
├── feature_engineering.py # ML feature extraction & engineering
├── models.py             # ML model training & prediction
├── recommender.py        # Time slot scoring & recommendation engine
└── requirements.txt      # Python dependencies
```

---

## 🤖 Machine Learning Models
| Model | Type | Description |
|-------|------|-------------|
| Random Forest | Ensemble | 150 trees, great for non-linear patterns |
| Gradient Boosting | Ensemble | Sequential boosting, robust performance |
| Ridge Regression | Linear | Fast, interpretable baseline |
| XGBoost | Gradient Boosting | High-performance (if installed) |

---

## ⚙️ Features Engineered
- **Cyclical encoding**: Hour, day-of-week, month, week-of-year (sin/cos)
- **Lag features**: 1h, 24h, 48h, 1-week lookback
- **Rolling statistics**: 6h/24h mean, std, max
- **Business flags**: business hours, morning peak, evening peak, lunch time
- **Seasonal**: quarter flags, weekend indicator, trend index
- **EWM**: Exponential weighted mean (12h span)

---

## 🎯 Recommendation Engine
Computes a composite **Engagement Score (0–100)** per (day, hour) slot:
- 40% → Visitor volume
- 30% → Bounce rate (inverted)
- 30% → Session duration
- 20% → Conversions
- 10% → Pages per visit

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📊 Dashboard Tabs
1. **Overview** — Dataset summary & daily traffic chart
2. **Exploratory Analysis** — Heatmaps, hourly/daily/monthly patterns
3. **Model Performance** — Metrics comparison, actual vs predicted, feature importance
4. **Traffic Forecast** — Next N days prediction with confidence band
5. **Recommendations** — Top time slots, per-day best hours, engagement heatmap

---

## 📁 CSV Upload Format
To use your own data, upload a CSV with at minimum:
- `datetime` or `date` column (parseable by pandas)
- `visitors` (or `users`, `sessions`, `traffic`) column

Optional columns for richer recommendations:
- `bounce_rate` (0.0–1.0)
- `avg_session_duration` (seconds)
- `conversions` (integer)
- `pages_per_visit` (float)

---

## 📈 Sample Output
- **Best slot**: Wednesday at 8:00 PM — Score: 87.4/100
- **Top period**: Evening (6–10 PM)
- **Best day type**: Weekday (avg score 72.3 vs Weekend 48.1)
