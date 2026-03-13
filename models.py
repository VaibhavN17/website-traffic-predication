import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_models():
    """Define ML models to train"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
        'Ridge Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=10.0))
        ]),
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )

    return models


def train_models(X_train, y_train, X_test, y_test, selected_models=None, progress_callback=None):
    """Train multiple ML models and return results"""
    models = get_models()
    if selected_models:
        models = {k: v for k, v in models.items() if k in selected_models}

    results = {}
    trained_models = {}

    for i, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(i / len(models), f"Training {name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Clip negatives

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100

        results[name] = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R²': round(r2, 4),
            'MAPE (%)': round(mape, 2),
            'y_pred': y_pred,
            'y_test': y_test.values,
        }
        trained_models[name] = model

    if progress_callback:
        progress_callback(1.0, "Training complete!")

    return results, trained_models


def get_best_model(results, trained_models):
    """Select best model based on R² score"""
    best_name = max(results, key=lambda k: results[k]['R²'])
    return best_name, trained_models[best_name]


def predict_future(model, df, feature_cols, future_hours=168):
    """Predict traffic for next N hours (default 7 days)"""
    last_row = df.iloc[-1]
    last_dt = pd.to_datetime(last_row['datetime'])

    # Build future timestamps
    future_records = []
    for i in range(1, future_hours + 1):
        future_dt = last_dt + pd.Timedelta(hours=i)
        hour = future_dt.hour
        dow = future_dt.dayofweek
        month = future_dt.month
        week = future_dt.isocalendar()[1]
        quarter = (month - 1) // 3 + 1
        is_weekend = int(dow >= 5)

        import math
        record = {
            'datetime': future_dt,
            'hour': hour,
            'day_of_week': dow,
            'month': month,
            'week_of_year': week,
            'quarter': quarter,
            'is_weekend': is_weekend,
            'is_business_hours': int(9 <= hour <= 17 and not is_weekend),
            'is_morning_peak': int(8 <= hour <= 11),
            'is_evening_peak': int(18 <= hour <= 22),
            'is_lunch_time': int(12 <= hour <= 13),
            'is_late_night': int(0 <= hour <= 5),
            'is_q4': int(quarter == 4),
            'is_q1': int(quarter == 1),
            'hour_sin': math.sin(2 * math.pi * hour / 24),
            'hour_cos': math.cos(2 * math.pi * hour / 24),
            'dow_sin': math.sin(2 * math.pi * dow / 7),
            'dow_cos': math.cos(2 * math.pi * dow / 7),
            'month_sin': math.sin(2 * math.pi * month / 12),
            'month_cos': math.cos(2 * math.pi * month / 12),
            'week_sin': math.sin(2 * math.pi * week / 52),
            'week_cos': math.cos(2 * math.pi * week / 52),
            'time_index': len(df) + i,
        }
        future_records.append(record)

    future_df = pd.DataFrame(future_records)

    # Fill lag/rolling features using last known values
    for col in feature_cols:
        if col not in future_df.columns:
            # Use mean from recent data
            if col in df.columns:
                future_df[col] = df[col].iloc[-24:].mean()
            else:
                future_df[col] = 0

    available_features = [f for f in feature_cols if f in future_df.columns]
    X_future = future_df[available_features]

    predictions = model.predict(X_future)
    predictions = np.maximum(predictions, 0)

    future_df['predicted_visitors'] = predictions
    return future_df


def get_feature_importance(model, feature_cols, model_name):
    """Extract feature importance from tree-based models"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'named_steps'):
            inner = model.named_steps.get('model')
            if hasattr(inner, 'coef_'):
                importances = np.abs(inner.coef_)
            else:
                return None
        else:
            return None

        fi_df = pd.DataFrame({
            'Feature': feature_cols[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        return fi_df
    except Exception:
        return None
