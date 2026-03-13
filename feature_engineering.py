import numpy as np
import pandas as pd


def engineer_features(df):
    """Create ML-ready features from traffic data"""
    df = df.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # === Time-based cyclical features ===
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # === Business hours flags ===
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17) & (df['is_weekend'] == 0)).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 8) & (df['hour'] <= 11)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
    df['is_lunch_time'] = ((df['hour'] >= 12) & (df['hour'] <= 13)).astype(int)
    df['is_late_night'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)

    # === Quarter flags ===
    df['is_q4'] = (df['quarter'] == 4).astype(int)
    df['is_q1'] = (df['quarter'] == 1).astype(int)

    # === Lag features (require historical data) ===
    if len(df) > 24:
        df['visitors_lag_1h'] = df['visitors'].shift(1)
        df['visitors_lag_24h'] = df['visitors'].shift(24)
        df['visitors_lag_48h'] = df['visitors'].shift(48)
        df['visitors_lag_168h'] = df['visitors'].shift(168)  # 1 week

        # Rolling statistics
        df['visitors_rolling_mean_6h'] = df['visitors'].shift(1).rolling(6).mean()
        df['visitors_rolling_mean_24h'] = df['visitors'].shift(1).rolling(24).mean()
        df['visitors_rolling_std_24h'] = df['visitors'].shift(1).rolling(24).std()
        df['visitors_rolling_max_24h'] = df['visitors'].shift(1).rolling(24).max()

        # Exponential weighted mean
        df['visitors_ewm_12h'] = df['visitors'].shift(1).ewm(span=12).mean()

    # Time index
    df['time_index'] = np.arange(len(df))

    # Drop NaN rows from lag features
    df = df.dropna().reset_index(drop=True)

    return df


def get_feature_columns():
    """Return list of feature columns for ML model"""
    return [
        'hour', 'day_of_week', 'month', 'week_of_year', 'quarter',
        'is_weekend', 'is_business_hours', 'is_morning_peak',
        'is_evening_peak', 'is_lunch_time', 'is_late_night',
        'is_q4', 'is_q1',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'month_sin', 'month_cos', 'week_sin', 'week_cos',
        'visitors_lag_1h', 'visitors_lag_24h', 'visitors_lag_48h', 'visitors_lag_168h',
        'visitors_rolling_mean_6h', 'visitors_rolling_mean_24h',
        'visitors_rolling_std_24h', 'visitors_rolling_max_24h',
        'visitors_ewm_12h', 'time_index',
    ]


def prepare_train_test(df, test_ratio=0.2):
    """Split data into train/test maintaining temporal order"""
    features = get_feature_columns()
    available_features = [f for f in features if f in df.columns]

    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[available_features]
    y_train = train_df['visitors']
    X_test = test_df[available_features]
    y_test = test_df['visitors']

    return X_train, y_train, X_test, y_test, available_features
