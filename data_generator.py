import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_traffic_data(days=365, start_date=None):
    """
    Generate realistic synthetic website traffic data with:
    - Daily/hourly patterns
    - Weekly seasonality
    - Monthly trends
    - Special events (weekends, holidays)
    - Random noise
    """
    if start_date is None:
        start_date = datetime(2023, 1, 1)

    np.random.seed(42)
    records = []

    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        day_of_week = current_date.weekday()  # 0=Monday, 6=Sunday
        month = current_date.month
        day_of_year = current_date.timetuple().tm_yday

        # Monthly seasonal factor (higher in mid-year)
        monthly_factor = 1 + 0.3 * np.sin((month - 3) * np.pi / 6)

        # Weekly trend growth
        weekly_trend = 1 + (day_offset / days) * 0.4

        # Weekend vs weekday base
        if day_of_week in [5, 6]:  # Weekend
            day_base = 0.65
        else:
            day_base = 1.0

        for hour in range(24):
            # Hourly traffic pattern (bimodal: morning peak + evening peak)
            morning_peak = 0.7 * np.exp(-((hour - 10) ** 2) / 8)
            evening_peak = 1.0 * np.exp(-((hour - 20) ** 2) / 10)
            night_dip = 0.05 if hour < 6 else 0
            hourly_pattern = morning_peak + evening_peak + night_dip + 0.1

            # Base traffic
            base_traffic = 500 * hourly_pattern * day_base * monthly_factor * weekly_trend

            # Special events: high traffic during specific hours on weekdays
            if day_of_week < 5 and 9 <= hour <= 11:
                base_traffic *= 1.4  # Office hours boost
            if day_of_week < 5 and 12 <= hour <= 13:
                base_traffic *= 1.2  # Lunch break boost

            # Holiday effect (simulate some holidays with spikes)
            if month == 12 and current_date.day in range(20, 32):
                base_traffic *= 0.7  # Holiday slowdown

            # Add Gaussian noise
            noise = np.random.normal(1.0, 0.15)
            visitors = max(1, int(base_traffic * noise))

            # Bounce rate (lower during peak hours)
            bounce_rate = np.clip(0.65 - 0.15 * hourly_pattern + np.random.normal(0, 0.05), 0.2, 0.95)

            # Session duration (higher during content-rich hours)
            avg_session = np.clip(180 + 60 * hourly_pattern + np.random.normal(0, 20), 30, 600)

            # Page views per visitor
            pages_per_visit = np.clip(2.0 + 1.5 * hourly_pattern + np.random.normal(0, 0.3), 1.0, 8.0)

            # Conversions
            conversion_rate = np.clip(0.02 + 0.03 * hourly_pattern + np.random.normal(0, 0.005), 0.005, 0.15)
            conversions = max(0, int(visitors * conversion_rate))

            records.append({
                'datetime': current_date + timedelta(hours=hour),
                'date': current_date.date(),
                'hour': hour,
                'day_of_week': day_of_week,
                'day_name': current_date.strftime('%A'),
                'month': month,
                'month_name': current_date.strftime('%B'),
                'week_of_year': current_date.isocalendar()[1],
                'quarter': (month - 1) // 3 + 1,
                'is_weekend': int(day_of_week >= 5),
                'visitors': visitors,
                'pageviews': int(visitors * pages_per_visit),
                'bounce_rate': round(bounce_rate, 4),
                'avg_session_duration': round(avg_session, 1),
                'conversions': conversions,
                'pages_per_visit': round(pages_per_visit, 2),
            })

    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def load_or_generate_data(uploaded_file=None, days=365):
    """Load uploaded CSV or generate synthetic data"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_uploaded_data(df)
            return df, "uploaded"
        except Exception as e:
            return None, f"error: {str(e)}"
    else:
        df = generate_traffic_data(days=days)
        return df, "generated"


def preprocess_uploaded_data(df):
    """Preprocess user-uploaded data to standard format"""
    # Try to find datetime column
    datetime_cols = [c for c in df.columns if any(k in c.lower() for k in ['date', 'time', 'datetime', 'timestamp'])]
    if datetime_cols:
        df['datetime'] = pd.to_datetime(df[datetime_cols[0]])
    else:
        raise ValueError("No datetime column found. Please include a 'date' or 'datetime' column.")

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_name'] = df['datetime'].dt.strftime('%A')
    df['month'] = df['datetime'].dt.month
    df['month_name'] = df['datetime'].dt.strftime('%B')
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['datetime'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['date'] = df['datetime'].dt.date

    # Try to find visitors column
    visitor_cols = [c for c in df.columns if any(k in c.lower() for k in ['visitor', 'user', 'session', 'traffic', 'views'])]
    if visitor_cols and 'visitors' not in df.columns:
        df['visitors'] = df[visitor_cols[0]]

    return df
