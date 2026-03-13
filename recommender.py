import numpy as np
import pandas as pd


def compute_traffic_score(df, visitors_col='visitors'):
    """
    Compute a composite engagement score per (day_of_week, hour) slot.
    Score combines: visitor count, bounce rate, session duration, conversions.
    """
    group_cols = ['day_of_week', 'day_name', 'hour']
    agg_dict = {visitors_col: ['mean', 'median', 'std', 'sum']}

    optional_cols = {
        'bounce_rate': 'mean',
        'avg_session_duration': 'mean',
        'conversions': ['mean', 'sum'],
        'pages_per_visit': 'mean',
    }
    for col, agg in optional_cols.items():
        if col in df.columns:
            agg_dict[col] = agg

    grouped = df.groupby(group_cols).agg(agg_dict)
    grouped.columns = ['_'.join(c).strip('_') for c in grouped.columns]
    grouped = grouped.reset_index()

    # Normalize visitor count
    v_col = f'{visitors_col}_mean'
    grouped['visitor_score'] = (grouped[v_col] - grouped[v_col].min()) / (
        grouped[v_col].max() - grouped[v_col].min() + 1e-8
    )

    # Normalize engagement metrics
    engagement_score = grouped['visitor_score'].copy()

    if 'bounce_rate_mean' in grouped.columns:
        # Lower bounce = better engagement
        br = grouped['bounce_rate_mean']
        grouped['bounce_score'] = 1 - (br - br.min()) / (br.max() - br.min() + 1e-8)
        engagement_score += 0.3 * grouped['bounce_score']

    if 'avg_session_duration_mean' in grouped.columns:
        sd = grouped['avg_session_duration_mean']
        grouped['session_score'] = (sd - sd.min()) / (sd.max() - sd.min() + 1e-8)
        engagement_score += 0.3 * grouped['session_score']

    if 'conversions_mean' in grouped.columns:
        cv = grouped['conversions_mean']
        grouped['conversion_score'] = (cv - cv.min()) / (cv.max() - cv.min() + 1e-8)
        engagement_score += 0.2 * grouped['conversion_score']

    if 'pages_per_visit_mean' in grouped.columns:
        pp = grouped['pages_per_visit_mean']
        grouped['pages_score'] = (pp - pp.min()) / (pp.max() - pp.min() + 1e-8)
        engagement_score += 0.1 * grouped['pages_score']

    # Normalize final score to 0-100
    grouped['engagement_score'] = (engagement_score - engagement_score.min()) / (
        engagement_score.max() - engagement_score.min() + 1e-8
    ) * 100

    grouped['traffic_rank'] = grouped['engagement_score'].rank(ascending=False).astype(int)
    grouped = grouped.sort_values('engagement_score', ascending=False)

    return grouped


def get_top_slots(score_df, top_n=10):
    """Get top N time slots"""
    return score_df.head(top_n).copy()


def get_best_hours_per_day(score_df, top_n=3):
    """Get top N hours for each day of week"""
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    result = {}
    for day_name in days_order:
        day_data = score_df[score_df['day_name'] == day_name].nlargest(top_n, 'engagement_score')
        if len(day_data) > 0:
            result[day_name] = day_data[['hour', 'engagement_score', 'visitors_mean']].values.tolist()
    return result


def generate_recommendation_report(score_df, future_df=None):
    """Generate a comprehensive recommendation report"""
    top_slots = get_top_slots(score_df, 5)
    best_per_day = get_best_hours_per_day(score_df, 3)

    # Overall best hour
    best_hour = int(top_slots.iloc[0]['hour'])
    best_day = top_slots.iloc[0]['day_name']
    best_score = round(top_slots.iloc[0]['engagement_score'], 1)

    # Best time range (morning vs afternoon vs evening)
    morning_avg = score_df[score_df['hour'].between(6, 11)]['engagement_score'].mean()
    afternoon_avg = score_df[score_df['hour'].between(12, 17)]['engagement_score'].mean()
    evening_avg = score_df[score_df['hour'].between(18, 22)]['engagement_score'].mean()
    night_avg = score_df[~score_df['hour'].between(6, 22)]['engagement_score'].mean()

    best_period = max(
        [('Morning (6–11 AM)', morning_avg),
         ('Afternoon (12–5 PM)', afternoon_avg),
         ('Evening (6–10 PM)', evening_avg),
         ('Night (11 PM–5 AM)', night_avg)],
        key=lambda x: x[1]
    )

    # Weekday vs Weekend
    weekday_avg = score_df[score_df['day_of_week'] < 5]['engagement_score'].mean()
    weekend_avg = score_df[score_df['day_of_week'] >= 5]['engagement_score'].mean()

    # Peak traffic for predicted data
    predicted_peak = None
    if future_df is not None and 'predicted_visitors' in future_df.columns:
        peak_row = future_df.loc[future_df['predicted_visitors'].idxmax()]
        predicted_peak = {
            'datetime': peak_row['datetime'],
            'visitors': int(peak_row['predicted_visitors']),
        }

    report = {
        'best_slot': {'day': best_day, 'hour': best_hour, 'score': best_score},
        'top_5_slots': top_slots[['day_name', 'hour', 'engagement_score', 'visitors_mean']].to_dict('records'),
        'best_per_day': best_per_day,
        'best_period': best_period[0],
        'periods': {
            'Morning (6–11 AM)': round(morning_avg, 1),
            'Afternoon (12–5 PM)': round(afternoon_avg, 1),
            'Evening (6–10 PM)': round(evening_avg, 1),
            'Night (11 PM–5 AM)': round(night_avg, 1),
        },
        'weekday_vs_weekend': {
            'weekday': round(weekday_avg, 1),
            'weekend': round(weekend_avg, 1),
        },
        'predicted_peak': predicted_peak,
    }
    return report


def format_hour(hour):
    """Convert 24h to 12h format"""
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"
