import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_traffic_data, load_or_generate_data
from feature_engineering import engineer_features, prepare_train_test, get_feature_columns
from models import train_models, get_best_model, predict_future, get_feature_importance
from recommender import compute_traffic_score, get_top_slots, generate_recommendation_report, format_hour

# ───────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Website Traffic Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────────────────────
# STYLING
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main bg */
    .stApp { background: #0f1117; color: #e8e8f0; }
    [data-testid="stSidebar"] { background: #161b2e; border-right: 1px solid #2a2f4e; }

    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102,126,234,0.3);
    }
    .main-header h1 { color: white; font-size: 2.2rem; margin: 0; font-weight: 700; }
    .main-header p { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; font-size: 1.05rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #232845 100%);
        border: 1px solid #2e3560; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h3 { color: #8892c8; font-size: 0.85rem; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card h2 { color: #fff; font-size: 1.8rem; margin: 0.3rem 0; font-weight: 700; }
    .metric-card p { color: #667eea; font-size: 0.8rem; margin: 0; }

    /* Recommendation cards */
    .rec-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
        border: 1px solid #2563eb44; border-radius: 12px;
        padding: 1.2rem; margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(37,99,235,0.15);
    }
    .rec-card-gold {
        background: linear-gradient(135deg, #3d2800 0%, #5c3d00 100%);
        border: 2px solid #f59e0b; border-radius: 12px;
        padding: 1.5rem; margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(245,158,11,0.3);
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #a5b4fc;
        border-bottom: 2px solid #3730a3; padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem;
    }
    .badge {
        background: #667eea; color: white; padding: 0.2rem 0.7rem;
        border-radius: 20px; font-size: 0.8rem; font-weight: 600;
    }
    .badge-green { background: #059669; }
    .badge-orange { background: #d97706; }
    .badge-red { background: #dc2626; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #8892c8; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #a5b4fc; }

    /* Selectbox, sliders */
    .stSelectbox > div, .stMultiSelect > div { background: #1a1f3a; border-color: #2e3560; }
    .stSlider > div > div > div { background: #667eea; }

    div[data-testid="stMetric"] {
        background: #1a1f3a; border: 1px solid #2e3560;
        border-radius: 10px; padding: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label { color: #8892c8 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #fff !important; }
</style>
""", unsafe_allow_html=True)

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
COLORS = px.colors.qualitative.Vivid

# ───────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ───────────────────────────────────────────────────────────────────────────────
for key in ['raw_df', 'engineered_df', 'results', 'trained_models',
            'best_model_name', 'best_model', 'future_df',
            'feature_cols', 'score_df', 'report']:
    if key not in st.session_state:
        st.session_state[key] = None

# ───────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    data_source = st.radio("Choose data source", ["Generate Synthetic Data", "Upload CSV"])

    if data_source == "Generate Synthetic Data":
        n_days = st.slider("Days of historical data", 90, 730, 365, step=30)
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload traffic CSV", type=["csv"])
        n_days = 365

    st.markdown("---")
    st.markdown("### 🤖 Model Settings")
    available_models = ['Random Forest', 'Gradient Boosting', 'Ridge Regression']
    try:
        import xgboost
        available_models.append('XGBoost')
    except:
        pass

    selected_models = st.multiselect(
        "Select models to train",
        available_models,
        default=['Random Forest', 'Gradient Boosting']
    )
    test_ratio = st.slider("Test split ratio", 0.1, 0.3, 0.2, 0.05)
    forecast_days = st.slider("Forecast horizon (days)", 1, 14, 7)

    st.markdown("---")
    st.markdown("### 🎯 Top N Recommendations")
    top_n = st.slider("Number of best slots", 3, 20, 10)

    st.markdown("---")
    if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Loading data..."):
            df, status = load_or_generate_data(uploaded_file, n_days)
            if df is not None:
                st.session_state.raw_df = df
            else:
                st.error(f"Data error: {status}")

        if st.session_state.raw_df is not None:
            with st.spinner("Engineering features..."):
                eng_df = engineer_features(st.session_state.raw_df)
                st.session_state.engineered_df = eng_df

            with st.spinner("Training models..."):
                X_train, y_train, X_test, y_test, feat_cols = prepare_train_test(eng_df, test_ratio)
                st.session_state.feature_cols = feat_cols

                prog = st.progress(0)
                def update_progress(val, msg):
                    prog.progress(val, msg)

                results, trained_models = train_models(
                    X_train, y_train, X_test, y_test,
                    selected_models=selected_models if selected_models else None,
                    progress_callback=update_progress
                )
                st.session_state.results = results
                st.session_state.trained_models = trained_models

                best_name, best_model = get_best_model(results, trained_models)
                st.session_state.best_model_name = best_name
                st.session_state.best_model = best_model

            with st.spinner("Generating predictions..."):
                future_df = predict_future(
                    best_model, eng_df, feat_cols,
                    future_hours=forecast_days * 24
                )
                st.session_state.future_df = future_df

            with st.spinner("Computing recommendations..."):
                score_df = compute_traffic_score(st.session_state.raw_df)
                st.session_state.score_df = score_df
                report = generate_recommendation_report(score_df, future_df)
                st.session_state.report = report

            st.success("✅ Pipeline complete!")

    # Info
    st.markdown("---")
    st.markdown("""
    <div style='color:#8892c8; font-size:0.8rem;'>
    <b>Models available:</b><br>
    🌲 Random Forest<br>
    📈 Gradient Boosting<br>
    📐 Ridge Regression<br>
    ⚡ XGBoost (if installed)<br><br>
    <b>Features engineered:</b><br>
    • Cyclical time encoding<br>
    • Lag & rolling features<br>
    • Business hour flags<br>
    • Seasonal indicators
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>📈 Website Traffic Prediction & Recommendation System</h1>
    <p>Predict future traffic & discover the best time slots to publish content using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🔍 Exploratory Analysis",
    "🤖 Model Performance",
    "🔮 Traffic Forecast",
    "🎯 Recommendations",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    if st.session_state.raw_df is None:
        st.info("👈 Configure settings in the sidebar and click **Run Full Pipeline** to start.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3>Step 1</h3>
                <h2>📂</h2>
                <p>Load or generate traffic data</p>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3>Step 2</h3>
                <h2>🤖</h2>
                <p>Train ML prediction models</p>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3>Step 3</h3>
                <h2>🎯</h2>
                <p>Get best time slot recommendations</p>
            </div>""", unsafe_allow_html=True)
    else:
        df = st.session_state.raw_df
        st.markdown("<div class='section-title'>📊 Dataset Summary</div>", unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric("Total Records", f"{len(df):,}")
        with c2:
            st.metric("Date Range", f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")
        with c3:
            st.metric("Avg Hourly Visitors", f"{df['visitors'].mean():.0f}")
        with c4:
            st.metric("Peak Visitors", f"{df['visitors'].max():,}")
        with c5:
            if 'bounce_rate' in df.columns:
                st.metric("Avg Bounce Rate", f"{df['bounce_rate'].mean():.1%}")

        # Quick overview chart
        st.markdown("<div class='section-title'>📅 Daily Traffic Overview</div>", unsafe_allow_html=True)
        daily = df.groupby('date')['visitors'].sum().reset_index()
        daily['date'] = pd.to_datetime(daily['date'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily['date'], y=daily['visitors'],
            mode='lines', name='Daily Visitors',
            line=dict(color='#667eea', width=1.5),
            fill='tozeroy', fillcolor='rgba(102,126,234,0.15)'
        ))
        # Rolling avg
        daily['rolling_avg'] = daily['visitors'].rolling(7).mean()
        fig.add_trace(go.Scatter(
            x=daily['date'], y=daily['rolling_avg'],
            mode='lines', name='7-Day Avg',
            line=dict(color='#f59e0b', width=2.5, dash='dash')
        ))
        fig.update_layout(
            template='plotly_dark', height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
            xaxis=dict(gridcolor='#1e2040'), yaxis=dict(gridcolor='#1e2040'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data sample
        with st.expander("🔎 Preview Raw Data"):
            st.dataframe(df.head(100), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    if st.session_state.raw_df is None:
        st.info("Run the pipeline first.")
    else:
        df = st.session_state.raw_df

        # ── Heatmap: Day × Hour ──
        st.markdown("<div class='section-title'>🔥 Traffic Heatmap: Day of Week × Hour</div>", unsafe_allow_html=True)
        pivot = df.pivot_table(values='visitors', index='day_name', columns='hour', aggfunc='mean')
        pivot = pivot.reindex(DAYS)

        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(pivot.columns),
            y=list(pivot.index), colorscale='Viridis',
            hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Avg Visitors: %{z:.0f}<extra></extra>'
        ))
        fig.update_layout(
            template='plotly_dark', height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
            xaxis_title='Hour of Day', yaxis_title='Day of Week',
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            # Hourly average
            st.markdown("<div class='section-title'>⏰ Avg Traffic by Hour</div>", unsafe_allow_html=True)
            hourly = df.groupby('hour')['visitors'].mean().reset_index()
            fig2 = px.bar(hourly, x='hour', y='visitors',
                          color='visitors', color_continuous_scale='Viridis',
                          labels={'hour': 'Hour of Day', 'visitors': 'Avg Visitors'})
            fig2.update_layout(template='plotly_dark', height=300,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                               showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # Daily of week average
            st.markdown("<div class='section-title'>📅 Avg Traffic by Day</div>", unsafe_allow_html=True)
            dow = df.groupby(['day_of_week', 'day_name'])['visitors'].mean().reset_index()
            dow = dow.sort_values('day_of_week')
            colors_bar = ['#ef4444' if d >= 5 else '#667eea' for d in dow['day_of_week']]
            fig3 = go.Figure(go.Bar(
                x=dow['day_name'], y=dow['visitors'],
                marker_color=colors_bar,
                hovertemplate='%{x}<br>Avg: %{y:.0f}<extra></extra>'
            ))
            fig3.update_layout(template='plotly_dark', height=300,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                               margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        # Monthly trend
        st.markdown("<div class='section-title'>📆 Monthly Traffic Trend</div>", unsafe_allow_html=True)
        monthly = df.groupby(['month', 'month_name'])['visitors'].agg(['mean', 'sum', 'std']).reset_index()
        monthly = monthly.sort_values('month')
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=monthly['month_name'], y=monthly['mean'],
            name='Avg Visitors', marker_color='#667eea', opacity=0.8
        ))
        fig4.add_trace(go.Scatter(
            x=monthly['month_name'], y=monthly['mean'],
            mode='lines+markers', name='Trend',
            line=dict(color='#f59e0b', width=2),
            marker=dict(size=8, color='#f59e0b')
        ))
        fig4.update_layout(template='plotly_dark', height=320,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                           xaxis_title='Month', yaxis_title='Avg Hourly Visitors',
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, use_container_width=True)

        # Distribution
        col3, col4 = st.columns(2)
        with col3:
            if 'bounce_rate' in df.columns:
                st.markdown("<div class='section-title'>📉 Bounce Rate Distribution</div>", unsafe_allow_html=True)
                fig5 = px.histogram(df, x='bounce_rate', nbins=40, color_discrete_sequence=['#a78bfa'])
                fig5.update_layout(template='plotly_dark', height=280,
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                                   margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig5, use_container_width=True)

        with col4:
            if 'avg_session_duration' in df.columns:
                st.markdown("<div class='section-title'>⏱️ Session Duration Distribution</div>", unsafe_allow_html=True)
                fig6 = px.histogram(df, x='avg_session_duration', nbins=40, color_discrete_sequence=['#34d399'])
                fig6.update_layout(template='plotly_dark', height=280,
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                                   margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig6, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    if st.session_state.results is None:
        st.info("Run the pipeline first.")
    else:
        results = st.session_state.results
        best_name = st.session_state.best_model_name

        st.markdown("<div class='section-title'>🏆 Model Comparison</div>", unsafe_allow_html=True)

        # Metrics table
        metrics_data = []
        for name, res in results.items():
            metrics_data.append({
                'Model': name,
                'MAE': res['MAE'],
                'RMSE': res['RMSE'],
                'R² Score': res['R²'],
                'MAPE (%)': res['MAPE (%)'],
                'Status': '🏆 Best' if name == best_name else '✓ Trained'
            })
        metrics_df = pd.DataFrame(metrics_data)

        # Color the best row
        def highlight_best(row):
            if row['Status'] == '🏆 Best':
                return ['background-color: rgba(102,126,234,0.25)'] * len(row)
            return [''] * len(row)

        st.dataframe(
            metrics_df.style.apply(highlight_best, axis=1)
                      .format({'R² Score': '{:.4f}', 'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'MAPE (%)': '{:.2f}'}),
            use_container_width=True, hide_index=True
        )

        # Metric comparison bars
        fig = make_subplots(rows=1, cols=4,
                            subplot_titles=['MAE (↓ better)', 'RMSE (↓ better)', 'R² (↑ better)', 'MAPE % (↓ better)'])

        model_names = list(results.keys())
        bar_colors = ['#667eea' if n == best_name else '#4b5563' for n in model_names]

        for col_idx, metric in enumerate(['MAE', 'RMSE', 'R²', 'MAPE (%)'], 1):
            vals = [results[n][metric] for n in model_names]
            fig.add_trace(go.Bar(
                x=model_names, y=vals,
                marker_color=bar_colors, showlegend=False,
                hovertemplate=f'%{{x}}<br>{metric}: %{{y:.4f}}<extra></extra>'
            ), row=1, col=col_idx)

        fig.update_layout(template='plotly_dark', height=350,
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                          margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        # Actual vs Predicted
        st.markdown(f"<div class='section-title'>📈 Actual vs Predicted — {best_name}</div>", unsafe_allow_html=True)

        best_res = results[best_name]
        n_show = min(500, len(best_res['y_test']))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=best_res['y_test'][:n_show], mode='lines', name='Actual',
            line=dict(color='#34d399', width=1.5)
        ))
        fig2.add_trace(go.Scatter(
            y=best_res['y_pred'][:n_show], mode='lines', name='Predicted',
            line=dict(color='#f59e0b', width=1.5, dash='dot')
        ))
        fig2.update_layout(
            template='plotly_dark', height=350,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
            xaxis_title='Test Sample Index', yaxis_title='Visitors',
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Residuals
        col1, col2 = st.columns(2)
        with col1:
            residuals = best_res['y_test'][:n_show] - best_res['y_pred'][:n_show]
            fig3 = px.histogram(residuals, nbins=50, color_discrete_sequence=['#a78bfa'],
                                title='Residual Distribution')
            fig3.add_vline(x=0, line_color='#f59e0b', line_dash='dash')
            fig3.update_layout(template='plotly_dark', height=280,
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                               margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            fi_df = get_feature_importance(
                st.session_state.best_model,
                st.session_state.feature_cols,
                best_name
            )
            if fi_df is not None:
                fig4 = px.bar(fi_df.head(12), x='Importance', y='Feature', orientation='h',
                              color='Importance', color_continuous_scale='Viridis',
                              title='Top Feature Importances')
                fig4.update_layout(template='plotly_dark', height=350,
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                                   showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    if st.session_state.future_df is None:
        st.info("Run the pipeline first.")
    else:
        future_df = st.session_state.future_df
        eng_df = st.session_state.engineered_df
        best_name = st.session_state.best_model_name

        st.markdown(f"<div class='section-title'>🔮 Traffic Forecast — Next {forecast_days} Days ({best_name})</div>",
                    unsafe_allow_html=True)

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Avg Predicted Visitors", f"{future_df['predicted_visitors'].mean():.0f}/hr")
        with c2:
            st.metric("Peak Predicted Visitors", f"{future_df['predicted_visitors'].max():,.0f}")
        with c3:
            peak_row = future_df.loc[future_df['predicted_visitors'].idxmax()]
            st.metric("Peak Time", f"{peak_row['datetime'].strftime('%a %b %d, %I %p')}")
        with c4:
            total = future_df['predicted_visitors'].sum()
            st.metric("Total Forecast Visitors", f"{total:,.0f}")

        # Forecast chart with history
        hist_recent = eng_df.tail(7 * 24)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_recent['datetime'], y=hist_recent['visitors'],
            mode='lines', name='Historical',
            line=dict(color='#34d399', width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=future_df['datetime'], y=future_df['predicted_visitors'],
            mode='lines', name='Forecast',
            line=dict(color='#f59e0b', width=2),
            fill='tozeroy', fillcolor='rgba(245,158,11,0.08)'
        ))
        # Add confidence band
        pred = future_df['predicted_visitors'].values
        noise = pred * 0.12
        fig.add_trace(go.Scatter(
            x=list(future_df['datetime']) + list(future_df['datetime'])[::-1],
            y=list(pred + noise) + list(pred - noise)[::-1],
            fill='toself', fillcolor='rgba(245,158,11,0.06)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Confidence Band', showlegend=True
        ))
        fig.add_vline(x=hist_recent['datetime'].iloc[-1], line_color='#a78bfa',
                      line_dash='dash', annotation_text='Forecast Start')
        fig.update_layout(
            template='plotly_dark', height=400,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
            xaxis_title='Date/Time', yaxis_title='Visitors',
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Daily breakdown of forecast
        st.markdown("<div class='section-title'>📅 Daily Forecast Breakdown</div>", unsafe_allow_html=True)
        future_df['forecast_date'] = future_df['datetime'].dt.date
        daily_forecast = future_df.groupby('forecast_date').agg(
            total=('predicted_visitors', 'sum'),
            avg=('predicted_visitors', 'mean'),
            peak=('predicted_visitors', 'max')
        ).reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=daily_forecast['forecast_date'].astype(str),
            y=daily_forecast['total'],
            name='Total Visitors',
            marker_color='#667eea', opacity=0.85
        ))
        fig2.add_trace(go.Scatter(
            x=daily_forecast['forecast_date'].astype(str),
            y=daily_forecast['peak'],
            mode='markers+lines', name='Peak Hour',
            line=dict(color='#f59e0b'), marker=dict(size=9, color='#f59e0b')
        ))
        fig2.update_layout(template='plotly_dark', height=320,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)

        # Hourly pattern of forecast
        st.markdown("<div class='section-title'>⏰ Predicted Traffic by Hour of Day</div>", unsafe_allow_html=True)
        hourly_forecast = future_df.groupby('hour')['predicted_visitors'].mean().reset_index()
        fig3 = go.Figure(go.Bar(
            x=hourly_forecast['hour'], y=hourly_forecast['predicted_visitors'],
            marker_color=hourly_forecast['predicted_visitors'],
            marker_colorscale='Plasma',
            hovertemplate='Hour: %{x}:00<br>Avg Predicted: %{y:.0f}<extra></extra>'
        ))
        fig3.update_layout(template='plotly_dark', height=300,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                           xaxis_title='Hour of Day', yaxis_title='Avg Predicted Visitors',
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        with st.expander("📋 Download Forecast Data"):
            csv = future_df[['datetime', 'hour', 'predicted_visitors']].to_csv(index=False)
            st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    if st.session_state.report is None:
        st.info("Run the pipeline first.")
    else:
        report = st.session_state.report
        score_df = st.session_state.score_df

        # ── Best Slot Banner ──
        best = report['best_slot']
        st.markdown(f"""
        <div class='rec-card-gold'>
            <h2 style='color:#f59e0b; margin:0;'>🏆 #1 Best Publishing Slot</h2>
            <h1 style='color:white; margin:0.3rem 0; font-size:2.2rem;'>
                {best['day']} at {format_hour(best['hour'])}
            </h1>
            <p style='color:#fbbf24; margin:0; font-size:1.1rem;'>
                Engagement Score: <b>{best['score']:.1f}/100</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Period & Weekday summary ──
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>📌 Best Period of Day</div>", unsafe_allow_html=True)
            periods = report['periods']
            max_period = max(periods, key=periods.get)
            for period, score in sorted(periods.items(), key=lambda x: -x[1]):
                color = '#f59e0b' if period == max_period else '#667eea'
                bar_pct = int(score / 100 * 100)
                st.markdown(f"""
                <div style='margin:0.4rem 0;'>
                    <span style='color:#ccc; font-size:0.9rem;'>{period}</span>
                    <div style='background:#1e2040; border-radius:6px; height:22px; overflow:hidden; margin-top:3px;'>
                        <div style='background:{color}; width:{bar_pct}%; height:100%; border-radius:6px;
                                    display:flex; align-items:center; padding-left:8px;'>
                            <span style='color:white; font-size:0.8rem; font-weight:600;'>{score:.1f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='section-title'>📌 Weekday vs Weekend</div>", unsafe_allow_html=True)
            wvw = report['weekday_vs_weekend']
            for label, score in [('Weekday', wvw['weekday']), ('Weekend', wvw['weekend'])]:
                color = '#34d399' if score == max(wvw.values()) else '#667eea'
                bar_pct = int(score)
                st.markdown(f"""
                <div style='margin:0.8rem 0;'>
                    <span style='color:#ccc; font-size:0.9rem;'>{label}</span>
                    <div style='background:#1e2040; border-radius:6px; height:28px; overflow:hidden; margin-top:5px;'>
                        <div style='background:{color}; width:{bar_pct}%; height:100%; border-radius:6px;
                                    display:flex; align-items:center; padding-left:10px;'>
                            <span style='color:white; font-size:0.9rem; font-weight:600;'>{score:.1f}/100</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Top N Slots Table + Chart ──
        st.markdown(f"<div class='section-title'>🏅 Top {top_n} Publishing Slots</div>", unsafe_allow_html=True)
        top_slots = get_top_slots(score_df, top_n)

        col3, col4 = st.columns([1, 1])
        with col3:
            display_cols = ['day_name', 'hour', 'engagement_score', 'visitors_mean']
            if 'bounce_rate_mean' in top_slots.columns:
                display_cols.append('bounce_rate_mean')
            if 'conversions_mean' in top_slots.columns:
                display_cols.append('conversions_mean')

            rename_map = {
                'day_name': 'Day', 'hour': 'Hour', 'engagement_score': 'Score (0-100)',
                'visitors_mean': 'Avg Visitors', 'bounce_rate_mean': 'Bounce Rate',
                'conversions_mean': 'Avg Conversions'
            }
            display_df = top_slots[display_cols].copy()
            display_df['hour'] = display_df['hour'].apply(format_hour)
            display_df = display_df.rename(columns=rename_map)
            display_df['Rank'] = range(1, len(display_df) + 1)
            display_df = display_df.set_index('Rank')

            st.dataframe(
                display_df.style.background_gradient(subset=['Score (0-100)'], cmap='YlOrRd'),
                use_container_width=True
            )

        with col4:
            fig = px.bar(
                top_slots.head(top_n), x='engagement_score',
                y=top_slots.head(top_n).apply(lambda r: f"{r['day_name']} {format_hour(int(r['hour']))}", axis=1),
                orientation='h', color='engagement_score',
                color_continuous_scale='Plasma',
                labels={'engagement_score': 'Engagement Score', 'y': 'Time Slot'}
            )
            fig.update_layout(template='plotly_dark', height=400,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                              showlegend=False, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)

        # ── Best hour per day ──
        st.markdown("<div class='section-title'>📅 Best Publishing Hours Per Day</div>", unsafe_allow_html=True)
        best_per_day = report['best_per_day']
        cols = st.columns(7)
        for i, (day, slots) in enumerate(best_per_day.items()):
            with cols[i]:
                is_weekend = day in ['Saturday', 'Sunday']
                badge_color = '#dc2626' if is_weekend else '#3b82f6'
                st.markdown(f"""
                <div style='background:#1a1f3a; border:1px solid #2e3560; border-radius:10px; padding:0.8rem; text-align:center;'>
                    <div style='font-size:0.75rem; font-weight:700; color:white; background:{badge_color};
                                border-radius:6px; padding:0.2rem; margin-bottom:0.5rem;'>{day[:3]}</div>
                """, unsafe_allow_html=True)

                for rank, (hour, score, visitors) in enumerate(slots, 1):
                    star = '⭐' if rank == 1 else f'#{rank}'
                    st.markdown(f"""
                    <div style='font-size:0.8rem; margin:0.3rem 0; color:#ccc;'>
                        {star} <b style='color:#f59e0b;'>{format_hour(int(hour))}</b><br>
                        <span style='color:#8892c8; font-size:0.7rem;'>{visitors:.0f} avg</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

        # ── Engagement heatmap ──
        st.markdown("<div class='section-title'>🔥 Engagement Score Heatmap</div>", unsafe_allow_html=True)
        score_pivot = score_df.pivot_table(values='engagement_score', index='day_name', columns='hour')
        score_pivot = score_pivot.reindex(DAYS)

        fig5 = go.Figure(go.Heatmap(
            z=score_pivot.values, x=list(score_pivot.columns), y=list(score_pivot.index),
            colorscale='RdYlGn',
            hovertemplate='Day: %{y}<br>Hour: %{x}:00<br>Score: %{z:.1f}<extra></extra>'
        ))
        fig5.update_layout(template='plotly_dark', height=350,
                           paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15,17,23,0.8)',
                           xaxis_title='Hour of Day', yaxis_title='Day of Week',
                           margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig5, use_container_width=True)

        # ── Predicted future peak ──
        if report['predicted_peak']:
            peak = report['predicted_peak']
            st.markdown(f"""
            <div class='rec-card' style='text-align:center; margin-top:1rem;'>
                <h3 style='color:#a5b4fc; margin:0;'>🔮 Predicted Peak in Next {forecast_days} Days</h3>
                <h2 style='color:white; margin:0.3rem 0;'>{peak['datetime'].strftime('%A, %B %d at %I:%M %p')}</h2>
                <p style='color:#34d399; margin:0; font-size:1.1rem;'>Expected {peak['visitors']:,} visitors</p>
                <p style='color:#8892c8; margin:0.3rem 0 0; font-size:0.85rem;'>📌 Consider scheduling your most important content around this time</p>
            </div>
            """, unsafe_allow_html=True)

        # Download report
        st.markdown("---")
        report_csv = score_df[['day_name', 'hour', 'engagement_score', 'visitors_mean']].copy()
        report_csv['hour_label'] = report_csv['hour'].apply(format_hour)
        report_csv = report_csv.sort_values('engagement_score', ascending=False)
        csv_bytes = report_csv.to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Recommendation Report (CSV)",
            csv_bytes, "time_slot_recommendations.csv", "text/csv",
            use_container_width=True
        )
