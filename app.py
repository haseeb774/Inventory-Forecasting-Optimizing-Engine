# dashboard/app.py

import streamlit as st
import pandas as pd
from src.forecasting_feature import Reorder
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="NestHaven — Inventory Intelligence",
    page_icon="📦",
    layout="wide"
)

# ─────────────────────────────────────────
# Custom styling
# ─────────────────────────────────────────
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 28px; }
    .status-critical { background-color: #FCEBEB; color: #A32D2D; padding: 4px 10px; border-radius: 12px; font-size: 13px; font-weight: 600; }
    .status-warning  { background-color: #FAEEDA; color: #854F0B; padding: 4px 10px; border-radius: 12px; font-size: 13px; font-weight: 600; }
    .status-healthy  { background-color: #EAF3DE; color: #3B6D11; padding: 4px 10px; border-radius: 12px; font-size: 13px; font-weight: 600; }
    .status-overstock{ background-color: #E6F1FB; color: #185FA5; padding: 4px 10px; border-radius: 12px; font-size: 13px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load data (cached so it doesn't reload every click)
# ─────────────────────────────────────────
@st.cache_data(ttl=300)
def load_alerts():
    return pd.read_csv("data/outputs/reorder_alerts.csv")

@st.cache_data(ttl=300)
def load_features():
    return pd.read_csv("data/features/features.csv", parse_dates=["order_date"])

@st.cache_resource
def load_model():
    with open("models/xgboost_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data(ttl=300)
def get_last_retrain_time():
    path = "models/xgboost_model.pkl"
    if os.path.exists(path):
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts)
    return None

alerts_df   = load_alerts()
features_df = load_features()
saved_model = load_model()
retrain_time = get_last_retrain_time()

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("📦 NestHaven — Inventory Intelligence")
    st.caption("Smart inventory & demand forecasting system")
with col_status:
    st.metric("Last model retrain", retrain_time.strftime("%b %d, %I:%M %p") if retrain_time else "Unknown")

st.divider()

# ─────────────────────────────────────────
# Top metrics row
# ─────────────────────────────────────────
total_skus      = len(alerts_df)
critical_count  = (alerts_df["status"] == "CRITICAL").sum()
warning_count   = (alerts_df["status"] == "WARNING").sum()
overstock_count = (alerts_df["status"] == "OVERSTOCK").sum()
overstock_value = (
    alerts_df[alerts_df["status"] == "OVERSTOCK"]["current_stock"] *
    features_df.drop_duplicates("sku").set_index("sku")["price"].reindex(
        alerts_df[alerts_df["status"] == "OVERSTOCK"]["sku"]
    ).fillna(0).values
).sum() if overstock_count > 0 else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active SKUs", total_skus)
m2.metric("Critical alerts", int(critical_count), delta=None)
m3.metric("Warning alerts", int(warning_count))
m4.metric("Overstock value tied up", f"${overstock_value:,.0f}")

st.divider()

# ─────────────────────────────────────────
# Reorder Alerts Table
# ─────────────────────────────────────────
st.subheader("Reorder alerts")

status_order = {"CRITICAL": 0, "WARNING": 1, "HEALTHY": 2, "OVERSTOCK": 3}
display_df = alerts_df.copy()
display_df["sort_key"] = display_df["status"].map(status_order)
display_df = display_df.sort_values("sort_key").drop(columns=["sort_key"])

def style_status(val):
    colors = {
        "CRITICAL":  "background-color: #FCEBEB; color: #A32D2D; font-weight: 600;",
        "WARNING":   "background-color: #FAEEDA; color: #854F0B; font-weight: 600;",
        "HEALTHY":   "background-color: #EAF3DE; color: #3B6D11; font-weight: 600;",
        "OVERSTOCK": "background-color: #E6F1FB; color: #185FA5; font-weight: 600;",
    }
    return colors.get(val, "")

styled = display_df.style.applymap(style_status, subset=["status"])
st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "sku": "SKU",
        "product_title": "Product",
        "current_stock": st.column_config.NumberColumn("Current stock"),
        "daily_demand": st.column_config.NumberColumn("Daily demand", format="%.1f"),
        "reorder_point": st.column_config.NumberColumn("Reorder point", format="%.0f"),
        "safety_stock": st.column_config.NumberColumn("Safety stock", format="%.0f"),
        "recommended_order_qty": st.column_config.NumberColumn("Order qty"),
        "days_until_stockout": st.column_config.NumberColumn("Days left", format="%.1f"),
        "status": "Status",
        "urgency_message": "Action needed",
    }
)

st.divider()

# ─────────────────────────────────────────
# SKU-level deep dive
# ─────────────────────────────────────────
st.subheader("Demand forecast — SKU detail")

selected_sku = st.selectbox(
    "Select a product",
    options=alerts_df["sku"].tolist(),
    format_func=lambda x: f"{x} — {alerts_df[alerts_df['sku']==x]['product_title'].values[0]}"
)

col_chart, col_override = st.columns([3, 1])

with col_chart:
    sku_hist = features_df[features_df["sku"] == selected_sku].sort_values("order_date").tail(90)

    # Generate forward forecast for just this SKU
    @st.cache_data(ttl=300)
    def get_sku_forecast(sku, days=30):
        from src.model_train import ModelTrainer
        trainer = ModelTrainer()
        trainer.model = saved_model["model"]
        trainer.feature_cols = saved_model["feature_cols"]
        full_df = pd.read_csv("data/features/features.csv", parse_dates=["order_date"])
        return trainer.forecast(full_df, sku=sku, days=days)

    forecast_df = get_sku_forecast(selected_sku, days=30)
    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    fig = go.Figure()

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=sku_hist["order_date"], y=sku_hist["units_sold"],
        mode="lines", name="Actual sales", line=dict(color="#1D9E75", width=2)
    ))
    # Historical 7-day average
    fig.add_trace(go.Scatter(
        x=sku_hist["order_date"], y=sku_hist["rolling_mean_7d"],
        mode="lines", name="7-day average", line=dict(color="#378ADD", width=2, dash="dash")
    ))
    # Forward forecast — connects from last actual point
    bridge_x = [sku_hist["order_date"].iloc[-1]] + list(forecast_df["date"])
    bridge_y = [sku_hist["units_sold"].iloc[-1]] + list(forecast_df["forecast_units"])
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode="lines", name="30-day forecast", line=dict(color="#BA7517", width=2, dash="dot")
    ))
    # Shaded region marking the forecast zone
    fig.add_vrect(
        x0=sku_hist["order_date"].iloc[-1], x1=forecast_df["date"].iloc[-1],
        fillcolor="#BA7517", opacity=0.06, line_width=0
    )

    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title=None, yaxis_title="Units sold",
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{selected_sku}")

    st.caption(f"Forecast window: {forecast_df['date'].iloc[0].strftime('%b %d')} – {forecast_df['date'].iloc[-1].strftime('%b %d, %Y')} (next 30 days from latest data)")
    

with col_override:
    st.markdown("**Manual override**")
    row = alerts_df[alerts_df["sku"] == selected_sku].iloc[0]
    st.write(f"Current recommendation: **{int(row['recommended_order_qty'])} units**")

    override_qty = st.number_input(
        "Adjust order quantity",
        min_value=0, max_value=2000,
        value=int(row["recommended_order_qty"]),
        step=10
    )
    override_reason = st.text_input("Reason (optional)", placeholder="e.g. planned promo")

    if st.button("Save override", use_container_width=True):
        os.makedirs("data/overrides", exist_ok=True)
        override_log = {
            "sku": selected_sku,
            "original_qty": int(row["recommended_order_qty"]),
            "override_qty": override_qty,
            "reason": override_reason,
            "timestamp": datetime.now().isoformat()
        }
        log_path = "data/overrides/override_log.csv"
        log_df = pd.DataFrame([override_log])
        if os.path.exists(log_path):
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, index=False)
        st.success(f"Override saved — {override_qty} units for {selected_sku}")

st.divider()

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.caption("NestHaven Inventory Intelligence System — built by Haseeb U Rehman — powered by XGBoost")