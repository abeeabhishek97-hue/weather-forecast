import streamlit as st
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Weather Forecast", page_icon="🌦️")

v = json.load(open("version.json"))
st.caption(f"Model v{v['version']} | Trained: {v['trained_on']} | "
           f"RMSE Technopark: {v['rmse_technopark']:.2f}°C | "
           f"RMSE Thampanoor: {v['rmse_thampanoor']:.2f}°C")

st.title("🌦️ Weather Forecast — Thiruvananthapuram")

cache = json.load(open("forecast_cache.json"))

LOCATIONS = {
    "Technopark": "technopark",
    "Thampanoor": "thampanoor",
}

tabs = st.tabs(list(LOCATIONS.keys()))

for tab, (loc_name, key) in zip(tabs, LOCATIONS.items()):
    with tab:
        data = cache[key]
        actuals = data["actuals"]
        actual_times = pd.to_datetime(data["times"])
        forecast = data["forecast"]
        last_time = actual_times[-1]
        forecast_times = [last_time + timedelta(hours=i+1) for i in range(24)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_times, y=actuals, name="Actual", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=forecast_times, y=forecast, name="Forecast", line=dict(color="tomato", dash="dash")))
        fig.update_layout(title=f"{loc_name} — 24h Forecast", xaxis_title="Time", yaxis_title="Temp (°C)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Min Forecast", f"{min(forecast):.1f}°C")
        col2.metric("Max Forecast", f"{max(forecast):.1f}°C")
        col3.metric("Avg Humidity", f"{data['humidity']:.0f}%")
