import streamlit as st
import json
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Weather Forecast", page_icon="🌦️")

v = json.load(open("version.json"))
st.caption(f"Model v{v['version']} | Trained: {v['trained_on']} | "
           f"RMSE Technopark: {v['rmse_technopark']:.2f}°C | "
           f"RMSE Thampanoor: {v['rmse_thampanoor']:.2f}°C")

st.title("🌦️ Weather Forecast — Thiruvananthapuram")

LOCATIONS = {
    "Technopark": {"lat": 8.5574, "lon": 76.8800, "key": "technopark"},
    "Thampanoor": {"lat": 8.4875, "lon": 76.9525, "key": "thampanoor"},
}

def fetch_recent(lat, lon, hours=72):
    end = datetime.utcnow().date()
    start = end - timedelta(days=4)
    url = "https://api.open-meteo.com/v1/forecast"
    r = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "start_date": str(start), "end_date": str(end),
        "timezone": "Asia/Kolkata",
    })
    data = r.json()["hourly"]
    df = pd.DataFrame(data).rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.tail(hours)

def make_forecast(df, key):
    scaler = pickle.load(open(f"models/{key}_scaler.pkl", "rb"))
    from tensorflow import keras
    model = keras.models.load_model(f"models/{key}_model.keras")

    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    features = ["temperature_2m","relative_humidity_2m","precipitation","wind_speed_10m","hour","dayofweek"]

    scaled = scaler.transform(df[features].values)
    X = scaled[-48:].reshape(1, 48, len(features))
    pred_scaled = model.predict(X, verbose=0)[0]

    dummy = np.zeros((24, len(features)))
    dummy[:, 0] = pred_scaled
    pred_temp = scaler.inverse_transform(dummy)[:, 0]
    return pred_temp

tabs = st.tabs(list(LOCATIONS.keys()))

for tab, (loc_name, info) in zip(tabs, LOCATIONS.items()):
    with tab:
        with st.spinner("Loading data..."):
            df = fetch_recent(info["lat"], info["lon"])
            forecast = make_forecast(df, info["key"])

        actuals = df["temperature_2m"].values[-48:]
        actual_times = df["datetime"].values[-48:]

        last_time = pd.to_datetime(actual_times[-1])
        forecast_times = [last_time + timedelta(hours=i+1) for i in range(24)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_times, y=actuals, name="Actual", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=forecast_times, y=forecast, name="Forecast", line=dict(color="tomato", dash="dash")))
        fig.update_layout(title=f"{loc_name} — 24h Forecast", xaxis_title="Time", yaxis_title="Temp (°C)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Min Forecast", f"{forecast.min():.1f}°C")
        col2.metric("Max Forecast", f"{forecast.max():.1f}°C")
        col3.metric("Avg Humidity", f"{df['relative_humidity_2m'].mean():.0f}%")
