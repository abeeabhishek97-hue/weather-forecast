import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import yaml
import os

params = yaml.safe_load(open("params.yaml"))
lookback = params["model"]["lookback"]
horizon = params["model"]["horizon"]
train_ratio = 0.8

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

def make_windows(data, lookback, horizon):
    X, y = [], []
    for i in range(len(data) - lookback - horizon):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, 0])
    return np.array(X), np.array(y)

for name in ["technopark", "thampanoor"]:
    df = pd.read_csv(f"data/raw/{name}.csv")
    df.dropna(subset=["temperature_2m"], inplace=True)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek

    features = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "hour", "dayofweek"]
    data = df[features].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    pickle.dump(scaler, open(f"models/{name}_scaler.pkl", "wb"))

    X, y = make_windows(data_scaled, lookback, horizon)

    split = int(len(X) * train_ratio)
    np.save(f"data/processed/{name}_X_train.npy", X[:split])
    np.save(f"data/processed/{name}_y_train.npy", y[:split])
    np.save(f"data/processed/{name}_X_test.npy", X[split:])
    np.save(f"data/processed/{name}_y_test.npy", y[split:])
    print(f"{name}: {split} train, {len(X)-split} test windows")
