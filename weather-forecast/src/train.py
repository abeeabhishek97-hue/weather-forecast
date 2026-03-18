import numpy as np
import json
import yaml
import subprocess
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

params = yaml.safe_load(open("params.yaml"))
p = params["model"]
os.makedirs("models", exist_ok=True)

metrics = {}
version = 1
if os.path.exists("version.json"):
    old = json.load(open("version.json"))
    version = old.get("version", 0) + 1

for name in ["technopark", "thampanoor"]:
    X_train = np.load(f"data/processed/{name}_X_train.npy")
    y_train = np.load(f"data/processed/{name}_y_train.npy")
    X_test  = np.load(f"data/processed/{name}_X_test.npy")
    y_test  = np.load(f"data/processed/{name}_y_test.npy")

    model = keras.Sequential([
        LSTM(p["lstm_units"], input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(p["dropout"]),
        Dense(p["horizon"])
    ])
    model.compile(optimizer=keras.optimizers.Adam(p["learning_rate"]), loss="mse")
    model.fit(
        X_train, y_train,
        epochs=p["epochs"],
        batch_size=p["batch_size"],
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=p["patience"], restore_best_weights=True)],
        verbose=1
    )

    preds = model.predict(X_test)
    mae  = float(mean_absolute_error(y_test, preds))
    rmse = float(mean_squared_error(y_test, preds) ** 0.5)
    metrics[name] = {"mae": round(mae, 4), "rmse": round(rmse, 4)}
    model.save(f"models/{name}_model.keras")
    print(f"{name} → MAE={mae:.4f} RMSE={rmse:.4f}")

try:
    sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
except Exception:
    sha = "unknown"

json.dump(metrics, open("metrics.json", "w"), indent=2)
json.dump({
    "version": version,
    "trained_on": datetime.utcnow().strftime("%Y-%m-%d"),
    "git_sha": sha,
    "rmse_technopark": metrics["technopark"]["rmse"],
    "rmse_thampanoor": metrics["thampanoor"]["rmse"],
}, open("version.json", "w"), indent=2)

print("Done. metrics.json and version.json saved.")
