import requests
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os

params = yaml.safe_load(open("params.yaml"))
locations = params["data"]["locations"]
months = params["data"]["history_months"]

os.makedirs("data/raw", exist_ok=True)

end = datetime.utcnow().date()
start = end - timedelta(days=30 * months)

VARIABLES = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"

for name, coords in locations.items():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params_req = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": str(start),
        "end_date": str(end),
        "hourly": VARIABLES,
        "timezone": "Asia/Kolkata",
    }

    resp = requests.get(url, params=params_req)
    resp.raise_for_status()
    data = resp.json()["hourly"]

    df = pd.DataFrame(data)
    df.rename(columns={"time": "datetime"}, inplace=True)

    out_path = f"data/raw/{name}.csv"
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        df = pd.concat([existing, df]).drop_duplicates("datetime").reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
