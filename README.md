# Weather Forecasting MLOps — Technopark & Thampanoor

A Streamlit app that forecasts 24-hour temperature for two Thiruvananthapuram locations using an LSTM model that retrains daily via GitHub Actions + DVC.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Init DVC
dvc init
dvc remote add -d myremote gdrive://<your-folder-id>

# 3. Run the full pipeline
dvc repro

# 4. Run the app
streamlit run app.py
```

## GitHub Secrets needed

| Secret | Purpose |
|--------|---------|
| `GDRIVE_CREDENTIALS_DATA` | Base64-encoded Google service account JSON |
| `GIT_TOKEN` | GitHub personal access token |
