"""Backend for the PJM hourly energy-consumption forecasting deployment."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "PJM_selected_deployment_model.joblib"
HISTORY_PATH = BASE_DIR / "PJMW_MW_cleaned_hourly_history.csv"
METADATA_PATH = BASE_DIR / "PJM_deployment_metadata.json"
FEATURES_PATH = BASE_DIR / "PJM_tree_feature_columns.json"

FEATURES = [
    "hour", "day_of_week", "day_of_year", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "year_sin", "year_cos",
    "lag_1", "lag_24", "lag_48", "lag_168",
    "rolling_mean_24", "rolling_std_24", "rolling_mean_168"
]

def load_metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))

def load_history():
    if not HISTORY_PATH.exists():
        raise FileNotFoundError(f"Missing {HISTORY_PATH.name}; keep it beside app.py.")
    frame = pd.read_csv(HISTORY_PATH, parse_dates=["Datetime"])
    if not {"Datetime", "PJMW_MW"}.issubset(frame.columns):
        raise ValueError("History CSV must contain Datetime and PJMW_MW columns.")
    frame = frame.dropna(subset=["Datetime", "PJMW_MW"]).sort_values("Datetime")
    series = (frame.drop_duplicates("Datetime", keep="last")
              .set_index("Datetime")["PJMW_MW"])
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    series = series.resample("h").mean().interpolate(
        method="time", limit=2, limit_area="inside"
    ).dropna()
    if len(series) < 168:
        raise ValueError("History must contain at least 168 hourly observations.")
    return series

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {MODEL_PATH.name}; keep it beside app.py.")
    return joblib.load(MODEL_PATH)

def make_feature_row(timestamp, history):
    history = history.loc[history.index < timestamp]
    if len(history) < 168:
        raise ValueError("At least 168 prior hourly observations are required.")
    hour, dow, doy = timestamp.hour, timestamp.dayofweek, timestamp.dayofyear
    return {
        "hour": hour, "day_of_week": dow, "day_of_year": doy,
        "month": timestamp.month, "is_weekend": int(dow >= 5),
        "hour_sin": np.sin(2*np.pi*hour/24), "hour_cos": np.cos(2*np.pi*hour/24),
        "dow_sin": np.sin(2*np.pi*dow/7), "dow_cos": np.cos(2*np.pi*dow/7),
        "year_sin": np.sin(2*np.pi*doy/365.25), "year_cos": np.cos(2*np.pi*doy/365.25),
        "lag_1": history.iloc[-1], "lag_24": history.iloc[-24],
        "lag_48": history.iloc[-48], "lag_168": history.iloc[-168],
        "rolling_mean_24": history.iloc[-24:].mean(),
        "rolling_std_24": history.iloc[-24:].std(ddof=0),
        "rolling_mean_168": history.iloc[-168:].mean(),
    }

def forecast_hours(model, history, hours=720):
    """Generate recursive hourly predictions with the notebook's feature order."""
    hours = int(hours)
    if not 1 <= hours <= 720:
        raise ValueError("Forecast horizon must be between 1 and 720 hours (30 days).")
    rolling_history = history.copy().sort_index()
    future_index = pd.date_range(
        rolling_history.index[-1] + pd.Timedelta(hours=1), periods=hours, freq="h"
    )
    predictions = []
    for timestamp in future_index:
        row = pd.DataFrame([make_feature_row(timestamp, rolling_history)], columns=FEATURES)
        value = float(model.predict(row)[0])
        value = max(0.0, value)
        predictions.append(value)
        rolling_history.loc[timestamp] = value
    return pd.DataFrame({"Datetime": future_index, "Predicted_MW": predictions})

def export_forecast(model, history, hours=720, output_path=None):
    result = forecast_hours(model, history, hours)
    output = Path(output_path) if output_path else BASE_DIR / "PJM_forecast_output.csv"
    result.to_csv(output, index=False)
    return result, output

if __name__ == "__main__":
    model, history = load_model(), load_history()
    forecast, path = export_forecast(model, history, 720)
    print(f"Saved {len(forecast)} hourly predictions to {path}")
    print(f"Forecast period: {forecast.Datetime.min()} to {forecast.Datetime.max()}")
