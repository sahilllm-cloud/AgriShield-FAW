from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "faw"
    / "FAW_Maize_Karnataka_Dataset_rebalanced.csv"
)

MODEL_PATH = ROOT / "models" / "lstm_weather_forecaster.keras"
SCALER_PATH = ROOT / "models" / "lstm_weather_scaler.pkl"


print("\nLoading LSTM model...")

model = load_model(MODEL_PATH)

artifact = joblib.load(SCALER_PATH)

scaler = artifact["scaler"]
weather_features = artifact["weather_features"]
sequence_length = artifact["sequence_length"]


print("Loading weather data...")

df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=weather_features
).reset_index(drop=True)


if "Year" in df.columns and "Month" in df.columns:
    df = df.sort_values(
        ["Year", "Month"]
    ).reset_index(drop=True)


# ------------------------------------------------------------
# TAKE LAST 7 WEATHER OBSERVATIONS
# ------------------------------------------------------------

recent_weather = (
    df[weather_features]
    .astype(float)
    .tail(sequence_length)
    .values
)


scaled_recent = scaler.transform(
    recent_weather
)


X = np.expand_dims(
    scaled_recent,
    axis=0
)


# ------------------------------------------------------------
# PREDICT NEXT WEATHER STATE
# ------------------------------------------------------------

prediction_scaled = model.predict(
    X,
    verbose=0
)


prediction = scaler.inverse_transform(
    prediction_scaled
)[0]


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n========================================")
print("LSTM FUTURE WEATHER FORECAST")
print("========================================")

for feature, value in zip(
    weather_features,
    prediction
):
    print(
        f"{feature}: {value:.2f}"
    )

print("========================================")
print("\nLSTM forecast generated successfully.")