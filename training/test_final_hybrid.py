from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

HYBRID_MODEL = ROOT / "models" / "hybrid_faw_model.pkl"
LSTM_MODEL = ROOT / "models" / "lstm_weather_forecaster.keras"
LSTM_SCALER = ROOT / "models" / "lstm_weather_scaler.pkl"

IMAGE_FEATURES = (
    ROOT
    / "data"
    / "features"
    / "swin_features_all_4225.csv"
)

WEATHER_DATA = (
    ROOT
    / "data"
    / "faw"
    / "FAW_Maize_Karnataka_Dataset_rebalanced.csv"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

hybrid_artifact = joblib.load(HYBRID_MODEL)
hybrid_model = hybrid_artifact["model"]

lstm_model = load_model(LSTM_MODEL)

lstm_artifact = joblib.load(LSTM_SCALER)

scaler = lstm_artifact["scaler"]
weather_features = lstm_artifact["weather_features"]
sequence_length = lstm_artifact["sequence_length"]


# ============================================================
# LOAD IMAGE FEATURES
# ============================================================

image_df = pd.read_csv(IMAGE_FEATURES)

metadata = {
    "date",
    "Date",
    "label",
    "split",
    "image_path",
    "path",
    "filename",
}

feature_columns = [
    c
    for c in image_df.columns
    if c not in metadata
    and pd.api.types.is_numeric_dtype(image_df[c])
]

image_df["Swin_Mean"] = image_df[feature_columns].mean(axis=1)
image_df["Swin_Std"] = image_df[feature_columns].std(axis=1)

image_df["Swin_Norm"] = np.linalg.norm(
    image_df[feature_columns].values,
    axis=1
)

image_df["Image_Class"] = pd.to_numeric(
    image_df["label"],
    errors="coerce"
).fillna(0)


# ============================================================
# LOAD WEATHER HISTORY FOR LSTM
# ============================================================

weather_df = pd.read_csv(WEATHER_DATA)

weather_df = weather_df.dropna(
    subset=weather_features
).reset_index(drop=True)

if "Year" in weather_df.columns and "Month" in weather_df.columns:
    weather_df = weather_df.sort_values(
        ["Year", "Month"]
    ).reset_index(drop=True)


# ============================================================
# LSTM FUTURE WEATHER
# ============================================================

recent_weather = (
    weather_df[weather_features]
    .astype(float)
    .tail(sequence_length)
    .values
)

scaled_recent = scaler.transform(
    recent_weather
)

lstm_input = np.expand_dims(
    scaled_recent,
    axis=0
)

future_scaled = lstm_model.predict(
    lstm_input,
    verbose=0
)

future_weather = scaler.inverse_transform(
    future_scaled
)[0]

future_weather = dict(
    zip(
        weather_features,
        future_weather
    )
)


print("\nLSTM future weather:")

for key, value in future_weather.items():
    print(f"{key}: {value:.2f}")


# ============================================================
# STAGE MAP
# ============================================================

stage_map = {
    "Seedling": 0,
    "Vegetative": 1,
    "Whorl": 2,
    "Tasseling": 3,
    "Silking": 4,
    "Maturity": 5,
}


# ============================================================
# FINAL PREDICTION FUNCTION
# ============================================================

def predict_for_image(
    image_index,
    selected_date,
    crop_stage="Whorl",
    crop_age_days=35,
):

    image = image_df.iloc[image_index]

    date = pd.Timestamp(selected_date)

    month = date.month

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    temperature = future_weather["Temperature_C"]
    humidity = future_weather["Humidity_%"]
    rainfall = future_weather["Rainfall_mm"]
    soil_moisture = future_weather["Soil_Moisture_%"]
    wind = future_weather["Wind_Speed_kmph"]

    stage_code = stage_map.get(
        crop_stage,
        2
    )

    sample = pd.DataFrame(
        [
            {
                "Month": month,
                "Month_sin": month_sin,
                "Month_cos": month_cos,

                "Temperature_C": temperature,
                "Humidity_%": humidity,
                "Rainfall_mm": rainfall,
                "Soil_Moisture_%": soil_moisture,
                "Wind_Speed_kmph": wind,

                "Crop_Age_Days": crop_age_days,
                "Crop_Stage_Code": stage_code,

                "Swin_Mean": image["Swin_Mean"],
                "Swin_Std": image["Swin_Std"],
                "Swin_Norm": image["Swin_Norm"],
                "Image_Class": image["Image_Class"],

                "Temperature_Humidity":
                    temperature * humidity,

                "Rainfall_SoilMoisture":
                    rainfall * soil_moisture,

                "Weather_Stress":
                    temperature
                    + 0.20 * humidity
                    + 0.30 * rainfall,

                "Crop_Stage": crop_stage,
                "Maize_Variety": "Local",
            }
        ]
    )

    probability = hybrid_model.predict(
        sample
    )[0]

    probability = float(
        np.clip(probability, 0, 100)
    )

    if probability < 33:
        risk = "LOW"
    elif probability < 66:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    return probability, risk


# ============================================================
# TEST DIFFERENT IMAGES AND DATES
# ============================================================

test_cases = [
    (0, "2026-10-08"),
    (100, "2026-10-08"),
    (500, "2026-10-08"),
    (1000, "2026-11-08"),
    (2000, "2026-12-08"),
]


print("\n")
print("=" * 65)
print("FINAL HYBRID MODEL TEST")
print("=" * 65)

for image_index, selected_date in test_cases:

    probability, risk = predict_for_image(
        image_index=image_index,
        selected_date=selected_date,
        crop_stage="Whorl",
        crop_age_days=35,
    )

    filename = Path(
        str(
            image_df.iloc[image_index].get(
                "image_path",
                image_index
            )
        )
    ).name

    print(
        f"{filename} | "
        f"Date: {selected_date} | "
        f"FAW Probability: {probability:.2f}% | "
        f"Risk: {risk}"
    )

print("=" * 65)
print("\nFINAL HYBRID TEST COMPLETE.")