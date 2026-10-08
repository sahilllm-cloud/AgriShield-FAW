from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "hybrid_faw_model.pkl"
FEATURE_FILE = ROOT / "data" / "features" / "swin_features_all_4225.csv"


print("\nLoading hybrid model...")
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]

print("Loading Swin features...")
df = pd.read_csv(FEATURE_FILE)

# ------------------------------------------------------------
# DATE
# ------------------------------------------------------------

if "date" in df.columns:
    df["Date"] = pd.to_datetime(
        df["date"].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
elif "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
else:
    raise ValueError("No date column found.")


# ------------------------------------------------------------
# SWIN FEATURES
# ------------------------------------------------------------

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
    c for c in df.columns
    if c not in metadata
    and pd.api.types.is_numeric_dtype(df[c])
]

print(f"Images available: {len(df)}")
print(f"Swin dimensions: {len(feature_columns)}")


# ------------------------------------------------------------
# IMAGE REPRESENTATION
# ------------------------------------------------------------

df["Swin_Mean"] = df[feature_columns].mean(axis=1)
df["Swin_Std"] = df[feature_columns].std(axis=1)
df["Swin_Norm"] = np.linalg.norm(
    df[feature_columns].values,
    axis=1
)

df["Image_Class"] = pd.to_numeric(
    df["label"],
    errors="coerce"
).fillna(0)


# ------------------------------------------------------------
# CREATE TEST ENVIRONMENT
# ------------------------------------------------------------

# We use a fixed environmental condition only to test
# whether the image representation changes the prediction.

test_weather = {
    "Month": 7,
    "Month_sin": np.sin(2 * np.pi * 7 / 12),
    "Month_cos": np.cos(2 * np.pi * 7 / 12),

    "Temperature_C": 25.0,
    "Humidity_%": 70.0,
    "Rainfall_mm": 8.0,
    "Soil_Moisture_%": 55.0,
    "Wind_Speed_kmph": 10.0,

    "Crop_Age_Days": 35,
    "Crop_Stage_Code": 2,

    "Crop_Stage": "Whorl",
    "Maize_Variety": "Local",
}


# ------------------------------------------------------------
# SELECT DIFFERENT IMAGES
# ------------------------------------------------------------

# Select images from different positions so we are not
# repeatedly testing the same image.

indices = np.linspace(
    0,
    len(df) - 1,
    10,
    dtype=int
)

test_rows = []

for idx in indices:

    row = df.iloc[idx]

    sample = {
        "Month": test_weather["Month"],
        "Month_sin": test_weather["Month_sin"],
        "Month_cos": test_weather["Month_cos"],

        "Temperature_C": test_weather["Temperature_C"],
        "Humidity_%": test_weather["Humidity_%"],
        "Rainfall_mm": test_weather["Rainfall_mm"],
        "Soil_Moisture_%": test_weather["Soil_Moisture_%"],
        "Wind_Speed_kmph": test_weather["Wind_Speed_kmph"],

        "Crop_Age_Days": test_weather["Crop_Age_Days"],
        "Crop_Stage_Code": test_weather["Crop_Stage_Code"],

        "Swin_Mean": row["Swin_Mean"],
        "Swin_Std": row["Swin_Std"],
        "Swin_Norm": row["Swin_Norm"],
        "Image_Class": row["Image_Class"],

        "Temperature_Humidity":
            test_weather["Temperature_C"]
            * test_weather["Humidity_%"],

        "Rainfall_SoilMoisture":
            test_weather["Rainfall_mm"]
            * test_weather["Soil_Moisture_%"],

        "Weather_Stress":
            test_weather["Temperature_C"]
            + 0.20 * test_weather["Humidity_%"]
            + 0.30 * test_weather["Rainfall_mm"],

        "Crop_Stage": test_weather["Crop_Stage"],
        "Maize_Variety": test_weather["Maize_Variety"],
    }

    test_rows.append(sample)


test_df = pd.DataFrame(test_rows)


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

predictions = model.predict(test_df)

predictions = np.clip(predictions, 0, 100)


# ------------------------------------------------------------
# DISPLAY
# ------------------------------------------------------------

print("\n==============================================")
print("HYBRID IMAGE TEST")
print("==============================================")

for i, (idx, probability) in enumerate(
    zip(indices, predictions),
    start=1
):

    filename = Path(
        str(df.iloc[idx].get("image_path", idx))
    ).name

    image_class = (
        "INFECTED"
        if df.iloc[idx]["Image_Class"] > 0
        else "HEALTHY"
    )

    if probability < 33:
        risk = "LOW"
    elif probability < 66:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    print(
        f"{i:02d}. "
        f"{filename} | "
        f"{image_class} | "
        f"{probability:.2f}% | "
        f"{risk}"
    )

print("==============================================")