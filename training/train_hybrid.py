from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "data" / "features" / "swin_features_all_4225.csv"
WEATHER_FILE = (
    ROOT
    / "data"
    / "faw"
    / "FAW_Maize_Karnataka_Dataset_rebalanced.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "hybrid_faw_model.pkl"


# ============================================================
# LOAD SWIN FEATURES
# ============================================================

print("\nLoading Swin image features...")

image_df = pd.read_csv(FEATURE_FILE)

print(f"Image feature rows: {len(image_df)}")

# Extract date from filename/date column
if "date" in image_df.columns:
    image_df["Date"] = pd.to_datetime(
        image_df["date"].astype(str),
        format="%Y%m%d",
        errors="coerce",
    )
elif "Date" in image_df.columns:
    image_df["Date"] = pd.to_datetime(image_df["Date"], errors="coerce")
else:
    raise ValueError("No date column found in Swin feature file.")


# ============================================================
# FIND SWIN FEATURE COLUMNS
# ============================================================

metadata_columns = {
    "date",
    "Date",
    "label",
    "split",
    "image_path",
    "path",
    "filename",
}

feature_columns = [
    col
    for col in image_df.columns
    if col not in metadata_columns
    and pd.api.types.is_numeric_dtype(image_df[col])
]

if not feature_columns:
    raise ValueError("No numeric Swin feature columns found.")

print(f"Swin feature dimensions: {len(feature_columns)}")


# ============================================================
# LOAD ENVIRONMENT DATA
# ============================================================

print("\nLoading environmental FAW dataset...")

weather_df = pd.read_csv(WEATHER_FILE)

print(f"Environmental rows: {len(weather_df)}")


# ============================================================
# REMOVE OLD PEST-HISTORY FEATURES
# ============================================================

remove_columns = [
    "Previous_Pest_Count",
    "Days_Since_Last_Attack",
]

weather_df = weather_df.drop(
    columns=[c for c in remove_columns if c in weather_df.columns],
    errors="ignore",
)


# ============================================================
# CREATE ENVIRONMENTAL TARGET
# ============================================================

risk_map = {
    "Low": 20.0,
    "Medium": 55.0,
    "High": 85.0,
}

weather_df["FAW_Base_Probability"] = (
    weather_df["Risk_Level"]
    .astype(str)
    .str.strip()
    .map(risk_map)
)

weather_df = weather_df.dropna(
    subset=["FAW_Base_Probability"]
).copy()


# ============================================================
# CROP STAGE ENCODING
# ============================================================

stage_map = {
    "Seedling": 0,
    "Vegetative": 1,
    "Whorl": 2,
    "Tasseling": 3,
    "Silking": 4,
    "Maturity": 5,
}

weather_df["Crop_Stage_Code"] = (
    weather_df["Crop_Stage"]
    .astype(str)
    .str.strip()
    .map(stage_map)
)

weather_df["Crop_Stage_Code"] = weather_df["Crop_Stage_Code"].fillna(-1)


# ============================================================
# BUILD IMAGE-LEVEL SUMMARY
# ============================================================

print("\nPreparing image representation...")

# Swin produces a high-dimensional representation.
# We retain the complete representation but calculate
# compact statistics for the hybrid model.

image_df["Swin_Mean"] = image_df[feature_columns].mean(axis=1)
image_df["Swin_Std"] = image_df[feature_columns].std(axis=1)
image_df["Swin_Norm"] = np.linalg.norm(
    image_df[feature_columns].values,
    axis=1,
)


# ============================================================
# IMAGE CLASS INFORMATION
# ============================================================

if "label" in image_df.columns:
    image_df["Image_Class"] = pd.to_numeric(
        image_df["label"],
        errors="coerce",
    ).fillna(0)
else:
    image_df["Image_Class"] = 0


# ============================================================
# IMPORTANT:
# CURRENT DATASETS DO NOT CONTAIN REAL IMAGE-WEATHER PAIRS.
#
# Therefore we do NOT pretend that the 2020 image dates
# correspond to the Karnataka environmental records.
#
# Instead, environmental examples are used to learn the
# environmental FAW-risk relationship, while the image
# representation provides the visual component.
# ============================================================

print("\nCreating hybrid training examples...")


# Repeat image representations across environmental records.
# To keep the training size manageable, use a controlled
# sample of image features.

MAX_IMAGES = 4225

if len(image_df) > MAX_IMAGES:
    image_sample = image_df.sample(
        MAX_IMAGES,
        random_state=42,
    ).reset_index(drop=True)
else:
    image_sample = image_df.reset_index(drop=True)


# Sample environmental rows to create the prototype
# fusion training set.

MAX_ENV = 2500

if len(weather_df) > MAX_ENV:
    env_sample = weather_df.sample(
        MAX_ENV,
        random_state=42,
    ).reset_index(drop=True)
else:
    env_sample = weather_df.reset_index(drop=True)


# Match rows by deterministic cycling rather than random noise.
image_idx = np.arange(len(env_sample)) % len(image_sample)

hybrid_df = env_sample.copy()

for col in [
    "Swin_Mean",
    "Swin_Std",
    "Swin_Norm",
    "Image_Class",
]:
    hybrid_df[col] = image_sample.iloc[image_idx][col].values


# ============================================================
# IMAGE FEATURE COMPRESSION
# ============================================================

# Instead of inserting thousands of raw Swin dimensions
# into the fusion model, use PCA-like summary statistics
# together with the visual class signal.

# These are deterministic features from the real images.


# ============================================================
# DATE / SEASON INFORMATION
# ============================================================

if "Month" in hybrid_df.columns:
    hybrid_df["Month"] = pd.to_numeric(
        hybrid_df["Month"],
        errors="coerce",
    ).fillna(1)
else:
    hybrid_df["Month"] = 1

hybrid_df["Month_sin"] = np.sin(
    2 * np.pi * hybrid_df["Month"] / 12
)

hybrid_df["Month_cos"] = np.cos(
    2 * np.pi * hybrid_df["Month"] / 12
)


# ============================================================
# WEATHER INTERACTION FEATURES
# ============================================================

hybrid_df["Temperature_Humidity"] = (
    hybrid_df["Temperature_C"]
    * hybrid_df["Humidity_%"]
)

hybrid_df["Rainfall_SoilMoisture"] = (
    hybrid_df["Rainfall_mm"]
    * hybrid_df["Soil_Moisture_%"]
)

hybrid_df["Weather_Stress"] = (
    hybrid_df["Temperature_C"]
    + 0.20 * hybrid_df["Humidity_%"]
    + 0.30 * hybrid_df["Rainfall_mm"]
)


# ============================================================
# TARGET
# ============================================================

# Environmental FAW risk is the available supervision.
#
# The image class acts as a visual adjustment signal.
#
# This is a prototype fusion target, NOT a claim of
# real paired image-weather attack measurements.

image_effect = np.where(
    hybrid_df["Image_Class"] > 0,
    18.0,
    -8.0,
)

hybrid_df["FAW_Attack_Probability"] = (
    hybrid_df["FAW_Base_Probability"]
    + image_effect
    + 0.10 * hybrid_df["Swin_Mean"]
    + 0.05 * hybrid_df["Swin_Std"]
)

hybrid_df["FAW_Attack_Probability"] = (
    hybrid_df["FAW_Attack_Probability"]
    .clip(0, 100)
)


# ============================================================
# FEATURES FOR LIGHTGBM
# ============================================================

numeric_features = [
    "Month",
    "Month_sin",
    "Month_cos",
    "Temperature_C",
    "Humidity_%",
    "Rainfall_mm",
    "Soil_Moisture_%",
    "Wind_Speed_kmph",
    "Crop_Age_Days",
    "Crop_Stage_Code",
    "Swin_Mean",
    "Swin_Std",
    "Swin_Norm",
    "Image_Class",
    "Temperature_Humidity",
    "Rainfall_SoilMoisture",
    "Weather_Stress",
]

numeric_features = [
    c for c in numeric_features
    if c in hybrid_df.columns
]

categorical_features = [
    "Crop_Stage",
    "Maize_Variety",
]

categorical_features = [
    c for c in categorical_features
    if c in hybrid_df.columns
]


X = hybrid_df[numeric_features + categorical_features].copy()
y = hybrid_df["FAW_Attack_Probability"].astype(float)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)


# ============================================================
# LIGHTGBM FUSION MODEL
# ============================================================

model = LGBMRegressor(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="regression",
)


pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining hybrid fusion model...")

pipeline.fit(X_train, y_train)


# ============================================================
# VALIDATION
# ============================================================

pred = pipeline.predict(X_val)

pred = np.clip(pred, 0, 100)

mae = mean_absolute_error(y_val, pred)
rmse = np.sqrt(mean_squared_error(y_val, pred))
r2 = r2_score(y_val, pred)

print("\n========================================")
print("HYBRID MODEL RESULTS")
print("========================================")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
print("========================================")


# ============================================================
# SAVE MODEL
# ============================================================

artifact = {
    "model": pipeline,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "stage_map": stage_map,
    "risk_map": risk_map,
    "feature_columns": feature_columns,
    "image_feature_file": str(FEATURE_FILE),
    "environment_file": str(WEATHER_FILE),
}

joblib.dump(
    artifact,
    MODEL_PATH,
)

print(f"\nHybrid model saved to:")
print(MODEL_PATH)

print("\nDONE.")