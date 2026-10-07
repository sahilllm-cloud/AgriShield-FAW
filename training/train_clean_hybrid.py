from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import torch

from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from synthetic_hybrid_pipeline import (
    generate_dataset,
    add_lstm_forecast_features,
    build_fusion_frame,
    FUSION_FEATURES,
    WEATHER_FEATURES,
    WeatherLSTM,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

POOL_PATH = ROOT / "data" / "swin_train_val_probability_pool.csv"

LSTM_PATH = ROOT / "models" / "synthetic_hybrid_lstm.pth"
SCALER_PATH = ROOT / "models" / "synthetic_hybrid_weather_scaler.pkl"

OUTPUT_MODEL = ROOT / "models" / "synthetic_hybrid_lightgbm_clean.pkl"
OUTPUT_META = ROOT / "models" / "synthetic_hybrid_clean_metadata.json"


# ============================================================
# LOAD SWIN TRAIN + VALIDATION PROBABILITIES
# ============================================================

print("Loading Swin train + validation probability pool...")

pool = pd.read_csv(POOL_PATH)

probabilities = pool["leaf_damage_probability"].to_numpy()

print("Probability pool size:", len(probabilities))


# ============================================================
# GENERATE SYNTHETIC DATASET
# ============================================================

print("\nGenerating synthetic hybrid dataset...")


class Config:
    seed = 42
    field_count = 500
    days_per_field = 100


config = Config()

data = generate_dataset(config)

print("Synthetic rows:", len(data))


# ============================================================
# REPLACE IMAGE FEATURE WITH REAL SWIN DISTRIBUTION
# ============================================================

rng = np.random.default_rng(42)

data["FAW_leaf_damage_probability"] = rng.choice(
    probabilities,
    size=len(data),
    replace=True,
)


# ============================================================
# TRAIN / TEST SPLIT BY FIELD
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        data,
        groups=data["Field_ID"],
    )
)

train_fields = data.iloc[train_idx].copy()
test_fields = data.iloc[test_idx].copy()

print("Training rows:", len(train_fields))
print("Test rows:", len(test_fields))


# ============================================================
# LOAD EXISTING LSTM + SCALER
# ============================================================

print("\nLoading existing LSTM...")

checkpoint = torch.load(
    LSTM_PATH,
    map_location="cpu",
    weights_only=False,
)

scaler_artifact = joblib.load(SCALER_PATH)

scaler = scaler_artifact["scaler"]

lstm = WeatherLSTM(
    len(WEATHER_FEATURES),
    checkpoint["hidden_size"],
    checkpoint["layers"],
    checkpoint["forecast_horizon"],
)

lstm.load_state_dict(
    checkpoint["state_dict"]
)

lstm.eval()

print("LSTM loaded successfully.")


# ============================================================
# GET FIELD LISTS
# ============================================================

train_field_ids = train_fields[
    "Field_ID"
].unique()

test_field_ids = test_fields[
    "Field_ID"
].unique()


# ============================================================
# ADD LSTM FORECAST FEATURES
# ============================================================

print("\nGenerating LSTM forecast features...")

train_fields = add_lstm_forecast_features(
    train_fields,
    lstm,
    scaler,
    train_field_ids,
)

test_fields = add_lstm_forecast_features(
    test_fields,
    lstm,
    scaler,
    test_field_ids,
)


# ============================================================
# BUILD LIGHTGBM FEATURES
# ============================================================

print("\nBuilding LightGBM features...")

train_frame = build_fusion_frame(
    train_fields
)

test_frame = build_fusion_frame(
    test_fields
)

X_train = train_frame.reindex(
    columns=FUSION_FEATURES,
    fill_value=0.0,
)

X_test = test_frame.reindex(
    columns=FUSION_FEATURES,
    fill_value=0.0,
)

y_train = train_fields[
    "FAW_Attack_Probability"
].to_numpy()

y_test = test_fields[
    "FAW_Attack_Probability"
].to_numpy()


# ============================================================
# TRAIN LIGHTGBM
# ============================================================

print("\nTraining clean LightGBM...")

model = LGBMRegressor(
    n_estimators=350,
    learning_rate=0.035,
    num_leaves=63,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    verbosity=-1,
    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# ============================================================
# EVALUATION
# ============================================================

predictions = np.clip(
    model.predict(X_test),
    0.0,
    1.0,
)

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)

r2 = r2_score(
    y_test,
    predictions,
)


print("\n================================")
print("CLEAN HYBRID MODEL RESULTS")
print("================================")

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")


# ============================================================
# SAVE SEPARATE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "features": FUSION_FEATURES,
    },
    OUTPUT_MODEL,
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "model": "LightGBM clean hybrid experiment",
    "image_probability_source": "Swin train + validation only",
    "test_image_probabilities_used_for_training": False,
    "lstm_source": "existing synthetic_hybrid_lstm.pth",
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2),
}

with open(
    OUTPUT_META,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        metadata,
        f,
        indent=2,
    )


print("\nModel saved:")
print(OUTPUT_MODEL)

print("Metadata saved:")
print(OUTPUT_META)