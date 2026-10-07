"""Hybrid-model preparation using weather and crop-timing inputs only.

The current module deliberately excludes image scores, pest-history columns,
and LSTM outputs because the repository does not yet contain validated inputs
for those signals. It is ready for a future training table with Sowing_Date
and DAS columns, without changing the existing V2/V3 pipelines.
"""

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


DATA_PATH = Path("data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_10000_v3_tracked.csv")
MODEL_PATH = Path("models/faw_hybrid_lightgbm.pkl")
SEED = 42
LABELS = ["Low", "Medium", "High"]
STAGES = ["Seedling", "Vegetative", "Whorl", "Tasseling", "Silking", "Maturity"]
WEATHER_FEATURES = [
    "Temperature_C", "Humidity_%", "Rainfall_mm", "Soil_Moisture_%", "Wind_Speed_kmph"
]
REQUIRED_TRAINING_COLUMNS = [*WEATHER_FEATURES, "Sowing_Date", "DAS", "Crop_Stage", "Risk_Level", "Source_ID"]
FAW_DAMAGE_PROBABILITY = "FAW_damage_probability"
EXPECTED_LSTM_FORECAST_FIELDS = [
    "Temperature_C", "Humidity_%", "Rainfall_mm", "Soil_Moisture_%", "Wind_Speed_kmph"
]


def calculate_das(sowing_date: str | date | datetime, current_date: str | date | datetime) -> int:
    """Return whole days after sowing; negative values are rejected."""
    sowing = pd.Timestamp(sowing_date).normalize()
    current = pd.Timestamp(current_date).normalize()
    das = int((current - sowing).days)
    if das < 0:
        raise ValueError("current_date must not be earlier than sowing_date")
    return das


def suggest_crop_stage(das: int) -> str:
    """Map DAS to the project's six documented crop-stage labels."""
    if int(das) < 0:
        raise ValueError("DAS must be non-negative")
    if das <= 14:
        return "Seedling"
    if das <= 35:
        return "Vegetative"
    if das <= 50:
        return "Whorl"
    if das <= 65:
        return "Tasseling"
    if das <= 80:
        return "Silking"
    return "Maturity"


def _validate_weather(current_weather: Mapping[str, Any]) -> dict[str, float]:
    values = {feature: float(current_weather[feature]) for feature in WEATHER_FEATURES}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("current_weather contains a non-finite value")
    return values


def build_hybrid_features(
    current_weather: Mapping[str, Any],
    crop_stage: str,
    das: int,
    forecast: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build the stable current hybrid schema.

    Current schema: five weather values, DAS, and one-hot Crop_Stage columns.
    The existing LSTM is a one-step classifier and does not reliably produce
    future-weather fields, so forecast is reserved for a future adapter. The
    expected adapter keys are listed in EXPECTED_LSTM_FORECAST_FIELDS.
    """
    if forecast is not None:
        raise ValueError(
            "forecast integration is not enabled: the existing LSTM does not produce reliable future-weather outputs; "
            f"expected future fields are {EXPECTED_LSTM_FORECAST_FIELDS}"
        )
    if crop_stage not in STAGES:
        raise ValueError(f"crop_stage must be one of {STAGES}")
    weather = _validate_weather(current_weather)
    if int(das) < 0:
        raise ValueError("DAS must be non-negative")
    values = {**weather, "DAS": int(das)}
    values.update({f"Crop_Stage_{stage}": float(stage == crop_stage) for stage in STAGES})
    return pd.DataFrame([values], columns=hybrid_feature_names())


def hybrid_feature_names() -> list[str]:
    return [*WEATHER_FEATURES, "DAS", *(f"Crop_Stage_{stage}" for stage in STAGES)]


def _training_features(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        rows.append(build_hybrid_features(row, str(row["Crop_Stage"]), int(row["DAS"])).iloc[0])
    return pd.DataFrame(rows, columns=hybrid_feature_names()).reset_index(drop=True)


def _metrics(y_true: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    high_index = LABELS.index("High")
    precision = precision_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    recall = recall_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    f1 = f1_score(y_true, predicted, labels=LABELS, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=LABELS, average="macro")),
        "weighted_f1": float(f1_score(y_true, predicted, labels=LABELS, average="weighted")),
        "high_precision": float(precision[high_index]),
        "high_recall": float(recall[high_index]),
        "high_f1": float(f1[high_index]),
    }


def train_hybrid_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Train on a future schema-compliant table using a grouped holdout.

    The current repository weather CSV is intentionally rejected because it
    has no Sowing_Date or DAS. Supplying fabricated DAS would create leakage
    or an invalid biological feature, so no fallback is provided.
    """
    data = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_TRAINING_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            f"Hybrid training requires {missing}; the current legacy dataset does not provide these fields. "
            "Add measured Sowing_Date/DAS data before training."
        )
    features = _training_features(data)
    target = data["Risk_Level"]
    groups = data["Source_ID"]
    train_indices, test_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(features, target, groups)
    )
    estimator = LGBMClassifier(
        objective="multiclass", n_estimators=300, learning_rate=0.05,
        num_leaves=31, max_depth=-1, min_child_samples=20,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    estimator.fit(features.iloc[train_indices], target.iloc[train_indices])
    metrics = _metrics(target.iloc[test_indices], estimator.predict(features.iloc[test_indices]))
    artifact = {
        "model": estimator,
        "feature_names": hybrid_feature_names(),
        "labels": LABELS,
        "excluded_features": [
            "Previous_Pest_Count", "Days_Since_Last_Attack",
            "FAW_damage_probability", "infected probability", "LSTM forecast fields",
        ],
    }
    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return metrics


def predict_faw_risk(
    current_weather: Mapping[str, Any],
    sowing_date: str | date | datetime,
    crop_stage: str | None = None,
    forecast: Mapping[str, Any] | None = None,
    current_date: str | date | datetime | None = None,
    model_path: str | Path = MODEL_PATH,
) -> dict[str, Any]:
    """Predict risk through a stable interface for a future backend adapter."""
    observation_date = current_date or date.today()
    das = calculate_das(sowing_date, observation_date)
    selected_stage = crop_stage or suggest_crop_stage(das)
    features = build_hybrid_features(current_weather, selected_stage, das, forecast)
    artifact = joblib.load(model_path)
    estimator = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    predicted = str(estimator.predict(features)[0]).upper()
    return {
        "risk": predicted,
        "das": das,
        "crop_stage": selected_stage,
        "features_used": list(features.columns),
    }


if __name__ == "__main__":
    print(json.dumps({
        "feature_schema": hybrid_feature_names(),
        "excluded_features": ["Previous_Pest_Count", "Days_Since_Last_Attack", FAW_DAMAGE_PROBABILITY],
        "lstm_forecast_interface": EXPECTED_LSTM_FORECAST_FIELDS,
        "model_path": str(MODEL_PATH),
    }, indent=2))