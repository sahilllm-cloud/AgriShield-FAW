"""Reproducible synthetic training pipeline for the AgriShield hybrid model.

The pipeline creates field-wise daily weather and crop observations, trains a
7-day-input/7-day-output multivariate LSTM on training fields, and trains a
grouped LightGBM regressor using current observations plus LSTM-only forecast
features. It is intentionally isolated from the repository's older training
scripts and model artifacts.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_hybrid" / "synthetic_maize_faw_50000.csv"
LSTM_PATH = ROOT / "models" / "synthetic_hybrid_lstm.pth"
SCALER_PATH = ROOT / "models" / "synthetic_hybrid_weather_scaler.pkl"
FUSION_PATH = ROOT / "models" / "synthetic_hybrid_lightgbm.pkl"
METADATA_PATH = ROOT / "models" / "synthetic_hybrid_metadata.json"
EVALUATION_PATH = ROOT / "evaluation" / "synthetic_hybrid_metrics.json"
EXAMPLES_PATH = ROOT / "evaluation" / "synthetic_hybrid_example_predictions.csv"

SEED = 42
INPUT_WINDOW = 7
FORECAST_HORIZON = 7
WEATHER_FEATURES = [
    "Temperature_C",
    "Humidity_pct",
    "Rainfall_mm",
    "Soil_Moisture_pct",
    "Wind_Speed_kmph",
]
STAGES = ["Seedling", "Vegetative", "Whorl", "Tasseling", "Silking", "Maturity"]
FORECAST_FEATURES = [f"forecast_{feature}_mean_7d" for feature in WEATHER_FEATURES]
CURRENT_FEATURES = [*WEATHER_FEATURES, "DAS", "FAW_leaf_damage_probability"]
FUSION_FEATURES = [*CURRENT_FEATURES, *FORECAST_FEATURES, *[f"Crop_Stage_{stage}" for stage in STAGES]]


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = SEED
    field_count: int = 500
    days_per_field: int = 100
    input_window: int = INPUT_WINDOW
    forecast_horizon: int = FORECAST_HORIZON
    lstm_hidden_size: int = 48
    lstm_layers: int = 2
    lstm_epochs: int = 5
    lstm_batch_size: int = 256
    lstm_learning_rate: float = 0.001
    test_size: float = 0.2


class WeatherLSTM(nn.Module):
    """Encode seven weather days and decode a fixed seven-day horizon."""

    def __init__(self, input_size: int, hidden_size: int, layers: int, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, horizon * input_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(inputs)
        return self.head(hidden[-1]).view(inputs.shape[0], self.horizon, self.input_size)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stage_for_das(das: np.ndarray) -> np.ndarray:
    return np.select(
        [das <= 14, das <= 35, das <= 50, das <= 65, das <= 80],
        ["Seedling", "Vegetative", "Whorl", "Tasseling", "Silking"],
        default="Maturity",
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30, 30)))


def generate_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Generate correlated field-wise daily observations and continuous targets."""
    rng = np.random.default_rng(config.seed)
    rows: list[pd.DataFrame] = []
    base_date = pd.Timestamp("2024-01-01")
    for field_number in range(config.field_count):
        field_id = f"FIELD_{field_number + 1:04d}"
        field_days = np.arange(config.days_per_field)
        dates = base_date + pd.to_timedelta(field_days, unit="D")
        sowing_offset = int(rng.integers(0, 35))
        sowing_date = dates[0] - pd.Timedelta(days=sowing_offset)
        phase = float(rng.uniform(0, 2 * np.pi))
        climate = rng.normal(0, 1)
        field_risk = rng.normal(0, 0.28)
        annual = np.sin((field_days + 15) / 365 * 2 * np.pi + phase)
        daily = np.sin(field_days / 9 + phase) + rng.normal(0, 0.22, config.days_per_field)
        temperature = 26.0 + 4.5 * annual + 1.3 * climate + 0.8 * daily + rng.normal(0, 0.65, config.days_per_field)
        humidity = np.clip(72.0 - 5.2 * annual - 1.5 * climate - 1.0 * daily + rng.normal(0, 2.2, config.days_per_field), 38, 98)
        rainfall = np.maximum(0, rng.gamma(1.6, 2.1, config.days_per_field) + 1.8 * np.maximum(annual, 0) + 0.35 * (humidity - 70))
        soil_moisture = np.clip(42 + 0.72 * rainfall + 0.45 * (humidity - 65) + rng.normal(0, 3.0, config.days_per_field), 8, 95)
        wind = np.clip(8.0 + 0.9 * climate + 1.1 * np.cos(field_days / 11 + phase) + rng.normal(0, 1.1, config.days_per_field), 0.5, 25)
        das = field_days + sowing_offset
        stages = stage_for_das(das)
        stage_effect = np.select(
            [stages == "Seedling", stages == "Vegetative", stages == "Whorl", stages == "Tasseling", stages == "Silking"],
            [-0.35, 0.20, 0.65, 0.55, 0.25],
            default=-0.10,
        )
        leaf_damage = _sigmoid(
            -2.0 + 0.045 * (temperature - 24) + 0.035 * (humidity - 60) + 0.018 * rainfall + stage_effect + field_risk
        )
        # Target-generation coefficients are heuristic synthetic-data parameters,
        # not literature-derived biological weights. The target includes the next
        # seven latent weather days, while the fusion model receives only the LSTM
        # forecast for those days.
        future_humidity = pd.Series(humidity).rolling(FORECAST_HORIZON, min_periods=1).mean().shift(-FORECAST_HORIZON + 1).to_numpy()
        future_rain = pd.Series(rainfall).rolling(FORECAST_HORIZON, min_periods=1).mean().shift(-FORECAST_HORIZON + 1).to_numpy()
        future_temperature = pd.Series(temperature).rolling(FORECAST_HORIZON, min_periods=1).mean().shift(-FORECAST_HORIZON + 1).to_numpy()
        future_soil_moisture = pd.Series(soil_moisture).rolling(FORECAST_HORIZON, min_periods=1).mean().shift(-FORECAST_HORIZON + 1).to_numpy()
        future_wind = pd.Series(wind).rolling(FORECAST_HORIZON, min_periods=1).mean().shift(-FORECAST_HORIZON + 1).to_numpy()
        future_humidity = np.nan_to_num(future_humidity, nan=humidity[-1])
        future_rain = np.nan_to_num(future_rain, nan=rainfall[-1])
        future_temperature = np.nan_to_num(future_temperature, nan=temperature[-1])
        future_soil_moisture = np.nan_to_num(future_soil_moisture, nan=soil_moisture[-1])
        future_wind = np.nan_to_num(future_wind, nan=wind[-1])
        current_weather_suitability = (
            0.25 * np.exp(-((temperature - 28) / 7) ** 2)
            + 0.25 * np.exp(-((humidity - 72) / 19) ** 2)
            + 0.20 * (1 - np.exp(-rainfall / 9))
            + 0.20 * np.exp(-((soil_moisture - 55) / 28) ** 2)
            + 0.10 * np.exp(-((wind - 7) / 8) ** 2)
        )
        future_weather_suitability = (
            0.25 * np.exp(-((future_temperature - 28) / 7) ** 2)
            + 0.25 * np.exp(-((future_humidity - 72) / 19) ** 2)
            + 0.20 * (1 - np.exp(-future_rain / 9))
            + 0.20 * np.exp(-((future_soil_moisture - 55) / 28) ** 2)
            + 0.10 * np.exp(-((future_wind - 7) / 8) ** 2)
        )
        stage_suitability = np.clip(0.50 + stage_effect / 2.0 + 0.12 * np.exp(-((das - 45) / 35) ** 2), 0, 1)
        latent_risk = (
            0.18 * leaf_damage
            + 0.27 * current_weather_suitability
            + 0.24 * stage_suitability
            + 0.31 * future_weather_suitability
            + 0.10 * field_risk
            + rng.normal(0, 0.055, config.days_per_field)
        )
        rows.append(pd.DataFrame({
            "Date": dates,
            "Field_ID": field_id,
            "Sowing_Date": sowing_date,
            "DAS": das,
            "Crop_Stage": stages,
            "Temperature_C": temperature,
            "Humidity_pct": humidity,
            "Rainfall_mm": rainfall,
            "Soil_Moisture_pct": soil_moisture,
            "Wind_Speed_kmph": wind,
            "FAW_leaf_damage_probability": leaf_damage,
            "_FAW_latent_risk": latent_risk,
        }))
    result = pd.concat(rows, ignore_index=True)
    ranks = result["_FAW_latent_risk"].rank(method="first", pct=True).to_numpy()
    calibrated_target = np.interp(
        ranks,
        [0.0, 0.40, 0.75, 1.0],
        [0.02, 0.44, 0.69, 0.98],
    )
    result["FAW_Attack_Probability"] = np.clip(
        calibrated_target + rng.normal(0, 0.018, len(result)), 0, 1
    )
    return result.drop(columns=["_FAW_latent_risk"])


def make_sequences(data: pd.DataFrame, scaler: StandardScaler, fields: set[str] | list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]]]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    keys: list[tuple[str, int]] = []
    for field_id in sorted(fields):
        field = data[data["Field_ID"] == field_id].sort_values("Date")
        values = scaler.transform(field[WEATHER_FEATURES]).astype(np.float32)
        for end in range(INPUT_WINDOW, len(field) - FORECAST_HORIZON + 1):
            inputs.append(values[end - INPUT_WINDOW:end])
            targets.append(values[end:end + FORECAST_HORIZON])
            keys.append((field_id, end))
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32), keys


def train_lstm(data: pd.DataFrame, train_fields: set[str], config: PipelineConfig) -> tuple[WeatherLSTM, StandardScaler, dict[str, float]]:
    train_rows = data[data["Field_ID"].isin(train_fields)]
    scaler = StandardScaler().fit(train_rows[WEATHER_FEATURES])
    inputs, targets, _ = make_sequences(data, scaler, train_fields)
    model = WeatherLSTM(len(WEATHER_FEATURES), config.lstm_hidden_size, config.lstm_layers, config.forecast_horizon)
    loader = DataLoader(TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets)), batch_size=config.lstm_batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lstm_learning_rate)
    loss_function = nn.MSELoss()
    for _ in range(config.lstm_epochs):
        model.train()
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            loss_function(model(batch_inputs), batch_targets).backward()
            optimizer.step()
    return model, scaler, {"training_sequences": float(len(inputs))}


def add_lstm_forecast_features(data: pd.DataFrame, model: WeatherLSTM, scaler: StandardScaler, fields: set[str]) -> pd.DataFrame:
    result = data.copy()
    for feature in FORECAST_FEATURES:
        result[feature] = np.nan
    model.eval()
    with torch.no_grad():
        for field_id in sorted(fields):
            field_indices = result.index[result["Field_ID"] == field_id].tolist()
            field = result.loc[field_indices].sort_values("Date")
            values = scaler.transform(field[WEATHER_FEATURES]).astype(np.float32)
            if len(field) <= INPUT_WINDOW - 1:
                continue
            sequences = np.asarray([values[end - INPUT_WINDOW:end] for end in range(INPUT_WINDOW, len(field))], dtype=np.float32)
            predictions = model(torch.from_numpy(sequences)).numpy()
            forecast_means = scaler.inverse_transform(predictions.reshape(-1, len(WEATHER_FEATURES))).reshape(-1, FORECAST_HORIZON, len(WEATHER_FEATURES)).mean(axis=1)
            target_indices = field.index[INPUT_WINDOW:]
            for column_index, feature in enumerate(FORECAST_FEATURES):
                result.loc[target_indices, feature] = forecast_means[:, column_index]
    return result.dropna(subset=FORECAST_FEATURES).reset_index(drop=True)


def build_fusion_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[[*CURRENT_FEATURES, *FORECAST_FEATURES, "Crop_Stage"]].copy()
    frame = pd.get_dummies(frame, columns=["Crop_Stage"], dtype=float)
    for stage in STAGES:
        column = f"Crop_Stage_{stage}"
        if column not in frame:
            frame[column] = 0.0
    return frame[FUSION_FEATURES]


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def run_pipeline(config: PipelineConfig = PipelineConfig()) -> dict[str, Any]:
    seed_everything(config.seed)
    data = generate_dataset(config)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(DATA_PATH, index=False)
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.seed)
    train_indices, test_indices = next(splitter.split(data, data["FAW_Attack_Probability"], data["Field_ID"]))
    train_fields = set(data.iloc[train_indices]["Field_ID"])
    test_fields = set(data.iloc[test_indices]["Field_ID"])
    model, scaler, lstm_info = train_lstm(data, train_fields, config)
    enriched = add_lstm_forecast_features(data, model, scaler, train_fields | test_fields)
    train_enriched = enriched[enriched["Field_ID"].isin(train_fields)]
    test_enriched = enriched[enriched["Field_ID"].isin(test_fields)]
    x_train = build_fusion_frame(train_enriched)
    x_test = build_fusion_frame(test_enriched)
    y_train = train_enriched["FAW_Attack_Probability"]
    y_test = test_enriched["FAW_Attack_Probability"]
    fusion = LGBMRegressor(n_estimators=350, learning_rate=0.035, num_leaves=63, max_depth=-1, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=config.seed, verbosity=-1, n_jobs=-1)
    fusion.fit(x_train, y_train)
    predictions = np.clip(fusion.predict(x_test), 0.0, 1.0)
    metrics = regression_metrics(y_test, predictions)
    lstm_inputs, lstm_targets, _ = make_sequences(data, scaler, test_fields)
    with torch.no_grad():
        lstm_predictions = model(torch.from_numpy(lstm_inputs)).numpy()
    actual_weather = scaler.inverse_transform(lstm_targets.reshape(-1, len(WEATHER_FEATURES)))
    predicted_weather = scaler.inverse_transform(lstm_predictions.reshape(-1, len(WEATHER_FEATURES)))
    lstm_metrics = {"mae": float(mean_absolute_error(actual_weather, predicted_weather)), "rmse": float(np.sqrt(mean_squared_error(actual_weather, predicted_weather)))}
    torch.save({"state_dict": model.state_dict(), "weather_features": WEATHER_FEATURES, "input_window": INPUT_WINDOW, "forecast_horizon": FORECAST_HORIZON, "hidden_size": config.lstm_hidden_size, "layers": config.lstm_layers, "seed": config.seed}, LSTM_PATH)
    joblib.dump({"scaler": scaler, "weather_features": WEATHER_FEATURES}, SCALER_PATH)
    joblib.dump({"model": fusion, "feature_names": FUSION_FEATURES, "stages": STAGES, "target": "FAW_Attack_Probability", "seed": config.seed}, FUSION_PATH)
    example_indices = np.linspace(0, len(test_enriched) - 1, 5, dtype=int)
    examples = test_enriched.iloc[example_indices][["Field_ID", "Date", "DAS", "Crop_Stage", "FAW_Attack_Probability"]].copy()
    examples["predicted_FAW_Attack_Probability"] = np.clip(fusion.predict(build_fusion_frame(test_enriched.iloc[example_indices])), 0.0, 1.0)
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(EXAMPLES_PATH, index=False)
    metadata = {"config": asdict(config), "dataset_path": str(DATA_PATH), "row_count": int(len(data)), "field_count": int(data["Field_ID"].nunique()), "train_fields": len(train_fields), "test_fields": len(test_fields), "weather_features": WEATHER_FEATURES, "fusion_features": FUSION_FEATURES, "forecast_source": "LSTM predictions only; actual future weather is excluded from fusion features", "artifacts": {"lstm": str(LSTM_PATH), "scaler": str(SCALER_PATH), "fusion": str(FUSION_PATH)}}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {"dataset": {"rows": len(data), "fields": len(train_fields | test_fields), "train_rows_after_forecast": len(train_enriched), "test_rows_after_forecast": len(test_enriched)}, "lstm": {**lstm_info, "test_weather_metrics": lstm_metrics}, "fusion": metrics, "examples_path": str(EXAMPLES_PATH)}
    EVALUATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("\nExample predictions:")
    print(examples.to_string(index=False))
    return report


if __name__ == "__main__":
    run_pipeline()