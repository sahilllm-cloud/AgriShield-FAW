"""Reusable multivariate LSTM forecaster for historical NASA POWER weather.

Historical weather
        -> 7-day input sequence
        -> multivariate LSTM
        -> 7-day future weather forecast
        -> future-weather features for hybrid_lightgbm.py

This module does not run training on import. The repository currently does
not contain the NASA POWER file referenced by the project documentation, so
the loader fails clearly until a real historical file is supplied.
"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_WINDOW = 7
FORECAST_HORIZON = 7
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
SEED = 42
MODEL_PATH = Path("models/lstm_weather_forecaster.pth")
SCALER_PATH = Path("models/lstm_weather_scaler.pkl")
NASA_POWER_CANDIDATES = [
    Path("data/weather/raw/NASA_POWER_Mysuru_2015_2025.csv"),
    Path("data/weather/raw/nasa_power_mysuru_2015_2025.csv"),
    Path("data/weather/raw/Mysuru_NASA_POWER_2015_2025.csv"),
]
DATE_COLUMNS = ["DATE", "Date", "date", "YYYY-MM-DD", "timestamp", "Timestamp"]
WEATHER_COLUMNS = [
    "T2M", "RH2M", "WS2M", "PRECTOTCORR", "PS",
    "Temperature_C", "Humidity_%", "Wind_Speed_kmph", "Rainfall_mm", "Soil_Moisture_%",
]


def resolve_data_path(data_path: str | Path | None = None) -> Path:
    """Resolve an existing NASA POWER file without falling back to another dataset."""
    if data_path is not None:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Historical weather dataset not found: {path}")
        return path
    for candidate in NASA_POWER_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No NASA POWER weather dataset found. Supply data_path explicitly; "
        f"checked {[str(path) for path in NASA_POWER_CANDIDATES]}"
    )


def _find_date_column(data: pd.DataFrame) -> str:
    for column in DATE_COLUMNS:
        if column in data.columns:
            return column
    for column in data.columns:
        parsed = pd.to_datetime(data[column], errors="coerce")
        if parsed.notna().mean() >= 0.95:
            return column
    raise ValueError("Historical weather data must contain a recognizable chronological date column")


def load_historical_weather(data_path: str | Path | None = None) -> tuple[pd.DataFrame, str, list[str]]:
    """Load, sort, and select only real chronological numeric weather columns."""
    data = pd.read_csv(resolve_data_path(data_path))
    date_column = _find_date_column(data)
    dates = pd.to_datetime(data[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Date column {date_column!r} contains invalid dates")
    selected = [column for column in WEATHER_COLUMNS if column in data.columns]
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("No supported numeric weather columns were found in the historical dataset")
    weather = data[[date_column, *selected]].copy()
    for column in selected:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    weather = weather.dropna(subset=selected).sort_values(date_column).reset_index(drop=True)
    if len(weather) < INPUT_WINDOW + FORECAST_HORIZON + 3:
        raise ValueError("Historical weather dataset is too short for the requested windows and time split")
    return weather, date_column, selected


def make_sequences(values: np.ndarray, input_window: int = INPUT_WINDOW, forecast_horizon: int = FORECAST_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """Create ordered sliding windows without shuffling observations."""
    inputs, targets = [], []
    for start in range(len(values) - input_window - forecast_horizon + 1):
        inputs.append(values[start:start + input_window])
        targets.append(values[start + input_window:start + input_window + forecast_horizon])
    return np.asarray(inputs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


class MultivariateLSTMForecaster(nn.Module):
    """LSTM encoder whose final hidden state predicts a fixed future horizon."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, horizon: int = FORECAST_HORIZON):
        super().__init__()
        self.horizon = horizon
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, horizon * input_size)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(sequence)
        forecast = self.output(hidden[-1])
        return forecast.view(sequence.shape[0], self.horizon, self.input_size)


def _time_split(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(len(values) * 0.70)
    validation_end = int(len(values) * 0.85)
    if train_end < INPUT_WINDOW + FORECAST_HORIZON:
        raise ValueError("Training period is too short for sequence creation")
    return values[:train_end], values[train_end:validation_end], values[validation_end:]


def _scaled_sequences(period: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    scaled = scaler.transform(period)
    return make_sequences(scaled)


def _evaluate(model: nn.Module, inputs: np.ndarray, targets: np.ndarray, scaler: StandardScaler, feature_names: list[str]) -> dict[str, dict[str, float]]:
    if len(inputs) == 0:
        return {}
    model.eval()
    with torch.no_grad():
        predicted = model(torch.from_numpy(inputs)).numpy()
    actual_values = scaler.inverse_transform(targets.reshape(-1, len(feature_names)))
    predicted_values = scaler.inverse_transform(predicted.reshape(-1, len(feature_names)))
    return {
        feature: {
            "mae": float(mean_absolute_error(actual_values[:, index], predicted_values[:, index])),
            "rmse": float(np.sqrt(mean_squared_error(actual_values[:, index], predicted_values[:, index]))),
        }
        for index, feature in enumerate(feature_names)
    }


def train_lstm_forecaster(
    data_path: str | Path | None = None,
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    hidden_size: int = 64,
    num_layers: int = 2,
) -> dict[str, Any]:
    """Train and save a time-ordered forecaster on a real NASA POWER file."""
    torch.manual_seed(SEED)
    weather, date_column, feature_names = load_historical_weather(data_path)
    raw_values = weather[feature_names].to_numpy(dtype=np.float32)
    train_period, validation_period, test_period = _time_split(raw_values)
    scaler = StandardScaler().fit(train_period)
    train_inputs, train_targets = _scaled_sequences(train_period, scaler)
    validation_inputs, validation_targets = _scaled_sequences(validation_period, scaler)
    test_inputs, test_targets = _scaled_sequences(test_period, scaler)
    model = MultivariateLSTMForecaster(len(feature_names), hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_targets)), batch_size=batch_size, shuffle=False)
    for _ in range(epochs):
        model.train()
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            loss_function(model(batch_inputs), batch_targets).backward()
            optimizer.step()
    metrics = {
        "validation": _evaluate(model, validation_inputs, validation_targets, scaler, feature_names),
        "test": _evaluate(model, test_inputs, test_targets, scaler, feature_names),
    }
    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "feature_names": feature_names,
        "input_window": INPUT_WINDOW, "forecast_horizon": FORECAST_HORIZON,
        "hidden_size": hidden_size, "num_layers": num_layers,
        "date_column": date_column, "metrics": metrics,
    }, model_output)
    scaler_output = Path(scaler_path)
    scaler_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_names": feature_names}, scaler_output)
    return {"feature_names": feature_names, "date_column": date_column, "metrics": metrics, "model_path": str(model_output), "scaler_path": str(scaler_output)}


def forecast_weather(
    history: pd.DataFrame | Sequence[Mapping[str, Any]],
    model_path: str | Path = MODEL_PATH,
    scaler_path: str | Path = SCALER_PATH,
) -> dict[str, list[dict[str, float | int]]]:
    """Forecast seven days from the most recent seven rows of real history."""
    checkpoint = torch.load(model_path, map_location="cpu")
    scaler_artifact = joblib.load(scaler_path)
    feature_names = list(checkpoint["feature_names"])
    scaler = scaler_artifact["scaler"]
    history_frame = history.copy() if isinstance(history, pd.DataFrame) else pd.DataFrame(history)
    missing = [feature for feature in feature_names if feature not in history_frame.columns]
    if missing:
        raise ValueError(f"history is missing forecast features: {missing}")
    recent = history_frame[feature_names].tail(INPUT_WINDOW).apply(pd.to_numeric, errors="coerce")
    if len(recent) != INPUT_WINDOW or recent.isna().any().any():
        raise ValueError(f"history must contain {INPUT_WINDOW} recent rows with finite weather values")
    model = MultivariateLSTMForecaster(len(feature_names), checkpoint["hidden_size"], checkpoint["num_layers"], checkpoint["forecast_horizon"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    scaled_history = scaler.transform(recent.to_numpy(dtype=np.float32))
    with torch.no_grad():
        scaled_forecast = model(torch.from_numpy(scaled_history[None, ...])).numpy()[0]
    forecast_values = scaler.inverse_transform(scaled_forecast)
    return {"forecast": [{"day": day + 1, **{feature: float(forecast_values[day, index]) for index, feature in enumerate(feature_names)}} for day in range(len(forecast_values))]}
