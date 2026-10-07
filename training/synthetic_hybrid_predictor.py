"""Reusable inference wrapper for the isolated synthetic hybrid artifacts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
import torch

from .synthetic_hybrid_pipeline import (
    FORECAST_FEATURES,
    FUSION_FEATURES,
    INPUT_WINDOW,
    STAGES,
    WEATHER_FEATURES,
    WeatherLSTM,
    build_fusion_frame,
    stage_for_das,
)


ROOT = Path(__file__).resolve().parents[1]
LSTM_PATH = ROOT / "models" / "synthetic_hybrid_lstm.pth"
SCALER_PATH = ROOT / "models" / "synthetic_hybrid_weather_scaler.pkl"
FUSION_PATH = ROOT / "models" / "synthetic_hybrid_lightgbm.pkl"


class SyntheticHybridPredictor:
    """Predict attack probability from seven recent daily observations."""

    def __init__(self, lstm_path: str | Path = LSTM_PATH, scaler_path: str | Path = SCALER_PATH, fusion_path: str | Path = FUSION_PATH):
        checkpoint = torch.load(lstm_path, map_location="cpu", weights_only=False)
        scaler_artifact = joblib.load(scaler_path)
        fusion_artifact = joblib.load(fusion_path)
        self.scaler = scaler_artifact["scaler"]
        self.lstm = WeatherLSTM(len(WEATHER_FEATURES), checkpoint["hidden_size"], checkpoint["layers"], checkpoint["forecast_horizon"])
        self.lstm.load_state_dict(checkpoint["state_dict"])
        self.lstm.eval()
        self.fusion = fusion_artifact["model"]

    def predict(self, recent_observations: pd.DataFrame | list[Mapping[str, Any]], das: int, crop_stage: str | None = None, leaf_damage_probability: float = 0.0) -> dict[str, Any]:
        history = recent_observations.copy() if isinstance(recent_observations, pd.DataFrame) else pd.DataFrame(recent_observations)
        missing = [feature for feature in WEATHER_FEATURES if feature not in history.columns]
        if missing or len(history) < INPUT_WINDOW:
            raise ValueError(f"recent_observations needs at least {INPUT_WINDOW} rows and columns {missing}")
        if int(das) < 0 or not 0 <= float(leaf_damage_probability) <= 1:
            raise ValueError("DAS must be non-negative and leaf damage probability must be between 0 and 1")
        selected_stage = crop_stage or str(stage_for_das(np.asarray([das]))[0])
        if selected_stage not in STAGES:
            raise ValueError(f"crop_stage must be one of {STAGES}")
        recent = history.tail(INPUT_WINDOW)[WEATHER_FEATURES].apply(pd.to_numeric, errors="coerce")
        if recent.isna().any().any():
            raise ValueError("recent weather observations must be numeric")
        scaled = self.scaler.transform(recent)
        with torch.no_grad():
            forecast_scaled = self.lstm(torch.from_numpy(scaled.astype(np.float32))[None, ...]).numpy()[0]
        forecast = self.scaler.inverse_transform(forecast_scaled)
        row = {feature: float(recent.iloc[-1][feature]) for feature in WEATHER_FEATURES}
        row.update({feature: float(forecast[:, index].mean()) for index, feature in enumerate(FORECAST_FEATURES)})
        row.update({"DAS": int(das), "FAW_leaf_damage_probability": float(leaf_damage_probability), "Crop_Stage": selected_stage})
        frame = pd.DataFrame([row])
        features = build_fusion_frame(frame).reindex(columns=FUSION_FEATURES, fill_value=0.0)
        probability = float(np.clip(self.fusion.predict(features)[0], 0.0, 1.0))
        return {"FAW_Attack_Probability": probability, "DAS": int(das), "Crop_Stage": selected_stage, "forecast_features": {feature: row[feature] for feature in FORECAST_FEATURES}, "prediction_date": date.today().isoformat()}


def predict_hybrid(recent_observations: pd.DataFrame | list[Mapping[str, Any]], das: int, crop_stage: str | None = None, leaf_damage_probability: float = 0.0) -> dict[str, Any]:
    return SyntheticHybridPredictor().predict(recent_observations, das, crop_stage, leaf_damage_probability)