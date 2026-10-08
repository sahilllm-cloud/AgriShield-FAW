from __future__ import annotations

import json
from datetime import date as date_type
from urllib import error, request
from typing import Any

import joblib
import numpy as np
from fastapi import HTTPException
from tensorflow.keras.models import load_model
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

LSTM_MODEL_PATH = (
    MODELS_DIR / "lstm_weather_forecaster.keras"
)

LSTM_SCALER_PATH = (
    MODELS_DIR / "lstm_weather_scaler.pkl"
)


# ============================================================
# LOAD LSTM
# ============================================================

_lstm_model = None
_lstm_artifact = None


def load_lstm():

    global _lstm_model
    global _lstm_artifact

    if _lstm_model is None:

        if not LSTM_MODEL_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="LSTM weather model not found.",
            )

        if not LSTM_SCALER_PATH.exists():
            raise HTTPException(
                status_code=500,
                detail="LSTM weather scaler not found.",
            )

        _lstm_model = load_model(
            LSTM_MODEL_PATH
        )

        _lstm_artifact = joblib.load(
            LSTM_SCALER_PATH
        )

    return _lstm_model, _lstm_artifact


# ============================================================
# FETCH WEATHER FOR SELECTED DATE
# ============================================================

def fetch_weather_sequence(
    date: str,
    latitude: float,
    longitude: float,
) -> dict[str, Any]:

    # --------------------------------------------------------
    # Validate date
    # --------------------------------------------------------

    try:

        selected_date = date_type.fromisoformat(
            date
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail="Invalid date. Use YYYY-MM-DD.",
        ) from exc

    # --------------------------------------------------------
    # Open-Meteo supports different endpoints depending
    # on whether the selected date is historical or future.
    #
    # For dates within the normal forecast window we use
    # /forecast.
    #
    # For older dates we use the archive endpoint.
    # --------------------------------------------------------

    from datetime import date as dt_date

    today = dt_date.today()

    days_from_today = (
        selected_date - today
    ).days

    if days_from_today >= 0 and days_from_today <= 16:

        base_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

    elif selected_date < today:

        base_url = (
            "https://archive-api.open-meteo.com/v1/archive"
        )

    else:

        # Open-Meteo's normal forecast API cannot provide
        # arbitrary far-future daily weather.
        #
        # For the prototype, use the maximum available
        # forecast window and clearly report the limitation.
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected future date is outside "
                "Open-Meteo's forecast range. "
                "Choose a date within the next 16 days."
            ),
        )

    url = (
        f"{base_url}"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={date}"
        f"&end_date={date}"
        "&hourly=temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "wind_speed_10m,"
        "soil_moisture_0_to_1cm"
        "&timezone=auto"
    )

    try:

        with request.urlopen(
            url,
            timeout=15,
        ) as response:

            payload = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

    except error.HTTPError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Open-Meteo rejected the selected "
                f"date ({date}). "
                f"HTTP status: {exc.code}."
            ),
        ) from exc

    except error.URLError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to fetch weather "
                "from Open-Meteo."
            ),
        ) from exc

    hourly = payload.get("hourly")

    if not hourly:

        raise HTTPException(
            status_code=502,
            detail=(
                "Open-Meteo did not return "
                "hourly weather data."
            ),
        )

    # --------------------------------------------------------
    # Convert hourly observations into one daily observation.
    # --------------------------------------------------------

    def daily_mean(key: str) -> float:

        values = [
            value
            for value in hourly.get(key, [])
            if value is not None
        ]

        if not values:
            return 0.0

        return float(
            np.mean(values)
        )

    def daily_sum(key: str) -> float:

        values = [
            value
            for value in hourly.get(key, [])
            if value is not None
        ]

        if not values:
            return 0.0

        return float(
            np.sum(values)
        )

    # --------------------------------------------------------
    # Soil moisture from Open-Meteo is a fraction.
    #
    # Example:
    # 0.28 = 28%
    #
    # The trained LSTM expects percentage-like values,
    # so convert fraction -> percentage.
    # --------------------------------------------------------

    soil_moisture_fraction = daily_mean(
        "soil_moisture_0_to_1cm"
    )

    soil_moisture_percentage = (
        soil_moisture_fraction * 100.0
    )

    return {
        "Temperature_C": daily_mean(
            "temperature_2m"
        ),

        "Humidity_%": daily_mean(
            "relative_humidity_2m"
        ),

        "Rainfall_mm": daily_sum(
            "precipitation"
        ),

        "Soil_Moisture_%": (
            soil_moisture_percentage
        ),

        "Wind_Speed_kmph": daily_mean(
            "wind_speed_10m"
        ),
    }


# ============================================================
# GENERATE LSTM FORECAST
# ============================================================

def generate_lstm_forecast(
    date: str,
    latitude: float = 12.972,
    longitude: float = 77.594,
) -> dict[str, Any]:

    model, artifact = load_lstm()

    scaler = artifact["scaler"]

    weather_features = artifact[
        "weather_features"
    ]

    sequence_length = artifact[
        "sequence_length"
    ]

    # --------------------------------------------------------
    # Get weather for the selected date.
    # --------------------------------------------------------

    selected_weather = fetch_weather_sequence(
        date=date,
        latitude=latitude,
        longitude=longitude,
    )

    # --------------------------------------------------------
    # Build the sequence expected by the trained LSTM.
    #
    # The current prototype has a 7-step LSTM but the
    # available external weather input is one selected date.
    #
    # Therefore the selected-date observation is repeated
    # only as an input adapter.
    # --------------------------------------------------------

    row = np.array(
        [
            selected_weather[feature]
            for feature in weather_features
        ],
        dtype=float,
    )

    sequence = np.tile(
        row,
        (
            sequence_length,
            1,
        ),
    )

    scaled_sequence = scaler.transform(
        sequence
    )

    lstm_input = np.expand_dims(
        scaled_sequence,
        axis=0,
    )

    prediction_scaled = model.predict(
        lstm_input,
        verbose=0,
    )

    prediction = scaler.inverse_transform(
        prediction_scaled
    )[0]

    forecast = {
        feature: float(value)
        for feature, value in zip(
            weather_features,
            prediction,
        )
    }

    return {
        "selected_date": date,
        "latitude": latitude,
        "longitude": longitude,
        "selected_weather": selected_weather,
        "lstm_forecast": forecast,
    }