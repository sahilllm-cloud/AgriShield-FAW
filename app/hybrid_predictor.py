from __future__ import annotations

import time

import io

import json

import hashlib

import logging

from pathlib import Path

from typing import Any

from datetime import date as date_type, timedelta

from urllib import error, request

import joblib

import numpy as np

import pandas as pd

import timm

import torch

from PIL import Image, UnidentifiedImageError

from fastapi import HTTPException, UploadFile

from torchvision import transforms

from training.synthetic_hybrid_predictor import (

    SyntheticHybridPredictor,

)

from app.maize_image_validator import (

    MaizeImageValidatorError,

    validate_maize_image,

)

from app.maize_crop_classifier import enforce_maize_crop_identity

# ============================================================

# PATHS

# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

SWIN_MODEL_PATH = MODELS_DIR / "swin_best.pth"

HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_faw_model.pkl"

# ============================================================

# CACHED MODELS

# ============================================================

_swin_model = None

_hybrid_artifact = None

_synthetic_predictor = None

logger = logging.getLogger(__name__)

# ============================================================

# LOAD SWIN

# ============================================================

def load_swin_model():

    global _swin_model

    if _swin_model is None:

        if not SWIN_MODEL_PATH.exists():

            raise HTTPException(

                status_code=500,

                detail="Swin model not found.",

            )

        model = timm.create_model(

            "swin_tiny_patch4_window7_224",

            pretrained=False,

        )

        model.head.fc = torch.nn.Linear(

            model.head.fc.in_features,

            2,

        )

        checkpoint = torch.load(

            SWIN_MODEL_PATH,

            map_location="cpu",

        )

        model.load_state_dict(checkpoint)

        model.eval()

        _swin_model = model

    return _swin_model

# ============================================================

# LOAD HYBRID MODEL

# ============================================================

def load_hybrid_model():

    global _hybrid_artifact

    if _hybrid_artifact is None:

        if not HYBRID_MODEL_PATH.exists():

            raise HTTPException(

                status_code=500,

                detail="Hybrid FAW model not found.",

            )

        _hybrid_artifact = joblib.load(

            HYBRID_MODEL_PATH

        )

    return _hybrid_artifact

def load_synthetic_predictor():

    global _synthetic_predictor

    if _synthetic_predictor is None:

        _synthetic_predictor = SyntheticHybridPredictor()

    return _synthetic_predictor

# Successful weather responses are cached for this backend process.

# This avoids repeated external API calls for the same date and coordinates.

_WEATHER_CACHE: dict[tuple[str, str, float, float], list[dict[str, float]]] = {}

# Successful weather results are cached until the backend restarts.
_WEATHER_CACHE: dict[tuple[str, str, float, float], list[dict[str, float]]] = {}


def fetch_weather_history(
    selected_date: str,
    latitude: float,
    longitude: float,
) -> list[dict[str, float]]:
    """Fetch seven daily weather rows with retries and missing-data handling."""
    from urllib.parse import urlencode

    try:
        end_date = date_type.fromisoformat(str(selected_date))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid date. Use YYYY-MM-DD.") from exc

    today = date_type.today()
    if end_date >= today:
        if (end_date - today).days > 16:
            raise HTTPException(status_code=400, detail="The selected future date is outside Open-Meteo's forecast range.")
        base_url = "https://api.open-meteo.com/v1/forecast"
    else:
        base_url = "https://archive-api.open-meteo.com/v1/archive"

    start_date = end_date - timedelta(days=6)
    cache_key = (start_date.isoformat(), end_date.isoformat(), round(float(latitude), 4), round(float(longitude), 4))
    cached = _WEATHER_CACHE.get(cache_key)
    if cached is not None:
        return [row.copy() for row in cached]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,soil_moisture_0_to_1cm",
        "timezone": "auto",
    }
    url = f"{base_url}?{urlencode(params)}"
    payload = None
    last_error = None

    for attempt in range(3):
        try:
            req = request.Request(url, headers={"User-Agent": "AgriShield-FAW/1.0"})
            with request.urlopen(req, timeout=30) as response:
                candidate = json.loads(response.read().decode("utf-8"))

            hourly = candidate.get("hourly") if isinstance(candidate, dict) else None
            times = hourly.get("time") if isinstance(hourly, dict) else None
            if not isinstance(times, list) or not times:
                raise ValueError("Open-Meteo returned no hourly timestamps.")

            for field in ("temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"):
                values = hourly.get(field)
                if not isinstance(values, list) or len(values) != len(times):
                    raise ValueError(f"Open-Meteo returned incomplete {field} data.")

            # Historical soil-moisture values may be null or absent. Treat this
            # field as optional here; downstream receives a documented fallback.
            soil_values = hourly.get("soil_moisture_0_to_1cm")
            if not isinstance(soil_values, list) or len(soil_values) != len(times):
                hourly["soil_moisture_0_to_1cm"] = [None] * len(times)

            payload = candidate
            break

        except error.HTTPError as exc:
            last_error = exc
            if exc.code not in (408, 429) and exc.code < 500:
                raise HTTPException(status_code=502, detail=f"Open-Meteo rejected the weather request ({exc.code}). Check the date and location.") from exc
        except (error.URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            last_error = exc

        if attempt < 2:
            delay = 2 ** attempt
            logger.warning("Open-Meteo request failed (attempt %d/3); retrying in %d seconds: %s", attempt + 1, delay, last_error)
            time.sleep(delay)

    if payload is None:
        logger.error("Open-Meteo failed after retries: %s", last_error)
        raise HTTPException(status_code=502, detail="The weather service timed out or is temporarily unavailable. Please try again shortly.") from last_error

    hourly = payload["hourly"]
    times = hourly["time"]
    rows: list[dict[str, float]] = []

    for day_index in range(7):
        day = start_date + timedelta(days=day_index)
        indexes = [i for i, timestamp in enumerate(times) if isinstance(timestamp, str) and timestamp.startswith(day.isoformat())]
        if not indexes:
            raise HTTPException(status_code=502, detail=f"Open-Meteo returned no hourly observations for {day.isoformat()}.")

        row: dict[str, float] = {}
        for field, target in (
            ("temperature_2m", "Temperature_C"),
            ("relative_humidity_2m", "Humidity_pct"),
            ("wind_speed_10m", "Wind_Speed_kmph"),
        ):
            series = hourly[field]
            samples = [series[i] for i in indexes if i < len(series) and series[i] is not None]
            if not samples:
                raise HTTPException(status_code=502, detail=f"Weather measurements for {field} are missing on {day.isoformat()}.")
            row[target] = float(np.mean(samples))

        rainfall_series = hourly["precipitation"]
        rainfall = [rainfall_series[i] for i in indexes if i < len(rainfall_series) and rainfall_series[i] is not None]
        if not rainfall:
            raise HTTPException(status_code=502, detail=f"Rainfall measurements are missing on {day.isoformat()}.")
        row["Rainfall_mm"] = float(np.sum(rainfall))

        soil_series = hourly.get("soil_moisture_0_to_1cm", [])
        soil_samples = [soil_series[i] for i in indexes if i < len(soil_series) and soil_series[i] is not None]
        if soil_samples:
            row["Soil_Moisture_pct"] = float(np.mean(soil_samples)) * 100.0
        else:
            # Compatibility fallback, not a measured value. This avoids a crash
            # but can affect model quality; logs explicitly flag the missing data.
            row["Soil_Moisture_pct"] = 0.0
            logger.warning("Soil moisture unavailable for %s; using fallback 0.0 (not observed data).", day.isoformat())

        rows.append(row)

    _WEATHER_CACHE[cache_key] = [row.copy() for row in rows]
    return rows


# ============================================================

# IMAGE FEATURES

# ============================================================

async def extract_image_features(

    file: UploadFile,

) -> dict[str, Any]:

    if not file.filename:

        raise HTTPException(

            status_code=400,

            detail="No image selected.",

        )

    if (

        not file.content_type

        or not file.content_type.startswith("image/")

    ):

        raise HTTPException(

            status_code=400,

            detail="Upload a valid image file.",

        )

    image_bytes = await file.read()

    image_sha256 = hashlib.sha256(image_bytes).hexdigest()

    logger.info(

        "Hybrid upload filename=%s size_bytes=%d sha256=%s",

        file.filename,

        len(image_bytes),

        image_sha256,

    )

    try:

        image = Image.open(

            io.BytesIO(image_bytes)

        ).convert("RGB")

    except UnidentifiedImageError as exc:

        raise HTTPException(

            status_code=400,

            detail="Could not read the uploaded image.",

        ) from exc

    enforce_maize_crop_identity(image, file.filename)

    try:

        maize_validation = validate_maize_image(image)

    except MaizeImageValidatorError as exc:

        logger.exception(

            "Maize image validator failed for filename=%s",

            file.filename,

        )

        raise HTTPException(

            status_code=503,

            detail=(

                "Maize image validation is temporarily unavailable. "

                "Please try again."

            ),

        ) from exc

    if not maize_validation.accepted:

        logger.info(

            "Rejected non-maize image filename=%s validation=%s",

            file.filename,

            maize_validation.as_log_dict(),

        )

        raise HTTPException(

            status_code=422,

            detail="Invalid image. Please upload a clear photo of a maize leaf.",

        )

    transform = transforms.Compose(

        [

            transforms.Resize((224, 224)),

            transforms.ToTensor(),

            transforms.Normalize(

                mean=[

                    0.485,

                    0.456,

                    0.406,

                ],

                std=[

                    0.229,

                    0.224,

                    0.225,

                ],

            ),

        ]

    )

    tensor = transform(

        image

    ).unsqueeze(0)

    model = load_swin_model()

    with torch.no_grad():

        # Get the Swin visual representation.

        features = model.forward_features(

            tensor

        )

        # Pool the representation and remove

        # the classifier head.

        features = model.forward_head(

            features,

            pre_logits=True,

        )

        # IMPORTANT:

        # \`features\` is already pooled.

        # Apply only the final linear classifier.

        logits = model.head.fc(

            features

        )

        probabilities = torch.softmax(

            logits,

            dim=1,

        )[0]

        prediction_index = int(

            torch.argmax(

                probabilities

            ).item()

        )

    feature_vector = (

        features.squeeze(0)

        .cpu()

        .numpy()

    )

    return {

        "Image_Filename": file.filename,

        "Image_Size_Bytes": float(len(image_bytes)),

        "Image_SHA256": image_sha256,

        "Swin_Mean": float(

            np.mean(feature_vector)

        ),

        "Swin_Std": float(

            np.std(feature_vector)

        ),

        "Swin_Norm": float(

            np.linalg.norm(feature_vector)

        ),

        "Image_Class": float(

            prediction_index

        ),

        "Healthy_Probability": float(

            probabilities[0].item()

        ),

        "Infected_Probability": float(

            probabilities[1].item()

        ),

    }

# ============================================================

# CROP STAGE

# ============================================================

STAGE_MAP = {

    "Seedling": 0,

    "Vegetative": 1,

    "Whorl": 2,

    "Tasseling": 3,

    "Silking": 4,

    "Maturity": 5,

}

# ============================================================

# FINAL HYBRID PREDICTION

# ============================================================

async def predict_hybrid(

    file: UploadFile,

    selected_date: str,

    latitude: float = 12.972,

    longitude: float = 77.594,

    crop_stage: str = "Whorl",

    crop_age_days: int = 35,

    maize_variety: str = "Local",

) -> dict[str, Any]:

    # --------------------------------------------------------

    # Validate date

    # --------------------------------------------------------

    try:

        date = pd.Timestamp(

            selected_date

        )

    except Exception as exc:

        raise HTTPException(

            status_code=400,

            detail="Invalid date. Use YYYY-MM-DD.",

        ) from exc

    # --------------------------------------------------------

    # Validate crop stage

    # --------------------------------------------------------

    normalized_stage = str(

        crop_stage

    ).strip()

    if normalized_stage not in STAGE_MAP:

        raise HTTPException(

            status_code=400,

            detail=(

                "Invalid crop stage. Use one of: "

                "Seedling, Vegetative, Whorl, "

                "Tasseling, Silking, Maturity."

            ),

        )

    # --------------------------------------------------------

    # SWIN

    # --------------------------------------------------------

    image_features = await extract_image_features(

        file

    )

    # --------------------------------------------------------

    # WEATHER + SYNTHETIC PYTORCH LSTM + LIGHTGBM

    # --------------------------------------------------------

    weather_history = fetch_weather_history(

        selected_date=selected_date,

        latitude=latitude,

        longitude=longitude,

    )

    current_weather = weather_history[-1]

    leaf_damage_probability = image_features[

        "Infected_Probability"

    ]

    synthetic_result = load_synthetic_predictor().predict(

        recent_observations=weather_history,

        das=crop_age_days,

        crop_stage=normalized_stage,

        leaf_damage_probability=leaf_damage_probability,

    )

    temperature = current_weather[

        "Temperature_C"

    ]

    humidity = current_weather[

        "Humidity_pct"

    ]

    rainfall = current_weather[

        "Rainfall_mm"

    ]

    soil_moisture = current_weather[

        "Soil_Moisture_pct"

    ]

    wind_speed = current_weather[

        "Wind_Speed_kmph"

    ]

    probability = float(

        np.clip(

            synthetic_result["FAW_Attack_Probability"] * 100,

            0,

            100,

        )

    )

    # --------------------------------------------------------

    # RISK LEVEL

    # --------------------------------------------------------

    if probability < 33:

        risk_level = "LOW"

    elif probability < 66:

        risk_level = "MEDIUM"

    else:

        risk_level = "HIGH"

    # --------------------------------------------------------

    # RESULT

    # --------------------------------------------------------

    print(

        "Hybrid prediction "

        f"filename={image_features['Image_Filename']} "

        f"sha256={image_features['Image_SHA256']} "

        f"swin_infected_probability={image_features['Infected_Probability']:.10f} "

        f"faw_leaf_damage_probability={leaf_damage_probability:.10f} "

        f"raw_lightgbm_output={synthetic_result['raw_lightgbm_output']:.10f} "

        f"final_percentage={probability:.10f} "

        f"risk_category={risk_level}",

        flush=True,

    )

    return {

        "prediction": {

            "faw_attack_probability": round(

                probability,

                2,

            ),

            "risk_level": risk_level,

        },

        "image": {

            "filename": file.filename,

            "predicted_class": (

                "infected"

                if image_features[

                    "Image_Class"

                ] == 1

                else "healthy"

            ),

            "healthy_probability": round(

                image_features[

                    "Healthy_Probability"

                ] * 100,

                2,

            ),

            "infected_probability": round(

                image_features[

                    "Infected_Probability"

                ] * 100,

                2,

            ),

        },

        "date": selected_date,

        "location": {

            "latitude": latitude,

            "longitude": longitude,

        },

        "crop": {

            "stage": normalized_stage,

            "age_days": crop_age_days,

            "maize_variety": maize_variety,

        },

        "weather": {

            "Temperature_C": round(temperature, 2),

            "Humidity_%": round(humidity, 2),

            "Rainfall_mm": round(rainfall, 2),

            "Soil_Moisture_%": round(soil_moisture, 2),

            "Wind_Speed_kmph": round(wind_speed, 2),

        },

        "selected_date_weather": {

            key: round(

                float(value),

                2,

            )

            for key, value in current_weather.items()

        },

        "faw_attack_probability": round(

            probability,

            2,

        ),

        "risk_level": risk_level,

        "leaf_damage_probability": round(

            leaf_damage_probability * 100,

            2,

        ),

        "das": crop_age_days,

        "crop_stage": normalized_stage,

        "current_weather": {

            key: round(

                float(value),

                2,

            )

            for key, value in current_weather.items()

        },

        "forecast_7_day": synthetic_result[

            "forecast_features"

        ],

        "latitude": latitude,

        "longitude": longitude,

    }
