from __future__ import annotations

import io
import json
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


def fetch_weather_history(
    selected_date: str,
    latitude: float,
    longitude: float,
) -> list[dict[str, float]]:

    try:
        end_date = date_type.fromisoformat(selected_date)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid date. Use YYYY-MM-DD.",
        ) from exc

    start_date = end_date - timedelta(days=6)
    today = date_type.today()

    if end_date >= today and (end_date - today).days <= 16:
        base_url = "https://api.open-meteo.com/v1/forecast"
    elif end_date < today:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "The selected future date is outside "
                "Open-Meteo's forecast range."
            ),
        )

    url = (
        f"{base_url}?latitude={latitude}&longitude={longitude}"
        f"&start_date={start_date.isoformat()}"
        f"&end_date={end_date.isoformat()}"
        "&hourly=temperature_2m,relative_humidity_2m,"
        "precipitation,wind_speed_10m,soil_moisture_0_to_1cm"
        "&timezone=auto"
    )

    try:
        with request.urlopen(url, timeout=15) as response:
            payload = json.loads(
                response.read().decode("utf-8")
            )
    except error.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Open-Meteo rejected the weather request ({exc.code}).",
        ) from exc
    except error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch weather from Open-Meteo.",
        ) from exc

    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        raise HTTPException(
            status_code=502,
            detail="Open-Meteo did not return hourly weather data.",
        )

    rows = []
    for day_index in range(7):
        values = {}
        start = day_index * 24
        end = start + 24

        for source, target in (
            ("temperature_2m", "Temperature_C"),
            ("relative_humidity_2m", "Humidity_pct"),
            ("wind_speed_10m", "Wind_Speed_kmph"),
            ("soil_moisture_0_to_1cm", "Soil_Moisture_pct"),
        ):
            samples = [
                value
                for value in hourly.get(source, [])[start:end]
                if value is not None
            ]
            values[target] = float(np.mean(samples)) if samples else 0.0

        rainfall = [
            value
            for value in hourly.get("precipitation", [])[start:end]
            if value is not None
        ]
        values["Rainfall_mm"] = float(np.sum(rainfall)) if rainfall else 0.0
        values["Soil_Moisture_pct"] *= 100.0
        rows.append(values)

    return rows


# ============================================================
# IMAGE FEATURES
# ============================================================

async def extract_image_features(
    file: UploadFile,
) -> dict[str, float]:

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

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError as exc:

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded image.",
        ) from exc

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
        # `features` is already pooled.
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
    print(
        "DEBUG hybrid image:",
        file.filename,
        "Infected_Probability:",
        leaf_damage_probability,
    )

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
    print(
        "DEBUG hybrid result:",
        file.filename,
        "faw_attack_probability:",
        probability,
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