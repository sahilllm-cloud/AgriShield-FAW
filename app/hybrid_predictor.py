from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, UploadFile
from torchvision import transforms

from app.weather_forecast import generate_lstm_forecast


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
    # WEATHER + LSTM
    # --------------------------------------------------------

    weather_result = generate_lstm_forecast(
        date=selected_date,
        latitude=latitude,
        longitude=longitude,
    )

    weather = weather_result[
        "lstm_forecast"
    ]

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    month = date.month

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    temperature = weather[
        "Temperature_C"
    ]

    humidity = weather[
        "Humidity_%"
    ]

    rainfall = weather[
        "Rainfall_mm"
    ]

    soil_moisture = weather[
        "Soil_Moisture_%"
    ]

    wind_speed = weather[
        "Wind_Speed_kmph"
    ]

    # --------------------------------------------------------
    # HYBRID FEATURES
    # --------------------------------------------------------

    sample = pd.DataFrame(
        [
            {
                "Month": month,

                "Month_sin": month_sin,

                "Month_cos": month_cos,

                "Temperature_C": temperature,

                "Humidity_%": humidity,

                "Rainfall_mm": rainfall,

                "Soil_Moisture_%": soil_moisture,

                "Wind_Speed_kmph": wind_speed,

                "Crop_Age_Days": crop_age_days,

                "Crop_Stage_Code": STAGE_MAP[
                    normalized_stage
                ],

                "Swin_Mean": image_features[
                    "Swin_Mean"
                ],

                "Swin_Std": image_features[
                    "Swin_Std"
                ],

                "Swin_Norm": image_features[
                    "Swin_Norm"
                ],

                "Image_Class": image_features[
                    "Image_Class"
                ],

                "Temperature_Humidity":
                    temperature * humidity,

                "Rainfall_SoilMoisture":
                    rainfall * soil_moisture,

                "Weather_Stress":
                    temperature
                    + 0.20 * humidity
                    + 0.30 * rainfall,

                "Crop_Stage": normalized_stage,

                "Maize_Variety": maize_variety,
            }
        ]
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    artifact = load_hybrid_model()

    model = artifact["model"]

    probability = model.predict(
        sample
    )[0]

    probability = float(
        np.clip(
            probability,
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

            key: round(
                float(value),
                2,
            )

            for key, value in weather.items()

        },

        "selected_date_weather": {

            key: round(
                float(value),
                2,
            )

            for key, value in weather_result[
                "selected_weather"
            ].items()

        },

    }