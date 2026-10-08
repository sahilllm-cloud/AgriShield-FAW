from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from urllib import error, request

import joblib
import numpy as np
import pandas as pd
import timm
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from database.database import get_connection, find_nearest_faw_record
from PIL import Image, UnidentifiedImageError
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from torchvision import transforms

from app.hybrid_predictor import predict_hybrid


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


app = FastAPI(
    title="AgriShield FAW Backend",
    version="1.0.0",
    description="Backend API for crop disease detection and FAW outbreak risk prediction.",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WeatherRiskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Month: int

    Temperature_C: float

    Humidity_pct: float = Field(
        validation_alias=AliasChoices(
            "Humidity_%",
            "Humidity_pct",
            "humidity_pct",
        ),
        serialization_alias="Humidity_%",
    )

    Rainfall_mm: float

    Soil_Moisture_pct: float = Field(
        validation_alias=AliasChoices(
            "Soil_Moisture_%",
            "Soil_Moisture_pct",
            "soil_moisture_pct",
        ),
        serialization_alias="Soil_Moisture_%",
    )

    Wind_Speed_kmph: float

    Crop_Stage: str

    Previous_Pest_Count: int

    Days_Since_Last_Attack: int

    @property
    def feature_dict(self) -> dict[str, Any]:
        return {
            "Month": self.Month,
            "Temperature_C": self.Temperature_C,
            "Humidity_%": self.Humidity_pct,
            "Rainfall_mm": self.Rainfall_mm,
            "Soil_Moisture_%": self.Soil_Moisture_pct,
            "Wind_Speed_kmph": self.Wind_Speed_kmph,
            "Crop_Stage": self._encode_crop_stage(
                self.Crop_Stage
            ),
            "Previous_Pest_Count": self.Previous_Pest_Count,
            "Days_Since_Last_Attack": self.Days_Since_Last_Attack,
        }

    @staticmethod
    def _encode_crop_stage(value: str) -> int:

        mapping = {
            "maturity": 0,
            "seedling": 1,
            "silking": 2,
            "tasseling": 3,
            "vegetative": 4,
            "whorl": 5,
        }

        normalized = str(value).strip().lower()

        return mapping.get(normalized, 1)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": "ok",
        "service": "AgriShield FAW Backend",
        "models": {
            "image": str(
                (MODELS_DIR / "swin_best.pth").exists()
            ),
            "weather": str(
                (MODELS_DIR / "lightgbm_model.pkl").exists()
            ),
            "hybrid": str(
                (MODELS_DIR / "hybrid_faw_model.pkl").exists()
            ),
            "lstm": str(
                (
                    MODELS_DIR
                    / "lstm_weather_forecaster.keras"
                ).exists()
            ),
        },
    }


@app.get("/")
def root() -> dict[str, str]:

    return {
        "message": "AgriShield FAW backend is running."
    }


# ============================================================
# LIVE WEATHER
# ============================================================

@app.get("/api/weather/live")
def get_live_weather() -> dict[str, Any]:

    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=12.972&longitude=77.594"
        "&current=temperature_2m,relative_humidity_2m,"
        "precipitation,wind_speed_10m,"
        "soil_moisture_0_1cm"
        "&timezone=auto"
    )

    try:

        with request.urlopen(
            url,
            timeout=10,
        ) as response:

            payload = json.loads(
                response.read().decode("utf-8")
            )

    except error.URLError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to fetch live weather data "
                "from Open-Meteo."
            ),
        ) from exc

    current = payload.get("current")

    if not current:

        raise HTTPException(
            status_code=502,
            detail=(
                "Open-Meteo response did not include "
                "current weather data."
            ),
        )

    soil_moisture = current.get(
        "soil_moisture_0_1cm"
    )

    if soil_moisture is None:

        soil_moisture = current.get(
            "soil_moisture_1_3cm"
        )

    return {
        "location": "Bengaluru",
        "temperature": current.get(
            "temperature_2m"
        ),
        "humidity": current.get(
            "relative_humidity_2m"
        ),
        "rainfall": current.get(
            "precipitation"
        ),
        "wind_speed": current.get(
            "wind_speed_10m"
        ),
        "soil_moisture": soil_moisture,
    }


# ============================================================
# DATE-SPECIFIC WEATHER
# ============================================================

@app.get("/api/weather/date")
def get_weather_for_date(
    date: str,
    latitude: float = 12.972,
    longitude: float = 77.594,
) -> dict[str, Any]:

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&start_date={date}"
        f"&end_date={date}"
        "&daily=temperature_2m_mean,"
        "relative_humidity_2m_mean,"
        "precipitation_sum,"
        "wind_speed_10m_mean"
        "&timezone=auto"
    )

    try:

        with request.urlopen(
            url,
            timeout=10,
        ) as response:

            payload = json.loads(
                response.read().decode("utf-8")
            )

    except error.URLError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Failed to fetch date-specific "
                "weather from Open-Meteo."
            ),
        ) from exc

    daily = payload.get("daily")

    if not daily:

        raise HTTPException(
            status_code=502,
            detail=(
                "Open-Meteo did not return "
                "daily weather data."
            ),
        )

    def first_value(key: str):

        values = daily.get(key, [])

        return values[0] if values else None

    return {
        "date": date,
        "latitude": latitude,
        "longitude": longitude,
        "temperature": first_value(
            "temperature_2m_mean"
        ),
        "humidity": first_value(
            "relative_humidity_2m_mean"
        ),
        "rainfall": first_value(
            "precipitation_sum"
        ),
        "wind_speed": first_value(
            "wind_speed_10m_mean"
        ),
    }


# ============================================================
# FINAL HYBRID PREDICTION
# ============================================================

@app.post("/api/predict/hybrid")
async def predict_hybrid_endpoint(
    file: UploadFile = File(...),
    date: str = "2026-10-08",
    latitude: float = 12.972,
    longitude: float = 77.594,
    crop_stage: str = "Whorl",
    crop_age_days: int = 35,
    maize_variety: str = "Local",
) -> dict[str, Any]:

    return await predict_hybrid(
        file=file,
        selected_date=date,
        latitude=latitude,
        longitude=longitude,
        crop_stage=crop_stage,
        crop_age_days=crop_age_days,
        maize_variety=maize_variety,
    )


# ============================================================
# FAW RECORDS
# ============================================================

@app.get("/api/faw/records")
def get_faw_records(
    limit: int = 10,
) -> dict[str, Any]:

    connection = get_connection()

    query = """
        SELECT *
        FROM faw_records
        LIMIT ?
    """

    df = pd.read_sql_query(
        query,
        connection,
        params=(limit,),
    )

    connection.close()

    return {
        "count": len(df),
        "records": df.to_dict(
            orient="records"
        ),
    }


# ============================================================
# FAW LOCATION
# ============================================================

@app.get("/api/faw/location")
def get_faw_location(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:

    record = find_nearest_faw_record(
        latitude,
        longitude,
    )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail="No FAW field record found.",
        )

    return {
        "location": {
            "latitude": record["Latitude"],
            "longitude": record["Longitude"],
            "district": record["District"],
            "taluk": record["Taluk"],
            "village": record["Village"],
        },
        "conditions": {
            "month": record["Month"],
            "year": record["Year"],
            "temperature": record["Temperature_C"],
            "humidity": record["Humidity_%"],
            "rainfall": record["Rainfall_mm"],
            "soil_moisture": record["Soil_Moisture_%"],
            "wind_speed": record["Wind_Speed_kmph"],
            "crop_stage": record["Crop_Stage"],
            "maize_variety": record["Maize_Variety"],
            "crop_age_days": record["Crop_Age_Days"],
            "previous_pest_count": record[
                "Previous_Pest_Count"
            ],
            "days_since_last_attack": record[
                "Days_Since_Last_Attack"
            ],
            "risk_level": record["Risk_Level"],
        },
    }


# ============================================================
# FAW LOCATION PREDICTION
# ============================================================

@app.get("/api/faw/predict-location")
def predict_faw_from_location(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:

    record = find_nearest_faw_record(
        latitude,
        longitude,
    )

    if record is None:

        raise HTTPException(
            status_code=404,
            detail="No FAW field record found.",
        )

    model_path = (
        MODELS_DIR
        / "lightgbm_faw_model.pkl"
    )

    encoders_path = (
        MODELS_DIR
        / "faw_encoders.pkl"
    )

    if not model_path.exists():

        raise HTTPException(
            status_code=500,
            detail="FAW model not found.",
        )

    if not encoders_path.exists():

        raise HTTPException(
            status_code=500,
            detail="FAW encoders not found.",
        )

    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)

    try:

        features = [
            encoders["district"].transform(
                [record["District"]]
            )[0],

            encoders["taluk"].transform(
                [record["Taluk"]]
            )[0],

            encoders["village"].transform(
                [record["Village"]]
            )[0],

            record["Latitude"],
            record["Longitude"],
            record["Month"],
            record["Year"],
            record["Temperature_C"],
            record["Humidity_%"],
            record["Rainfall_mm"],
            record["Soil_Moisture_%"],
            record["Wind_Speed_kmph"],

            encoders["crop_stage"].transform(
                [record["Crop_Stage"]]
            )[0],

            encoders["maize_variety"].transform(
                [record["Maize_Variety"]]
            )[0],

            record["Crop_Age_Days"],
            record["Previous_Pest_Count"],
            record["Days_Since_Last_Attack"],
        ]

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=f"Encoder error: {exc}",
        ) from exc

    feature_names = [
        "District",
        "Taluk",
        "Village",
        "Latitude",
        "Longitude",
        "Month",
        "Year",
        "Temperature_C",
        "Humidity_%",
        "Rainfall_mm",
        "Soil_Moisture_%",
        "Wind_Speed_kmph",
        "Crop_Stage",
        "Maize_Variety",
        "Crop_Age_Days",
        "Previous_Pest_Count",
        "Days_Since_Last_Attack",
    ]

    input_df = pd.DataFrame(
        [features],
        columns=feature_names,
    )

    prediction = int(
        model.predict(input_df)[0]
    )

    probabilities = model.predict_proba(
        input_df
    )[0]

    risk_encoder = encoders["risk"]

    risk_label = str(
        risk_encoder.inverse_transform(
            [prediction]
        )[0]
    )

    return {
        "location": {
            "district": record["District"],
            "taluk": record["Taluk"],
            "village": record["Village"],
            "latitude": record["Latitude"],
            "longitude": record["Longitude"],
        },
        "risk_level": risk_label,
        "confidence": float(
            np.max(probabilities)
        ),
        "probabilities": {
            str(
                risk_encoder.inverse_transform(
                    [i]
                )[0]
            ): float(
                probabilities[i]
            )
            for i in range(
                len(probabilities)
            )
        },
    }


# ============================================================
# OLD WEATHER PREDICTION
# ============================================================

@app.post("/api/predict/weather")
def predict_weather_risk(
    payload: WeatherRiskRequest,
) -> dict[str, Any]:

    model_path = (
        MODELS_DIR
        / "lightgbm_model.pkl"
    )

    if not model_path.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Weather model not found in "
                "models/lightgbm_model.pkl"
            ),
        )

    model = joblib.load(model_path)

    feature_order = [
        "Month",
        "Temperature_C",
        "Humidity_%",
        "Rainfall_mm",
        "Soil_Moisture_%",
        "Wind_Speed_kmph",
        "Crop_Stage",
        "Previous_Pest_Count",
        "Days_Since_Last_Attack",
    ]

    df = pd.DataFrame(
        [payload.feature_dict],
        columns=feature_order,
    )

    prediction_code = int(
        model.predict(df)[0]
    )

    probabilities = model.predict_proba(
        df
    )[0]

    risk_labels = {
        0: "High",
        1: "Low",
        2: "Medium",
    }

    top_label = risk_labels.get(
        prediction_code,
        "Unknown",
    )

    return {
        "risk_level": top_label,
        "confidence": float(
            np.max(probabilities)
        ),
        "probabilities": {
            risk_labels.get(
                i,
                str(i)
            ): float(
                probabilities[i]
            )
            for i in range(
                len(probabilities)
            )
        },
    }


# ============================================================
# IMAGE PREDICTION
# ============================================================

@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
) -> dict[str, Any]:

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected.",
        )

    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Upload a valid image file.",
        )

    model_path = (
        MODELS_DIR
        / "swin_best.pth"
    )

    if not model_path.exists():

        raise HTTPException(
            status_code=500,
            detail=(
                "Image model not found in "
                "models/swin_best.pth"
            ),
        )

    image_bytes = await file.read()

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except UnidentifiedImageError as exc:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the uploaded image."
            ),
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

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
    )

    model.head.fc = torch.nn.Linear(
        model.head.fc.in_features,
        2,
    )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        prediction_index = int(
            torch.argmax(
                logits,
                dim=1,
            ).item()
        )

    label_names = [
        "healthy",
        "infected",
    ]

    prediction_label = (
        label_names[prediction_index]
    )

    probs = probabilities.numpy().tolist()

    return {
        "predicted_class": prediction_label,
        "confidence": float(
            probs[prediction_index]
        ),
        "probabilities": {
            label_names[i]: float(
                probs[i]
            )
            for i in range(
                len(label_names)
            )
        },
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )