from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
import joblib

from torchvision import datasets, transforms

from synthetic_hybrid_pipeline import (
    generate_dataset,
    build_fusion_frame,
    FUSION_FEATURES,
    FORECAST_FEATURES,
    WEATHER_FEATURES,
    WeatherLSTM,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = ROOT / "data" / "images" / "processed" / "test"

SWIN_PATH = ROOT / "models" / "swin_best.pth"

LSTM_PATH = ROOT / "models" / "synthetic_hybrid_lstm.pth"

SCALER_PATH = ROOT / "models" / "synthetic_hybrid_weather_scaler.pkl"

CLEAN_MODEL_PATH = (
    ROOT / "models" / "synthetic_hybrid_lightgbm_clean.pkl"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# LOAD SWIN
# ============================================================

print("\nLoading Swin...")

swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=2,
)

swin.load_state_dict(
    torch.load(
        SWIN_PATH,
        map_location=device,
    )
)

swin = swin.to(device)
swin.eval()

print("Swin loaded successfully.")


# ============================================================
# TEST DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform,
)


# ============================================================
# SPECIFIC 4 TEST IMAGES
# ============================================================

target_images = [
    "20200610_063444.jpg",
    "20200610_063514.jpg",
    "20200612_103624.jpg",
    "20200612_104321.jpg",
]


# ============================================================
# FIND IMAGES
# ============================================================

image_indices = []

for target_name in target_images:

    found = False

    for index, (path, label) in enumerate(dataset.samples):

        if Path(path).name == target_name:

            image_indices.append(index)
            found = True
            break

    if not found:

        print(
            f"WARNING: {target_name} was not found."
        )


# ============================================================
# GET SWIN PROBABILITIES
# ============================================================

image_results = []

print("\nGetting Swin probabilities...")

with torch.no_grad():

    for index in image_indices:

        image, label = dataset[index]

        image = image.unsqueeze(0).to(device)

        output = swin(image)

        probability = torch.softmax(
            output,
            dim=1,
        )[0, 1].item()

        image_results.append({
            "path": dataset.samples[index][0],
            "class": dataset.classes[label],
            "swin_probability": probability,
        })


# ============================================================
# LOAD CLEAN LIGHTGBM
# ============================================================

print("\nLoading clean LightGBM...")

artifact = joblib.load(
    CLEAN_MODEL_PATH
)

clean_model = artifact["model"]

print("LightGBM loaded successfully.")


# ============================================================
# LOAD LSTM
# ============================================================

print("\nLoading LSTM...")

checkpoint = torch.load(
    LSTM_PATH,
    map_location="cpu",
    weights_only=False,
)

scaler_artifact = joblib.load(
    SCALER_PATH
)

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
# CREATE REFERENCE WEATHER
# ============================================================

print("\nCreating reference weather...")

class Config:
    seed = 42
    field_count = 500
    days_per_field = 100


data = generate_dataset(Config())

first_field_id = data["Field_ID"].iloc[0]

reference_field = data[
    data["Field_ID"] == first_field_id
].copy()

reference_field = reference_field.tail(7).copy()


# ============================================================
# LSTM INPUT
# ============================================================

recent_weather = reference_field[
    WEATHER_FEATURES
].apply(
    pd.to_numeric,
    errors="coerce"
)

scaled_weather = scaler.transform(
    recent_weather
)

lstm_input = torch.tensor(
    scaled_weather,
    dtype=torch.float32
).unsqueeze(0)


# ============================================================
# FORECAST
# ============================================================

with torch.no_grad():

    forecast_scaled = lstm(
        lstm_input
    ).numpy()[0]

forecast = scaler.inverse_transform(
    forecast_scaled
)


# ============================================================
# CREATE BASE ROW
# ============================================================

latest = reference_field.iloc[-1]

row = {}

for feature in WEATHER_FEATURES:

    row[feature] = float(
        latest[feature]
    )


row["DAS"] = int(
    latest["DAS"]
)


row["Crop_Stage"] = str(
    latest["Crop_Stage"]
)


# ============================================================
# FORECAST FEATURES
# ============================================================

for index, feature in enumerate(
    FORECAST_FEATURES
):

    row[feature] = float(
        forecast[:, index].mean()
    )


# ============================================================
# FINAL HYBRID TEST
# ============================================================

print("\n========================================")
print("FINAL CLEAN HYBRID IMAGE TEST")
print("========================================")


for result in image_results:

    test_row = row.copy()

    test_row[
        "FAW_leaf_damage_probability"
    ] = result["swin_probability"]


    frame = pd.DataFrame(
        [test_row]
    )


    features = build_fusion_frame(
        frame
    ).reindex(
        columns=FUSION_FEATURES,
        fill_value=0.0,
    )


    prediction = float(
        np.clip(
            clean_model.predict(
                features
            )[0],
            0.0,
            1.0,
        )
    )


    if prediction < 0.45:

        risk = "LOW"

    elif prediction < 0.70:

        risk = "MEDIUM"

    else:

        risk = "HIGH"


    print("\n----------------------------------------")

    print(
        "Image:",
        Path(result["path"]).name
    )

    print(
        "Class:",
        result["class"]
    )

    print(
        "Swin infected probability:",
        f"{result['swin_probability'] * 100:.4f}%"
    )

    print(
        "FINAL FAW attack probability:",
        f"{prediction * 100:.4f}%"
    )

    print(
        "Risk:",
        risk
    )


print("\n========================================")
print("TEST COMPLETED")
print("========================================")