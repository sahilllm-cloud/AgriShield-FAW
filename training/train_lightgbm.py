import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib


# ==========================================
# Load FAW Dataset
# ==========================================

DATA_PATH = Path(
    "data/faw/FAW_Maize_Karnataka_Dataset_rebalanced.csv"
)

df = pd.read_csv(DATA_PATH)

print("\nFAW Dataset Loaded Successfully!")
print("-" * 50)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ==========================================
# Encode Categorical Features
# ==========================================

district_encoder = LabelEncoder()
taluk_encoder = LabelEncoder()
village_encoder = LabelEncoder()
crop_stage_encoder = LabelEncoder()
variety_encoder = LabelEncoder()
risk_encoder = LabelEncoder()

df["District"] = district_encoder.fit_transform(df["District"])
df["Taluk"] = taluk_encoder.fit_transform(df["Taluk"])
df["Village"] = village_encoder.fit_transform(df["Village"])
df["Crop_Stage"] = crop_stage_encoder.fit_transform(df["Crop_Stage"])
df["Maize_Variety"] = variety_encoder.fit_transform(df["Maize_Variety"])

df["Risk_Level"] = risk_encoder.fit_transform(df["Risk_Level"])


# ==========================================
# Feature Selection
# ==========================================

features = [
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
    "Days_Since_Last_Attack"
]

X = df[features]
y = df["Risk_Level"]


# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split")
print("-" * 50)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ==========================================
# Create LightGBM Model
# ==========================================

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

print("\nLightGBM Model Created Successfully!")


# ==========================================
# Train Model
# ==========================================

print("\nTraining LightGBM...")
print("-" * 50)

model.fit(X_train, y_train)

print("Training Completed!")


# ==========================================
# Predictions
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 40)
print(f"Test Accuracy : {accuracy * 100:.2f}%")
print("=" * 40)


# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=risk_encoder.classes_
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    "models/lightgbm_faw_model.pkl"
)

print("\nLightGBM FAW model saved successfully!")


# ==========================================
# Save Encoders
# ==========================================

encoders = {
    "district": district_encoder,
    "taluk": taluk_encoder,
    "village": village_encoder,
    "crop_stage": crop_stage_encoder,
    "maize_variety": variety_encoder,
    "risk": risk_encoder
}

joblib.dump(
    encoders,
    "models/faw_encoders.pkl"
)

print("FAW encoders saved successfully!")


# ==========================================
# Feature Importance
# ==========================================

print("\nFeature Importance")
print("-" * 50)

importance = model.feature_importances_

for feature, score in sorted(
    zip(features, importance),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{feature:30} {score}")