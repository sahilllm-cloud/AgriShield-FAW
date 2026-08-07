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
# Load Dataset
# ==========================================

DATA_PATH = Path(
    "data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_1000.csv"
)

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully!")
print("-" * 50)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ==========================================
# Encode Labels
# ==========================================

crop_encoder = LabelEncoder()
risk_encoder = LabelEncoder()

df["Crop_Stage"] = crop_encoder.fit_transform(
    df["Crop_Stage"]
)

df["Risk_Level"] = risk_encoder.fit_transform(
    df["Risk_Level"]
)

# ==========================================
# Feature Selection
# ==========================================

features = [
    "Month",
    "Temperature_C",
    "Humidity_%",
    "Rainfall_mm",
    "Soil_Moisture_%",
    "Wind_Speed_kmph",
    "Crop_Stage",
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

print("✅ Training Completed!")
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

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "High",
            "Low",
            "Medium"
        ]
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)# ==========================================
# Save Model
# ==========================================

joblib.dump(
    model,
    "models/lightgbm_model.pkl"
)

print("\n✅ LightGBM model saved successfully!")
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