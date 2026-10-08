from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "faw"
    / "FAW_Maize_Karnataka_Dataset_rebalanced.csv"
)

MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "environment_faw_risk.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AgriShield-FAW — ENVIRONMENTAL RISK MODEL")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"\nRows loaded: {len(df)}")


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "Month",
    "Year",
    "Temperature_C",
    "Humidity_%",
    "Rainfall_mm",
    "Soil_Moisture_%",
    "Wind_Speed_kmph",
    "Crop_Stage",
]

TARGET = "Risk_Level"


X = df[FEATURES].copy()

y = df[TARGET].copy()


# ============================================================
# REMOVE UNUSED PEST-HISTORY FEATURES
# ============================================================

print("\nFeatures used:")
for feature in FEATURES:
    print(" -", feature)

print("\nExcluded:")
print(" - Previous_Pest_Count")
print(" - Days_Since_Last_Attack")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nDATA SPLIT")
print("-" * 50)

print("Training   :", len(X_train))
print("Validation :", len(X_val))


# ============================================================
# CATEGORICAL + NUMERIC FEATURES
# ============================================================

categorical_features = [
    "Crop_Stage"
]

numeric_features = [
    "Month",
    "Year",
    "Temperature_C",
    "Humidity_%",
    "Rainfall_mm",
    "Soil_Moisture_%",
    "Wind_Speed_kmph",
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        ),
    ]
)


# ============================================================
# LIGHTGBM MODEL
# ============================================================

classifier = LGBMClassifier(
    objective="multiclass",
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    verbosity=-1
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\nTRAINING ENVIRONMENT MODEL")
print("=" * 70)

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION
# ============================================================

predictions = pipeline.predict(
    X_val
)

probabilities = pipeline.predict_proba(
    X_val
)


accuracy = accuracy_score(
    y_val,
    predictions
)


print("\nVALIDATION RESULTS")
print("-" * 50)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\nMODEL SAVED")
print("-" * 50)

print(MODEL_PATH)


# ============================================================
# CHECK PROBABILITIES
# ============================================================

print("\nSAMPLE RISK PROBABILITIES")
print("-" * 50)

classes = pipeline.classes_

for i in range(min(5, len(probabilities))):

    print(
        f"\nSample {i + 1}"
    )

    for class_name, probability in zip(
        classes,
        probabilities[i]
    ):

        print(
            f"  {class_name}: "
            f"{probability * 100:.2f}%"
        )


print("\nENVIRONMENT MODEL TRAINING COMPLETED")
print("=" * 70)