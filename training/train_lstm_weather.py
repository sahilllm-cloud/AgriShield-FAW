from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "faw"
    / "FAW_Maize_Karnataka_Dataset_rebalanced.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "lstm_weather_forecaster.keras"
SCALER_PATH = MODEL_DIR / "lstm_weather_scaler.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading weather data...")

df = pd.read_csv(DATA_PATH)

print(f"Rows loaded: {len(df)}")


# ============================================================
# WEATHER FEATURES
# ============================================================

weather_features = [
    "Temperature_C",
    "Humidity_%",
    "Rainfall_mm",
    "Soil_Moisture_%",
    "Wind_Speed_kmph",
]

missing = [
    col for col in weather_features
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing weather columns: {missing}"
    )


# ============================================================
# REMOVE INVALID VALUES
# ============================================================

df = df.dropna(
    subset=weather_features
).reset_index(drop=True)


# ============================================================
# SORT BY AVAILABLE TIME INFORMATION
# ============================================================

if "Year" in df.columns and "Month" in df.columns:
    df = df.sort_values(
        ["Year", "Month"]
    ).reset_index(drop=True)


# ============================================================
# SCALE WEATHER DATA
# ============================================================

values = df[weather_features].astype(float).values

scaler = MinMaxScaler()

scaled = scaler.fit_transform(values)


# ============================================================
# CREATE SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 7

X = []
y = []

for i in range(
    SEQUENCE_LENGTH,
    len(scaled)
):

    X.append(
        scaled[
            i - SEQUENCE_LENGTH:i
        ]
    )

    y.append(
        scaled[i]
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.float32)

print(f"Training sequences: {len(X)}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Weather variables: {len(weather_features)}")


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split = int(len(X) * 0.80)

X_train = X[:split]
y_train = y[:split]

X_val = X[split:]
y_val = y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")


# ============================================================
# LSTM MODEL
# ============================================================

model = Sequential(
    [
        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                SEQUENCE_LENGTH,
                len(weather_features),
            ),
        ),

        Dropout(0.2),

        LSTM(32),

        Dropout(0.2),

        Dense(32, activation="relu"),

        Dense(
            len(weather_features),
            activation="linear",
        ),
    ]
)


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining LSTM weather model...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)


# ============================================================
# VALIDATION
# ============================================================

val_loss, val_mae = model.evaluate(
    X_val,
    y_val,
    verbose=0,
)

print("\n========================================")
print("LSTM WEATHER RESULTS")
print("========================================")
print(f"Validation MSE : {val_loss:.6f}")
print(f"Validation MAE : {val_mae:.6f}")
print("========================================")


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

joblib.dump(
    {
        "scaler": scaler,
        "weather_features": weather_features,
        "sequence_length": SEQUENCE_LENGTH,
    },
    SCALER_PATH,
)

print("\nLSTM model saved:")
print(MODEL_PATH)

print("\nLSTM scaler saved:")
print(SCALER_PATH)

print("\nDONE.")
