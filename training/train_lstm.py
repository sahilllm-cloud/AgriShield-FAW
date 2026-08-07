import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# Dataset Path
# ==========================================

DATA_PATH = Path(
    "data/weather/raw/Fall_Armyworm_Maize_Weather_Dataset_1000.csv"
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(DATA_PATH)

print("\nDataset Loaded Successfully!")
print("-" * 50)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\nColumn Names")
print(df.columns.tolist())

# ==========================================
# Dataset Information
# ==========================================

print("\nDataset Information")
print("-" * 50)

print(df.info())

# ==========================================
# First 5 Rows
# ==========================================

print("\nFirst 5 Rows")
print("-" * 50)

print(df.head())

# ==========================================
# Missing Values
# ==========================================

print("\nMissing Values")
print("-" * 50)

print(df.isnull().sum())

# ==========================================
# Risk Level Distribution
# ==========================================

print("\nRisk Level Distribution")
print("-" * 50)

print(df["Risk_Level"].value_counts())

# ==========================================
# Crop Stage Distribution
# ==========================================

print("\nCrop Stage Distribution")
print("-" * 50)

print(df["Crop_Stage"].value_counts())
# ==========================================
# Encode Categorical Features
# ==========================================

crop_stage_encoder = LabelEncoder()
risk_encoder = LabelEncoder()

df["Crop_Stage"] = crop_stage_encoder.fit_transform(
    df["Crop_Stage"]
)

df["Risk_Level"] = risk_encoder.fit_transform(
    df["Risk_Level"]
)

print("\nEncoded Dataset")
print("-" * 50)

print(df.head())

print("\nRisk Label Mapping")
print("-" * 50)

for index, label in enumerate(risk_encoder.classes_):
    print(f"{label} --> {index}")

print("\nCrop Stage Mapping")
print("-" * 50)

for index, label in enumerate(crop_stage_encoder.classes_):
    print(f"{label} --> {index}")
    # ==========================================
# Feature Selection
# ==========================================

feature_columns = [
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

X = df[feature_columns]

y = df["Risk_Level"]

print("\nSelected Features")
print("-" * 50)

print(X.head())

print("\nTarget Labels")
print("-" * 50)

print(y.head())
# ==========================================
# Feature Scaling
# ==========================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures Normalized Successfully!")
print("-" * 50)

print("Shape :", X_scaled.shape)

print("\nFirst 5 Rows")

print(X_scaled[:5])
# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split")
print("-" * 50)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
# ==========================================
# Reshape for LSTM
# ==========================================

X_train = X_train.reshape(
    X_train.shape[0],
    1,
    X_train.shape[1]
)

X_test = X_test.reshape(
    X_test.shape[0],
    1,
    X_test.shape[1]
)

print("\nLSTM Input Shape")
print("-" * 50)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
# ==========================================
# Device Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing Device : {device}")
# ==========================================
# Convert to PyTorch Tensors
# ==========================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.values,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test.values,
    dtype=torch.long
)

print("\nTensor Shapes")
print("-" * 50)

print("X_train :", X_train.shape)
print("y_train :", y_train.shape)

print("X_test  :", X_test.shape)
print("y_test  :", y_test.shape)
# ==========================================
# DataLoaders
# ==========================================

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("\nDataLoaders Ready!")

print(f"Training Batches : {len(train_loader)}")
print(f"Testing Batches  : {len(test_loader)}")
# ==========================================
# LSTM Model
# ==========================================

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(
            64,
            3
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        output = hidden[-1]

        output = self.fc(output)

        return output


model = LSTMModel().to(device)

print("\nLSTM Model")
print("-" * 50)

print(model)

# ==========================================
# Loss Function & Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

NUM_EPOCHS = 20

print("\nTraining Configuration")
print("-" * 50)

print(f"Loss Function : {criterion}")
print(f"Optimizer     : {optimizer.__class__.__name__}")
print(f"Learning Rate : 0.001")
print(f"Epochs        : {NUM_EPOCHS}")
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
# ==========================================
# Training Loop
# ==========================================

best_accuracy = 0

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    print("-" * 50)

    for batch_idx, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 5 == 0:

            print(
                f"Batch [{batch_idx+1}/{len(train_loader)}] "
                f"Loss : {loss.item():.4f}"
            )

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total

    print("\nTraining Results")
    print("-" * 30)

    print(f"Loss     : {train_loss:.4f}")
    print(f"Accuracy : {train_accuracy:.2f}%")
    # ==========================================
# Test Evaluation
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for inputs, labels in test_loader:

        inputs = inputs.to(device)

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.cpu().numpy())

        actuals.extend(labels.numpy())
        # ==========================================
# Test Accuracy
# ==========================================

accuracy = accuracy_score(
    actuals,
    predictions
)

print("\n" + "="*40)

print(f"Test Accuracy : {accuracy*100:.2f}%")

print("="*40)
print("\nClassification Report\n")

print(
    classification_report(
        actuals,
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
        actuals,
        predictions
    )
)
# ==========================================
# Save Model
# ==========================================

torch.save(
    model.state_dict(),
    "models/lstm_best.pth"
)

print("\n✅ LSTM model saved successfully!")