print("Step 1")

import torch
import torch.nn as nn
import timm

from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("All imports successful")

# ==========================================
# Device Configuration
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")
# ==========================================
# Load Swin Transformer
# ==========================================

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False
)

model.head.fc = nn.Linear(
    model.head.fc.in_features,
    2
)

MODEL_PATH = Path("models/swin_best.pth")

print("Loading model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("✅ Model loaded successfully!")
# ==========================================
# Test Dataset
# ==========================================

TEST_DIR = Path("data/images/processed/test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("✅ Test dataset loaded!")
print(f"Images : {len(test_dataset)}")
print(f"Classes: {test_dataset.classes}")
# ==========================================
# Evaluation
# ==========================================

all_predictions = []
all_labels = []

correct = 0
total = 0

print("Evaluating model...")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n==============================")
print(f"Test Accuracy : {accuracy*100:.2f}%")
print("==============================")

print("\nClassification Report\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)