from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "swin_best.pth"

TRAIN_DIR = ROOT / "data" / "images" / "processed" / "train"
VAL_DIR = ROOT / "data" / "images" / "processed" / "val"

OUTPUT_PATH = ROOT / "data" / "swin_train_val_probability_pool.csv"

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Dataset
# -----------------------------
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Train images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Classes:", train_dataset.classes)

# -----------------------------
# Load Swin
# -----------------------------
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

# -----------------------------
# Extract probabilities
# -----------------------------
def extract_probabilities(loader, dataset_name):

    probabilities = []
    labels = []
    paths = []

    with torch.no_grad():

        start_index = 0

        for images, batch_labels in loader:

            images = images.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)

            # Class index 1 = infected
            infected_probs = probs[:, 1].cpu().numpy()

            probabilities.extend(infected_probs.tolist())
            labels.extend(batch_labels.numpy().tolist())

            batch_size = len(batch_labels)

            batch_paths = [
                loader.dataset.samples[start_index + i][0]
                for i in range(batch_size)
            ]

            paths.extend(batch_paths)

            start_index += batch_size

    return pd.DataFrame({
        "image_path": paths,
        "label": labels,
        "leaf_damage_probability": probabilities,
        "dataset": dataset_name
    })


# -----------------------------
# Train + Validation only
# -----------------------------
train_results = extract_probabilities(
    train_loader,
    "train"
)

val_results = extract_probabilities(
    val_loader,
    "val"
)

pool = pd.concat(
    [train_results, val_results],
    ignore_index=True
)

# -----------------------------
# Save
# -----------------------------
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

pool.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nProbability pool created!")
print("Total images:", len(pool))
print("Saved to:", OUTPUT_PATH)

print("\nProbability statistics:")
print(pool["leaf_damage_probability"].describe())