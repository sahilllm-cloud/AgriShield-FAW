```python
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 4

CLASS_NAMES = ["Healthy", "Infected"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("AgriShield-FAW - SWIN TRANSFORMER TRAINING")
print("=" * 65)
print(f"Device: {device}")


# ============================================================
# PROJECT PATHS
# ============================================================

TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAINING_DIR.parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from date_aware_dataset import DateAwareImageDataset

DATA_DIR = PROJECT_DIR / "data" / "images" / "processed"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Save experimental training separately.
# This protects the production checkpoint.
BEST_MODEL_PATH = MODELS_DIR / "swin_best_retrained.pth"


# ============================================================
# CHECK DATA DIRECTORIES
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {folder}"
        )


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLOADING DATASETS")
print("-" * 65)

train_dataset = DateAwareImageDataset(
    TRAIN_DIR,
    transform=train_transform,
)

val_dataset = DateAwareImageDataset(
    VAL_DIR,
    transform=eval_transform,
)

test_dataset = DateAwareImageDataset(
    TEST_DIR,
    transform=eval_transform,
)

if len(train_dataset) == 0:
    raise ValueError("Training dataset is empty.")

if len(val_dataset) == 0:
    raise ValueError("Validation dataset is empty.")

if len(test_dataset) == 0:
    raise ValueError("Test dataset is empty.")

print(f"Training images   : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Test images       : {len(test_dataset)}")
print(
    f"Total images      : "
    f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
)


# ============================================================
# CREATE SWIN TRANSFORMER
# ============================================================

print("\nCREATING SWIN TRANSFORMER")
print("-" * 65)

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=2,
)

model = model.to(device)

print("Architecture : Swin Transformer Tiny")
print("Input size   : 224 x 224")
print("Class 0      : Healthy")
print("Class 1      : Infected")


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
)


# ============================================================
# TRAINING AND EVALUATION FUNCTIONS
# ============================================================

def run_epoch(loader, training=False):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device).long()

            if training:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels, all_predictions
    ) * 100

    balanced_accuracy = balanced_accuracy_score(
        all_labels, all_predictions
    ) * 100

    return (
        average_loss,
        accuracy,
        balanced_accuracy,
        all_labels,
        all_predictions,
    )


def print_metrics(title, labels, predictions):
    print(f"\n{title}")
    print("-" * 65)

    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
            digits=4,
        )
    )

    print("Confusion matrix:")
    print("Rows = actual; columns = predicted")
    print("Order = Healthy, Infected")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))


# ============================================================
# TRAINING LOOP WITH EARLY STOPPING
# ============================================================

print("\nSTARTING TRAINING")
print("=" * 65)

best_val_balanced_accuracy = -1.0
epochs_without_improvement = 0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    print("-" * 65)

    train_results = run_epoch(train_loader, training=True)

    (
        train_loss,
        train_accuracy,
        train_balanced_accuracy,
        _,
        _,
    ) = train_results

    val_results = run_epoch(val_loader, training=False)

    (
        val_loss,
        val_accuracy,
        val_balanced_accuracy,
        val_labels,
        val_predictions,
    ) = val_results

    scheduler.step(val_balanced_accuracy)

    print(f"Train loss                : {train_loss:.4f}")
    print(f"Train accuracy            : {train_accuracy:.2f}%")
    print(f"Train balanced accuracy   : {train_balanced_accuracy:.2f}%")
    print(f"Validation loss           : {val_loss:.4f}")
    print(f"Validation accuracy       : {val_accuracy:.2f}%")
    print(f"Validation balanced acc.   : {val_balanced_accuracy:.2f}%")
    print(f"Learning rate              : {optimizer.param_groups[0]['lr']:.2e}")

    if val_balanced_accuracy > best_val_balanced_accuracy:
        best_val_balanced_accuracy = val_balanced_accuracy
        epochs_without_improvement = 0

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        print(f"\nBest experimental checkpoint saved:")
        print(BEST_MODEL_PATH)

        print_metrics(
            "VALIDATION CLASS-WISE METRICS",
            val_labels,
            val_predictions,
        )

    else:
        epochs_without_improvement += 1
        print(
            f"No improvement: "
            f"{epochs_without_improvement}/{PATIENCE}"
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

if not BEST_MODEL_PATH.exists():
    raise RuntimeError("No trained checkpoint was saved.")

model.load_state_dict(
    torch.load(
        BEST_MODEL_PATH,
        map_location=device,
    )
)

model.eval()

(
    test_loss,
    test_accuracy,
    test_balanced_accuracy,
    test_labels,
    test_predictions,
) = run_epoch(test_loader, training=False)

print(f"\nTest loss              : {test_loss:.4f}")
print(f"Test accuracy          : {test_accuracy:.2f}%")
print(f"Test balanced accuracy : {test_balanced_accuracy:.2f}%")

print_metrics(
    "FINAL TEST CLASSIFICATION REPORT",
    test_labels,
    test_predictions,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SWIN TRAINING COMPLETED")
print("=" * 65)

print(f"Training images             : {len(train_dataset)}")
print(f"Validation images           : {len(val_dataset)}")
print(f"Test images                 : {len(test_dataset)}")
print(
    f"Best validation balanced acc: "
    f"{best_val_balanced_accuracy:.2f}%"
)
print(f"Final test accuracy         : {test_accuracy:.2f}%")
print(f"Final test balanced accuracy: {test_balanced_accuracy:.2f}%")
print(f"Experimental model saved at : {BEST_MODEL_PATH}")

print("\nProduction checkpoint was not overwritten.")
print("=" * 65)
```