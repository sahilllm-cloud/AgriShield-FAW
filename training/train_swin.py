import torch
import torch.nn as nn
import timm

from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# Device Configuration
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")


# ==========================================
# Load Pretrained Swin Transformer
# ==========================================

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True
)

print("\nPretrained Swin Transformer loaded successfully.")


# ==========================================
# Replace Classification Head
# ==========================================

model.head.fc = nn.Linear(
    model.head.fc.in_features,
    2
)

print("\nUpdated Classification Head:")
print(model.head)


# ==========================================
# Move Model to GPU
# ==========================================

model = model.to(device)

print(f"\nModel is on: {next(model.parameters()).device}")


# ==========================================
# Loss Function
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)


# ==========================================
# Training Configuration
# ==========================================

print("\nTraining configuration ready!")
print(f"Loss Function : {criterion}")
print(f"Optimizer     : {optimizer.__class__.__name__}")
print("Learning Rate : 1e-4")
print("Weight Decay  : 0.01")
# ==========================================
# Dataset Paths
# ==========================================

DATA_DIR = Path("data/images/processed")

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"


# ==========================================
# Image Transforms
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
# ==========================================
# Load Datasets
# ==========================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)
# ==========================================
# DataLoaders
# ==========================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)
# ==========================================
# Dataset Information
# ==========================================

print("\nDataset Loaded Successfully!")
print("-" * 40)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")

print(f"Classes           : {train_dataset.classes}")

print(f"Batch Size        : {BATCH_SIZE}")
# ==========================================
# Training Hyperparameters
# ==========================================

NUM_EPOCHS = 10

best_val_accuracy = 0.0

print("\nTraining Hyperparameters")
print("-" * 40)
print(f"Epochs            : {NUM_EPOCHS}")
print(f"Training Batches  : {len(train_loader)}")
print(f"Validation Batches: {len(val_loader)}")
# ==========================================
# Create Models Directory
# ==========================================

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ==========================================
# Training Loop
# ==========================================

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    print("-" * 50)

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # ======================================
    # Training
    # ======================================

    for batch_idx, (images, labels) in enumerate(train_loader):

        # Move to GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Clear gradients
        optimizer.zero_grad()

        # Forward Pass
        outputs = model(images)

        # Compute Loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        # -----------------------------
        # Statistics
        # -----------------------------

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total_samples += labels.size(0)

        correct_predictions += (predicted == labels).sum().item()

        # -----------------------------
        # Batch Progress
        # -----------------------------

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):

            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    # ======================================
    # Training Summary
    # ======================================

    train_loss = running_loss / len(train_loader)

    train_accuracy = (
        correct_predictions /
        total_samples
    ) * 100

    print("\nTraining Results")
    print("-" * 30)
    print(f"Loss     : {train_loss:.4f}")
    print(f"Accuracy : {train_accuracy:.2f}%")
        # ======================================
    # Validation
    # ======================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()

    # ======================================
    # Validation Summary
    # ======================================

    avg_val_loss = val_loss / len(val_loader)

    val_accuracy = (val_correct / val_total) * 100

    print("\nValidation Results")
    print("-" * 30)
    print(f"Loss     : {avg_val_loss:.4f}")
    print(f"Accuracy : {val_accuracy:.2f}%")

    # ======================================
    # Save Best Model
    # ======================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODELS_DIR / "swin_best.pth"
        )

        print("\n✅ Best model updated and saved!")

    print("=" * 60)

# ==========================================
# Training Completed
# ==========================================

print("\n🎉 Training Completed Successfully!")
print(f"Best Validation Accuracy : {best_val_accuracy:.2f}%")
print(f"Best Model Saved At      : {MODELS_DIR / 'swin_best.pth'}")