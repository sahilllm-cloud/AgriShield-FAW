import sys
from pathlib import Path

import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader
from torchvision import transforms


# ============================================================
# ALLOW IMPORT FROM training/ DIRECTORY
# ============================================================

TRAINING_DIR = Path(__file__).resolve().parent

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from date_aware_dataset import DateAwareImageDataset


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("AgriShield-FAW — DATE-AWARE SWIN TRAINING")
print("=" * 60)

print(f"Device: {device}")


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "images" / "processed"

TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

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


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLOADING DATASETS")
print("-" * 60)

train_dataset = DateAwareImageDataset(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = DateAwareImageDataset(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = DateAwareImageDataset(
    TEST_DIR,
    transform=eval_transform
)


print("\nDATASET SUMMARY")
print("-" * 60)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Test Images       : {len(test_dataset)}")

print(
    f"Total Images      : "
    f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
)


# ============================================================
# DATALOADERS
# ============================================================

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# CREATE SWIN TRANSFORMER
# ============================================================

print("\nLOADING SWIN TRANSFORMER")
print("-" * 60)

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=2
)

model = model.to(device)

print("Model       : Swin Transformer Tiny")
print("Input Size  : 224 x 224")
print("Classes     : Healthy / Infected")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 10

best_val_accuracy = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\nSTARTING SWIN TRAINING")
print("=" * 60)

for epoch in range(NUM_EPOCHS):

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print("-" * 60)

    model.train()

    running_loss = 0.0

    correct_predictions = 0

    total_samples = 0


    # ========================================================
    # TRAINING BATCHES
    # ========================================================

    for batch_idx, batch in enumerate(train_loader):

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )


        # Clear gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update model
        optimizer.step()


        # Statistics
        running_loss += loss.item()

        predictions = outputs.argmax(
            dim=1
        )

        total_samples += labels.size(0)

        correct_predictions += (
            predictions == labels
        ).sum().item()


        # Progress
        if (
            (batch_idx + 1) % 10 == 0
            or
            (batch_idx + 1) == len(train_loader)
        ):

            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )


    # ========================================================
    # TRAINING RESULTS
    # ========================================================

    train_loss = (
        running_loss /
        len(train_loader)
    )

    train_accuracy = (
        correct_predictions /
        total_samples
    ) * 100


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    validation_loss = 0.0

    validation_correct = 0

    validation_total = 0


    with torch.no_grad():

        for batch in val_loader:

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            labels = batch["label"].to(
                device,
                non_blocking=True
            )


            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            validation_loss += loss.item()


            predictions = outputs.argmax(
                dim=1
            )


            validation_total += labels.size(0)

            validation_correct += (
                predictions == labels
            ).sum().item()


    validation_avg_loss = (
        validation_loss /
        len(val_loader)
    )


    validation_accuracy = (
        validation_correct /
        validation_total
    ) * 100


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\nEPOCH RESULTS")
    print("-" * 40)

    print(
        f"Train Loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy : "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Val Loss       : "
        f"{validation_avg_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{validation_accuracy:.2f}%"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_val_accuracy:

        best_val_accuracy = validation_accuracy


        torch.save(
            model.state_dict(),
            MODELS_DIR / "swin_best.pth"
        )


        print(
            "\nBEST SWIN MODEL SAVED"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 60)

print("LOADING BEST MODEL FOR FINAL TEST")

print("=" * 60)


model.load_state_dict(
    torch.load(
        MODELS_DIR / "swin_best.pth",
        map_location=device
    )
)

model.eval()


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_correct = 0

test_total = 0


with torch.no_grad():

    for batch in test_loader:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )


        outputs = model(images)


        predictions = outputs.argmax(
            dim=1
        )


        test_total += labels.size(0)

        test_correct += (
            predictions == labels
        ).sum().item()


test_accuracy = (
    test_correct /
    test_total
) * 100


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)

print("FINAL SWIN RESULTS")

print("=" * 60)

print(
    f"Training Images   : "
    f"{len(train_dataset)}"
)

print(
    f"Validation Images : "
    f"{len(val_dataset)}"
)

print(
    f"Test Images       : "
    f"{len(test_dataset)}"
)

print(
    f"Total Images      : "
    f"{len(train_dataset) + len(val_dataset) + len(test_dataset)}"
)

print()

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Final Test Accuracy      : "
    f"{test_accuracy:.2f}%"
)

print()

print(
    "Model saved at:"
)

print(
    MODELS_DIR / "swin_best.pth"
)

print("\nSWIN TRAINING COMPLETED")
print("=" * 60)