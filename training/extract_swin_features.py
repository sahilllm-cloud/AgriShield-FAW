from pathlib import Path
import sys

import torch
import timm
import pandas as pd

from torch.utils.data import DataLoader
from torchvision import transforms


# ============================================================
# IMPORT DATE-AWARE DATASET
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

print("=" * 70)
print("AgriShield-FAW — SWIN FEATURE EXTRACTION")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "images" / "processed"

MODEL_PATH = PROJECT_DIR / "models" / "swin_best.pth"

OUTPUT_DIR = PROJECT_DIR / "data" / "features"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD SWIN MODEL
# ============================================================

print("\nLOADING TRAINED SWIN MODEL")
print("-" * 70)

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=2
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Loaded:", MODEL_PATH)


# ============================================================
# LOAD ALL THREE SPLITS
# ============================================================

splits = {
    "train": DATA_DIR / "train",
    "val": DATA_DIR / "val",
    "test": DATA_DIR / "test"
}


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(dataset, split_name):

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    all_features = []

    all_labels = []

    all_dates = []

    all_paths = []

    print(
        f"\nExtracting {split_name} features..."
    )

    with torch.no_grad():

        for batch_idx, batch in enumerate(loader):

            images = batch["image"].to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Extract Swin visual representation
            # ------------------------------------------------

            features = model.forward_features(images)

            # Swin output:
            # [batch, tokens, channels]
            #
            # forward_head(pre_logits=True)
            # performs the model's learned normalization/pooling
            # and returns the feature representation before
            # the final classification layer.

            features = model.forward_head(
                features,
                pre_logits=True
            )

            features = features.cpu()

            all_features.append(
                features
            )

            all_labels.extend(
                batch["label"].tolist()
            )

            all_dates.extend(
                list(batch["date"])
            )

            all_paths.extend(
                list(batch["path"])
            )

            if (
                (batch_idx + 1) % 10 == 0
                or
                (batch_idx + 1) == len(loader)
            ):

                print(
                    f"Batch "
                    f"[{batch_idx + 1}/{len(loader)}]"
                )


    # --------------------------------------------------------
    # Combine feature batches
    # --------------------------------------------------------

    all_features = torch.cat(
        all_features,
        dim=0
    ).numpy()


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    feature_columns = [
        f"swin_feature_{i}"
        for i in range(
            all_features.shape[1]
        )
    ]


    feature_df = pd.DataFrame(
        all_features,
        columns=feature_columns
    )


    # --------------------------------------------------------
    # Add metadata
    # --------------------------------------------------------

    feature_df.insert(
        0,
        "image_path",
        all_paths
    )

    feature_df.insert(
        1,
        "date",
        all_dates
    )

    feature_df.insert(
        2,
        "label",
        all_labels
    )

    feature_df.insert(
        3,
        "split",
        split_name
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR /
        f"swin_features_{split_name}.csv"
    )

    feature_df.to_csv(
        output_path,
        index=False
    )


    print(
        f"\nSaved: {output_path}"
    )

    print(
        f"Images: {len(feature_df)}"
    )

    print(
        f"Feature dimensions: "
        f"{all_features.shape[1]}"
    )


    return feature_df


# ============================================================
# EXTRACT ALL SPLITS
# ============================================================

all_data = []

for split_name, split_path in splits.items():

    dataset = DateAwareImageDataset(
        split_path,
        transform=transform
    )

    df = extract_features(
        dataset,
        split_name
    )

    all_data.append(df)


# ============================================================
# COMBINE ALL 4,225 IMAGES
# ============================================================

print("\n" + "=" * 70)
print("COMBINING ALL IMAGE FEATURES")
print("=" * 70)

combined_df = pd.concat(
    all_data,
    ignore_index=True
)


combined_path = (
    OUTPUT_DIR /
    "swin_features_all_4225.csv"
)


combined_df.to_csv(
    combined_path,
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\nFINAL FEATURE DATASET")
print("-" * 70)

print(
    f"Total images : "
    f"{len(combined_df)}"
)

print(
    f"Train        : "
    f"{(combined_df['split'] == 'train').sum()}"
)

print(
    f"Validation   : "
    f"{(combined_df['split'] == 'val').sum()}"
)

print(
    f"Test         : "
    f"{(combined_df['split'] == 'test').sum()}"
)

print(
    f"Unique dates : "
    f"{combined_df['date'].nunique()}"
)

print(
    "\nDates:"
)

for date, count in (
    combined_df["date"]
    .value_counts()
    .sort_index()
    .items()
):

    print(
        f"  {date}: {count}"
    )


print("\nSaved combined file:")
print(combined_path)

print("\nSWIN FEATURE EXTRACTION COMPLETED")
print("=" * 70)