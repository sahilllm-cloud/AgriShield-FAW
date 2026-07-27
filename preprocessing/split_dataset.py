from pathlib import Path
import random
import shutil

# -----------------------------
# Configuration
# -----------------------------

RAW_DATA = Path("data/images/raw")
OUTPUT_DATA = Path("data/images/processed")

CLASSES = ["healthy", "infected"]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

random.seed(RANDOM_SEED)


# -----------------------------
# Create output folders
# -----------------------------

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (OUTPUT_DATA / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# -----------------------------
# Split each class separately
# -----------------------------

for class_name in CLASSES:

    source_folder = RAW_DATA / class_name

    images = [
        file
        for file in source_folder.iterdir()
        if file.suffix.lower() in VALID_EXTENSIONS
    ]

    # Sort first so the split is reproducible
    images = sorted(images)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        destination = OUTPUT_DATA / split_name / class_name

        for image_path in split_images:
            shutil.copy2(
                image_path,
                destination / image_path.name
            )

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")


print("\nDataset split completed successfully.")