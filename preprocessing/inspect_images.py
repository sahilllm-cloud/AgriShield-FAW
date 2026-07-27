from pathlib import Path
from PIL import Image
from collections import Counter

# Path to the raw image dataset
DATASET_PATH = Path("data/images/raw")

# Dataset classes
CLASS_FOLDERS = {
    "healthy": DATASET_PATH / "healthy",
    "infected": DATASET_PATH / "infected",
}

# Supported image formats
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

print("\n=== AgriShield Dataset Inspection ===\n")

for class_name, folder_path in CLASS_FOLDERS.items():

    total_images = 0
    corrupted_images = []
    formats = Counter()
    dimensions = Counter()

    for file_path in folder_path.iterdir():

        # Ignore non-image files such as Annotation-export.csv
        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        total_images += 1

        try:
            with Image.open(file_path) as img:
                img.verify()

            # Reopen after verify() to read image information
            with Image.open(file_path) as img:
                formats[img.format] += 1
                dimensions[img.size] += 1

        except Exception as error:
            corrupted_images.append((file_path.name, str(error)))

    print(f"Class: {class_name}")
    print(f"Valid image files found: {total_images}")
    print(f"Corrupted images: {len(corrupted_images)}")
    print(f"Image formats: {dict(formats)}")

    print("Most common image dimensions:")
    for size, count in dimensions.most_common(5):
        print(f"  {size}: {count} images")

    if corrupted_images:
        print("\nCorrupted files:")
        for filename, error in corrupted_images:
            print(f"  {filename} -> {error}")

    print("-" * 50)

print("\nInspection complete.")