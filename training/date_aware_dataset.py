
from pathlib import Path
from datetime import datetime

from PIL import Image
from torch.utils.data import Dataset


class DateAwareImageDataset(Dataset):
    """
    Loads healthy and infected images.

    Expected structure:
        root_dir/
            healthy/
            infected/

    Labels:
        healthy = 0
        infected = 1

    If the filename starts with YYYYMMDD, that date is used.
    Otherwise, the date is recorded as None.
    """

    VALID_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    CLASS_LABELS = {
        "healthy": 0,
        "infected": 1,
    }

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        found_class_dirs = set()

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}"
            )

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name.lower()

            if class_name not in self.CLASS_LABELS:
                continue

            found_class_dirs.add(class_name)
            label = self.CLASS_LABELS[class_name]

            for image_path in sorted(class_dir.iterdir()):
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in self.VALID_EXTENSIONS:
                    continue

                # Extract date only when the filename begins with YYYYMMDD.
                date_str = None
                filename_prefix = image_path.name[:8]

                if (
                    len(filename_prefix) == 8
                    and filename_prefix.isdigit()
                ):
                    try:
                        datetime.strptime(filename_prefix, "%Y%m%d")
                        date_str = filename_prefix
                    except ValueError:
                        date_str = None

                self.samples.append({
                    "path": image_path,
                    "label": label,
                    "class_name": class_name,
                    "date": date_str,
                })

        missing_class_dirs = sorted(
            set(self.CLASS_LABELS) - found_class_dirs
        )

        if missing_class_dirs:
            missing = ", ".join(missing_class_dirs)
            raise FileNotFoundError(
                f"Dataset split {self.root_dir} is missing required "
                f"class directory/directories: {missing}. "
                "Expected both 'healthy' and 'infected' directories."
            )

        class_counts = {
            "healthy": 0,
            "infected": 0,
        }

        for sample in self.samples:
            class_counts[sample["class_name"]] += 1

        print(f"\nDataset: {self.root_dir}")
        print(f"Total images : {len(self.samples)}")
        print(f"Healthy      : {class_counts['healthy']}")
        print(f"Infected     : {class_counts['infected']}")

        empty_classes = [
            class_name
            for class_name, count in class_counts.items()
            if count == 0
        ]

        if empty_classes:
            empty = ", ".join(empty_classes)
            raise ValueError(
                f"Dataset split {self.root_dir} has no valid images "
                f"for class(es): {empty}. "
                "Check the class directories and supported image extensions."
            )

        if not self.samples:
            raise ValueError(
                f"No valid images found in {self.root_dir}. "
                "Check the folder structure and image extensions."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with Image.open(sample["path"]) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "date": sample["date"],
            "path": str(sample["path"]),
        }


if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    project_dir = Path(__file__).resolve().parent.parent
    data_dir = project_dir / "data" / "images" / "processed"

    for split in ["train", "val", "test"]:
        dataset = DateAwareImageDataset(
            data_dir / split,
            transform=transform,
        )

        print(f"{split}: {len(dataset)} images")

        if len(dataset) > 0:
            sample = dataset[0]
            print("Sample image :", sample["image"].shape)
            print("Sample label :", sample["label"])
            print("Sample date  :", sample["date"])
            print("Sample path  :", sample["path"])
