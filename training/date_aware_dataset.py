from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DateAwareImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.samples = []

        # ImageFolder-style structure:
        # root_dir/
        #   healthy/
        #   infected/

        for class_dir in sorted(self.root_dir.iterdir()):

            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            if class_name not in ["healthy", "infected"]:
                continue

            label = 0 if class_name == "healthy" else 1

            for image_path in sorted(class_dir.glob("*")):

                if image_path.suffix.lower() not in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:
                    continue

                # Date is first 8 characters of filename
                date_str = image_path.name[:8]

                if len(date_str) != 8 or not date_str.isdigit():
                    continue

                self.samples.append(
                    {
                        "path": image_path,
                        "label": label,
                        "class_name": class_name,
                        "date": date_str
                    }
                )

        print(f"Loaded {len(self.samples)} images from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "date": sample["date"],
            "path": str(sample["path"])
        }


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = DateAwareImageDataset(
        "data/images/processed/train",
        transform=transform
    )

    val_dataset = DateAwareImageDataset(
        "data/images/processed/val",
        transform=transform
    )

    test_dataset = DateAwareImageDataset(
        "data/images/processed/test",
        transform=transform
    )

    print("\nDATASET CHECK")
    print("-" * 40)

    print("Train :", len(train_dataset))
    print("Val   :", len(val_dataset))
    print("Test  :", len(test_dataset))

    sample = train_dataset[0]

    print("\nSAMPLE CHECK")
    print("-" * 40)

    print("Image :", sample["image"].shape)
    print("Label :", sample["label"])
    print("Date  :", sample["date"])
    print("Path  :", sample["path"])