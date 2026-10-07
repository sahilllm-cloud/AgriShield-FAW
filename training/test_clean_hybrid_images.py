from pathlib import Path

import torch
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = ROOT / "data" / "images" / "processed" / "test"

MODEL_PATH = ROOT / "models" / "swin_best.pth"


# ============================================================
# LOAD SWIN
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

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


# ============================================================
# TEST DATA
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)


# ============================================================
# FIND TWO HEALTHY + TWO INFECTED
# ============================================================

healthy_indices = [
    i
    for i, (_, label) in enumerate(dataset.samples)
    if dataset.classes[label] == "healthy"
]

infected_indices = [
    i
    for i, (_, label) in enumerate(dataset.samples)
    if dataset.classes[label] == "infected"
]

selected_indices = (
    healthy_indices[:2]
    + infected_indices[:2]
)


# ============================================================
# GET SWIN PROBABILITIES
# ============================================================

print("\n========================================")
print("SWIN TEST IMAGE RESULTS")
print("========================================")

with torch.no_grad():

    for index in selected_indices:

        image, label = dataset[index]

        image = image.unsqueeze(0).to(device)

        output = model(image)

        probability = torch.softmax(
            output,
            dim=1
        )[0, 1].item()

        print("\nImage:")
        print(Path(dataset.samples[index][0]).name)

        print("Class:")
        print(dataset.classes[label])

        print(
            "Swin infected probability:",
            f"{probability:.6f}"
        )


print("\n========================================")
print("Swin test completed.")
print("========================================")