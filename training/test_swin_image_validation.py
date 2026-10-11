
from pathlib import Path
from collections import Counter
import sys

import timm
import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = ROOT / "models" / "swin_best.pth"

TEST_FOLDERS = [
    ROOT / "validator_dataset" / "maize_test" / "maize_healthy",
    ROOT / "validator_dataset" / "maize_test" / "maize_infected",
    ROOT / "validator_dataset" / "non_maize",
]


# ============================================================
# IMAGE PREPROCESSING
# Matches app/hybrid_predictor.py
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD TEST IMAGES
# ============================================================

def collect_images():
    images = []

    for folder in TEST_FOLDERS:
        if not folder.exists():
            print(f"WARNING: Folder not found: {folder}")
            continue

        images.extend(
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    return sorted(images)


# ============================================================
# LOAD EXISTING SWIN MODEL
# ============================================================

def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Swin checkpoint not found: {CHECKPOINT}"
        )

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
    )

    model.head.fc = torch.nn.Linear(
        model.head.fc.in_features,
        2,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    model.load_state_dict(checkpoint)
    model.eval()

    return model


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(model, image_path):
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        features = model.forward_features(tensor)
        features = model.forward_head(features, pre_logits=True)
        logits = model.head.fc(features)
        scores = torch.softmax(logits, dim=1)[0]

    class_0_score = float(scores[0].item())
    class_1_score = float(scores[1].item())
    predicted_class = int(scores.argmax().item())

    return class_0_score, class_1_score, predicted_class


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print("=" * 65)
    print("AGRISHIELD-FAW — SWIN IMAGE VALIDATION EXPERIMENT")
    print("=" * 65)

    print("\nClass mapping from training code:")
    print("Class 0 = Healthy")
    print("Class 1 = Infected")

    print("\nLoading existing Swin checkpoint...")
    model = load_model()

    images = collect_images()

    if not images:
        raise RuntimeError(
            "No images found. Check validator_dataset folders."
        )

    print(f"\nImages found: {len(images)}")
    print("Running inference on CPU...")
    print("\nIndividual predictions:\n")

    counts = Counter()
    failures = []

    for index, path in enumerate(images, start=1):
        try:
            class_0, class_1, predicted_class = predict_image(
                model,
                path,
            )

        except Exception as exc:
            failures.append((path, str(exc)))
            print(f"[ERROR] {path.name}: {exc}")
            continue

        is_maize = "maize_test" in str(path).lower()
        group = "maize" if is_maize else "non_maize"

        counts[(group, predicted_class)] += 1
        counts[(group, "total")] += 1

        predicted_label = (
            "HEALTHY" if predicted_class == 0 else "INFECTED"
        )

        print(
            f"{index:02d}/{len(images)} | "
            f"{group.upper():8s} | "
            f"{path.name:15s} | "
            f"class_0={class_0:.4f} | "
            f"class_1={class_1:.4f} | "
            f"prediction={predicted_label}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    maize_total = counts[("maize", "total")]
    non_maize_total = counts[("non_maize", "total")]

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    print(f"Total images found: {len(images)}")
    print(f"Images evaluated: {maize_total + non_maize_total}")
    print(f"Images with errors: {len(failures)}")

    print("\nMAIZE IMAGES")
    print(f"Total: {maize_total}")
    print(
        "Predicted healthy:",
        counts[("maize", 0)],
    )
    print(
        "Predicted infected:",
        counts[("maize", 1)],
    )

    print("\nNON-MAIZE IMAGES")
    print(f"Total: {non_maize_total}")
    print(
        "Predicted healthy:",
        counts[("non_maize", 0)],
    )
    print(
        "Predicted infected:",
        counts[("non_maize", 1)],
    )

    if failures:
        print("\nFAILED IMAGES")
        for path, error in failures:
            print(f"{path}: {error}")

    print("\nIMPORTANT:")
    print(
        "Healthy/infected predictions are not maize/non-maize "
        "predictions."
    )
    print(
        "This experiment does not establish that Swin can validate "
        "whether an image contains maize."
    )
    print("Production code and model files were not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
