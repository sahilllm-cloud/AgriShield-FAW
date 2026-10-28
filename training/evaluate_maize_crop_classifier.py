
"""Evaluate the maize/non-maize classifier and report misclassified filenames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "models" / "maize_crop_classifier.pth"
TEST_DIR = ROOT / "data" / "maize_crop_classifier" / "test"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    dataset = datasets.ImageFolder(args.test_dir, transform=transform)
    expected = {"maize": 0, "non_maize": 1}

    if dataset.class_to_idx != expected:
        raise ValueError(
            f"Expected class folders {expected}, got {dataset.class_to_idx}"
        )
    if not dataset.samples:
        raise ValueError(f"No test images found under {args.test_dir}")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    if checkpoint.get("class_to_idx") != expected:
        raise ValueError("Checkpoint class mapping does not match the test set.")

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, 2
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    scores, labels = [], []

    with torch.inference_mode():
        for images, batch_labels in loader:
            probabilities = torch.softmax(model(images), dim=1)[:, 0]
            scores.extend(probabilities.tolist())
            labels.extend(batch_labels.tolist())

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    threshold = float(checkpoint["maize_threshold"])
    accepted = scores >= threshold
    predicted_maize = scores >= 0.5

    maize = labels == 0
    non_maize = labels == 1

    results = {
        "test_images": int(len(labels)),
        "maize_images": int(maize.sum()),
        "non_maize_images": int(non_maize.sum()),
        "checkpoint_threshold": threshold,
        "maize_accept_rate": (
            float(accepted[maize].mean()) if maize.any() else None
        ),
        "non_maize_false_accept_rate": (
            float(accepted[non_maize].mean()) if non_maize.any() else None
        ),
        "accuracy_at_probability_0_5": float(
            np.mean(predicted_maize == (labels == 0))
        ),
    }

    print("\n=== TEST METRICS ===")
    print(json.dumps(results, indent=2))

    print("\n=== MISCLASSIFIED IMAGES AT CHECKPOINT THRESHOLD ===")
    errors = 0

    for index, (path, true_label) in enumerate(dataset.samples):
        accepted_as_maize = bool(accepted[index])
        actual_label = dataset.classes[true_label]

        # At the operational threshold, maize is accepted and
        # non-maize is rejected.
        incorrect = (
            (actual_label == "non_maize" and accepted_as_maize)
            or (actual_label == "maize" and not accepted_as_maize)
        )

        if incorrect:
            errors += 1
            relative_path = Path(path).relative_to(args.test_dir)
            print(
                f"File: {relative_path} | "
                f"Actual: {actual_label} | "
                f"Maize probability: {scores[index]:.6f} | "
                f"Threshold: {threshold:.6f} | "
                f"Accepted: {accepted_as_maize}"
            )

    if errors == 0:
        print("No errors at the checkpoint threshold.")
    else:
        print(f"\nTotal errors at checkpoint threshold: {errors}")

    print("\nNote: a small test set cannot establish real-world reliability.")
    print("Keep the API gate disabled until results are reviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
