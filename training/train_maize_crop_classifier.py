"""Train the lightweight maize/non-maize identity gate.

Expected folders:
  data/maize_crop_classifier/{train,val}/{maize,non_maize}/

Use a separate test folder with crop species and images not used to train or
choose the threshold. The validation threshold targets a maximum non-maize
false-accept rate; it is not a guarantee for unseen species.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "maize_crop_classifier"
DEFAULT_OUTPUT = ROOT / "models" / "maize_crop_classifier.pth"
CLASS_TO_IDX = {"maize": 0, "non_maize": 1}
ARCHITECTURE = "mobilenet_v3_small"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-false-accept-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_datasets(data_dir: Path):
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    datasets_by_split = {
        "train": datasets.ImageFolder(data_dir / "train", transform=train_transform),
        "val": datasets.ImageFolder(data_dir / "val", transform=eval_transform),
    }
    for split, dataset in datasets_by_split.items():
        if dataset.class_to_idx != CLASS_TO_IDX:
            raise ValueError(
                f"{split} class folders must be maize and non_maize; "
                f"observed mapping: {dataset.class_to_idx}"
            )
        if len(dataset) == 0:
            raise ValueError(f"No images found in {data_dir / split}.")
        counts = {name: 0 for name in CLASS_TO_IDX}
        for _, label in dataset.samples:
            counts[dataset.classes[label]] += 1
        if min(counts.values()) == 0:
            raise ValueError(f"Both classes need images in {split}: {counts}")
        print(f"{split}: {len(dataset)} images; class counts={counts}")
    return datasets_by_split


def build_model(pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    final_layer = model.classifier[-1]
    model.classifier[-1] = nn.Linear(final_layer.in_features, 2)
    return model


def score_dataset(model: nn.Module, loader: DataLoader, device: torch.device):
    all_scores: list[float] = []
    all_labels: list[int] = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(device)), dim=1)[:, 0]
            all_scores.extend(probabilities.cpu().tolist())
            all_labels.extend(labels.tolist())
    return np.asarray(all_scores, dtype=float), np.asarray(all_labels, dtype=int)


def choose_threshold(scores: np.ndarray, labels: np.ndarray, max_far: float) -> tuple[float, dict[str, float]]:
    negative_scores = scores[labels == 1]
    positive_scores = scores[labels == 0]
    if len(negative_scores) == 0 or len(positive_scores) == 0:
        raise ValueError("Validation set must contain both classes.")

    candidates = sorted(set(negative_scores.tolist()) | {1.000001})
    valid = []
    for threshold in candidates:
        false_accept_rate = float(np.mean(negative_scores >= threshold))
        maize_accept_rate = float(np.mean(positive_scores >= threshold))
        if false_accept_rate <= max_far:
            valid.append((maize_accept_rate, -false_accept_rate, threshold, false_accept_rate))
    if not valid:
        threshold = 1.000001
        false_accept_rate = float(np.mean(negative_scores >= threshold))
        maize_accept_rate = float(np.mean(positive_scores >= threshold))
    else:
        maize_accept_rate, _, threshold, false_accept_rate = max(valid)
    metrics = {
        "validation_maize_accept_rate": float(np.mean(positive_scores >= threshold)),
        "validation_non_maize_false_accept_rate": float(np.mean(negative_scores >= threshold)),
        "target_max_false_accept_rate": float(max_far),
    }
    return float(threshold), metrics


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        print("ERROR: epochs/batch-size must be positive and learning-rate > 0.", file=sys.stderr)
        return 2
    if not 0 < args.max_false_accept_rate < 1:
        print("ERROR: max-false-accept-rate must be between 0 and 1.", file=sys.stderr)
        return 2
    if args.workers < 0:
        print("ERROR: workers cannot be negative.", file=sys.stderr)
        return 2

    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA was requested but is unavailable.", file=sys.stderr)
        return 2
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    data = make_datasets(args.data_dir)
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )
        for split, dataset in data.items()
    }

    print(f"Device: {device}. Loading pretrained MobileNetV3-Small weights if not cached.")
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(
        model.classifier[-1].parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Keep the frozen feature extractor in eval mode (no batch-stat drift).
        model.features.eval()
        running_loss = 0.0
        seen = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.inference_mode():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                loss = criterion(model(images), labels)
                val_loss_sum += float(loss.item()) * len(labels)
                val_count += len(labels)
        train_loss = running_loss / max(seen, 1)
        val_loss = val_loss_sum / max(val_count, 1)
        print(f"Epoch {epoch:02d}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    val_scores, val_labels = score_dataset(model, loaders["val"], device)
    threshold, threshold_metrics = choose_threshold(
        val_scores, val_labels, args.max_false_accept_rate
    )

    checkpoint = {
        "architecture": ARCHITECTURE,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "class_to_idx": CLASS_TO_IDX,
        "image_size": IMAGE_SIZE,
        "maize_threshold": threshold,
        "threshold_metrics": threshold_metrics,
        "validation_loss": best_val_loss,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, args.output)
    print(f"Saved checkpoint: {args.output}")
    print(f"Selected maize threshold: {threshold:.6f}")
    print(json.dumps(threshold_metrics, indent=2))
    print("Now evaluate against the separate test split before enabling the API gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
