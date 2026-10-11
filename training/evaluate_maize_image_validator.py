"""Evaluate an OpenCLIP zero-shot maize-leaf image-validation prototype.

This script is intentionally independent of the production prediction API. It
reports cosine similarities between each image and the candidate text prompts;
the scores are not calibrated probabilities and are not used as a rejection
threshold.

Install the documented dependency before running:
    .\.venv\Scripts\python.exe -m pip install open_clip_torch

The first run downloads the selected checkpoint through OpenCLIP/Hugging Face.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, UnidentifiedImageError


MODEL_NAME = "ViT-B-32"
PRETRAINED_WEIGHTS = "laion2b_s34b_b79k"

CANDIDATE_LABELS = (
    "a healthy maize leaf",
    "an infected maize leaf with visible pest or disease damage",
    "a leaf from an unrelated plant species",
    "a person",
    "a building or an urban structure",
    "a landscape or outdoor scenery",
    "a pet or other animal",
    "an unrelated everyday object",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print OpenCLIP zero-shot similarity scores for image paths."
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="One or more local image paths to evaluate.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"OpenCLIP model name (default: {MODEL_NAME}).",
    )
    parser.add_argument(
        "--pretrained",
        default=PRETRAINED_WEIGHTS,
        help=(
            "OpenCLIP pretrained tag (default: "
            f"{PRETRAINED_WEIGHTS})."
        ),
    )
    return parser.parse_args()


def load_open_clip():
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "OpenCLIP is not installed. Install it with: "
            r".\.venv\Scripts\python.exe -m pip install open_clip_torch"
        ) from exc
    return open_clip


def readable_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.convert("RGB")
    except FileNotFoundError as exc:
        raise ValueError(f"file does not exist: {path}") from exc
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"unsupported or corrupt image: {path}") from exc


def score_image(
    model,
    preprocess,
    image_path: Path,
    text_features: torch.Tensor,
) -> list[tuple[str, float]]:
    image = readable_image(image_path)
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        image_features = model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        similarities = (image_features @ text_features.T).squeeze(0)

    return [
        (label, float(score))
        for label, score in zip(CANDIDATE_LABELS, similarities.tolist())
    ]


def evaluate(images: Sequence[Path], model_name: str, pretrained: str) -> int:
    open_clip = load_open_clip()
    print(f"Model: {model_name} ({pretrained})")
    print("Scores: cosine image/text similarity; not calibrated probabilities.")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device="cpu",
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    text_tokens = tokenizer(list(CANDIDATE_LABELS))
    with torch.inference_mode():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    failures = 0
    for image_path in images:
        print(f"\nImage: {image_path}")
        try:
            scores = score_image(
                model,
                preprocess,
                image_path,
                text_features,
            )
        except ValueError as exc:
            failures += 1
            print(f"ERROR: {exc}", file=sys.stderr)
            continue

        for label, score in sorted(scores, key=lambda item: item[1], reverse=True):
            print(f"  {score:+.6f}  {label}")

    return 1 if failures else 0


def main() -> int:
    args = parse_args()
    try:
        return evaluate(args.images, args.model, args.pretrained)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
