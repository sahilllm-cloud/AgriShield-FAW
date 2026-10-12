"""Stream a small maize/non-maize subset from PlantVillage into ImageFolder folders.

Uses geraldmc/plantvillage-full with streaming so it does not intentionally cache
the full 54k-image dataset. Saves up to 100 maize and 100 non-maize images by
default (60 train, 20 validation, 20 test per class). PlantVillage images have
controlled backgrounds; this is only a starter experiment, not field validation.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "maize_crop_classifier"
DATASET_ID = "geraldmc/plantvillage-full"
REVISION = "v0.1.0"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-maize-train", type=int, default=60)
    parser.add_argument("--max-negative-train", type=int, default=60)
    parser.add_argument("--max-maize-val", type=int, default=20)
    parser.add_argument("--max-negative-val", type=int, default=20)
    parser.add_argument("--max-maize-test", type=int, default=20)
    parser.add_argument("--max-negative-test", type=int, default=20)
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=2000,
        help="Streaming shuffle buffer; larger gives better mixing but uses more memory.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")[:100] or "leaf"


def save_image(row, destination: Path) -> str | None:
    image = row.get("image")
    if image is None:
        return None
    image = image.convert("RGB")
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.with_suffix(".jpg")
    image.save(destination, format="JPEG", quality=94)
    return digest


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    splits = {
        "train": (args.max_maize_train, args.max_negative_train),
        "val": (args.max_maize_val, args.max_negative_val),
        "test": (args.max_maize_test, args.max_negative_test),
    }
    if args.shuffle_buffer < 100:
        print("ERROR: --shuffle-buffer must be at least 100.", file=sys.stderr)
        return 2
    if any(maize <= 0 or non_maize <= 0 for maize, non_maize in splits.values()):
        print("ERROR: all requested image counts must be positive.", file=sys.stderr)
        return 2

    targets = {
        "maize": sum(pair[0] for pair in splits.values()),
        "non_maize": sum(pair[1] for pair in splits.values()),
    }
    selected = {"maize": [], "non_maize": []}
    seen_hashes: set[str] = set()

    print(f"Streaming {DATASET_ID} revision {REVISION}; full dataset is not explicitly materialized.")
    dataset = load_dataset(
        DATASET_ID,
        revision=REVISION,
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    for row in dataset:
        host = str(row.get("host", "")).strip().lower()
        if host in {"corn", "corn (maize)", "maize"}:
            label = "maize"
        elif host:
            label = "non_maize"
        else:
            continue

        if len(selected[label]) >= targets[label]:
            if all(len(selected[name]) >= targets[name] for name in targets):
                break
            continue

        image = row.get("image")
        if image is None:
            continue
        image = image.convert("RGB")
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        selected[label].append({
            "image": image,
            "host": str(row.get("host", "unknown")),
            "class_label": str(row.get("class_label", "leaf")),
            "digest": digest,
        })
        if all(len(selected[name]) >= targets[name] for name in targets):
            break

    for label in targets:
        if len(selected[label]) < targets[label]:
            print(
                f"ERROR: requested {targets[label]} {label} images but found "
                f"{len(selected[label])}. Check dataset access/host labels.",
                file=sys.stderr,
            )
            return 1

    # Clear only image files in the six folders owned by this script.
    for split in splits:
        for label in ("maize", "non_maize"):
            folder = args.output / split / label
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        path.unlink()

    cursor = {"maize": 0, "non_maize": 0}
    summary = {}
    for split, (maize_count, non_maize_count) in splits.items():
        split_summary = {}
        for label, count in (("maize", maize_count), ("non_maize", non_maize_count)):
            rows = selected[label][cursor[label]:cursor[label] + count]
            cursor[label] += count
            written = 0
            for i, row in enumerate(rows):
                filename = f"{i:04d}_{safe_name(row['host'])}_{safe_name(row['class_label'])}.jpg"
                destination = args.output / split / label / filename
                row["image"].save(destination, format="JPEG", quality=94)
                written += 1
            split_summary[label] = written
        summary[split] = split_summary
        print(f"{split}: {split_summary}")

    print(f"Done. Images saved under: {args.output}")
    print(f"Total images: maize={targets['maize']}, non_maize={targets['non_maize']}")
    print("Note: use the held-out test results cautiously; add field images before real-world claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
