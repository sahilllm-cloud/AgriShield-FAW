"""Download a small, labelled maize/non-maize subset from PlantVillage into ImageFolder folders.

Source: geraldmc/plantvillage-full (PlantVillage images; dataset card lists CC0 1.0).
This script downloads only selected rows/images, not the entire dataset archive.
Non-maize species are split by species group so validation/test include crop types
not used as negative training examples. This is still a controlled-background dataset;
add field images before making broad real-world generalization claims.
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
# Keep some plant species entirely out of negative training.
TRAIN_NEGATIVE_HOSTS = {"Apple", "Grape", "Tomato", "Potato", "Pepper, bell"}
VAL_NEGATIVE_HOSTS = {"Orange", "Soybean", "Squash"}
TEST_NEGATIVE_HOSTS = {"Blueberry", "Cherry (including sour)", "Peach", "Raspberry", "Strawberry"}
MAX_MAIZE_PER_SPLIT = {"train": 1000, "val": 250, "test": 300}
MAX_NEGATIVE_PER_SPLIT = {"train": 1500, "val": 500, "test": 700}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-maize-train", type=int, default=1000)
    parser.add_argument("--max-negative-train", type=int, default=1500)
    parser.add_argument("--max-maize-val", type=int, default=250)
    parser.add_argument("--max-negative-val", type=int, default=500)
    parser.add_argument("--max-maize-test", type=int, default=300)
    parser.add_argument("--max-negative-test", type=int, default=700)
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned[:100] or "image"


def save_image(row, destination: Path, used_hashes: set[str]) -> bool:
    image = row.get("image")
    if image is None:
        return False
    image = image.convert("RGB")
    # Hash pixels so duplicate images cannot cross splits.
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    if digest in used_hashes:
        return False
    used_hashes.add(digest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination = destination.with_suffix(".jpg")
    image.save(destination, format="JPEG", quality=94)
    return True


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    caps = {
        "train": (args.max_maize_train, args.max_negative_train),
        "val": (args.max_maize_val, args.max_negative_val),
        "test": (args.max_maize_test, args.max_negative_test),
    }
    if any(a <= 0 or b <= 0 for a, b in caps.values()):
        print("ERROR: image caps must be positive.", file=sys.stderr)
        return 2

    print(f"Loading {DATASET_ID} revision {REVISION}. First run downloads dataset shards.")
    dataset = load_dataset(DATASET_ID, revision=REVISION, split="train")
    needed = {"image", "host", "class_label"}
    if not needed.issubset(set(dataset.column_names)):
        raise RuntimeError(f"Dataset fields changed; expected {needed}, got {dataset.column_names}")

    # Stable per-species allocation avoids putting the same non-maize species
    # in train and held-out splits. Maize itself is sampled separately per split.
    buckets = {split: {"maize": [], "non_maize": []} for split in caps}
    unknown_hosts = set()
    for row in dataset:
        host = str(row.get("host", "")).strip()
        if host.lower() in {"corn", "corn (maize)", "maize"}:
            buckets["train"]["maize"].append(row)
            # Repartition maize examples below, not by disease label.
            continue
        if host in TRAIN_NEGATIVE_HOSTS:
            buckets["train"]["non_maize"].append(row)
        elif host in VAL_NEGATIVE_HOSTS:
            buckets["val"]["non_maize"].append(row)
        elif host in TEST_NEGATIVE_HOSTS:
            buckets["test"]["non_maize"].append(row)
        else:
            unknown_hosts.add(host)

    maize_rows = buckets["train"]["maize"]
    random.shuffle(maize_rows)
    n_train, n_val, n_test = caps["train"][0], caps["val"][0], caps["test"][0]
    maize_splits = {
        "train": maize_rows[:n_train],
        "val": maize_rows[n_train:n_train + n_val],
        "test": maize_rows[n_train + n_val:n_train + n_val + n_test],
    }
    if not maize_splits["train"] or not maize_splits["val"] or not maize_splits["test"]:
        raise RuntimeError("Not enough maize images to fill all splits.")
    for split, rows in maize_splits.items():
        buckets[split]["maize"] = rows

    print("Dataset hosts not selected for this small starter subset:", ", ".join(sorted(unknown_hosts)))
    print("Non-maize host split: train=", sorted(TRAIN_NEGATIVE_HOSTS),
          "| val=", sorted(VAL_NEGATIVE_HOSTS), "| test=", sorted(TEST_NEGATIVE_HOSTS))

    # Remove only files we own within the six target class folders to make reruns deterministic.
    for split in caps:
        for label in ("maize", "non_maize"):
            folder = args.output / split / label
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                        path.unlink()

    used_hashes: set[str] = set()
    summary = {}
    for split, (maize_cap, negative_cap) in caps.items():
        split_summary = {}
        for label, cap in (("maize", maize_cap), ("non_maize", negative_cap)):
            rows = list(buckets[split][label])
            random.shuffle(rows)
            selected = rows[:cap]
            written = 0
            for index, row in enumerate(selected):
                class_label = safe_name(str(row.get("class_label", "leaf")))
                filename = f"{index:05d}_{class_label}.jpg"
                if save_image(row, args.output / split / label / filename, used_hashes):
                    written += 1
            split_summary[label] = {"available": len(rows), "written": written}
        summary[split] = split_summary
        print(f"{split}: {split_summary}")

    print(f"Images saved under: {args.output}")
    print("Review sample images and licenses/citation in the source dataset card before research publication.")
    print("This is a starter subset, not a field-diverse guarantee; add local field images and evaluate before enabling the gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
