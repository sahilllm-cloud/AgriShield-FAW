"""Inventory and evaluate a labeled maize-versus-non-maize image manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


VALID_LABELS = {"maize", "non_maize"}
SUPPORTED_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory and evaluate a maize-image validation manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validator_dataset/maize_validation_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the inventory and evaluation report.",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Do not load OpenCLIP; only validate the manifest and image files.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read manifest {path}: {exc}") from exc

    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("categories"), list
    ):
        raise ValueError("Manifest must contain a categories list.")
    return manifest


def inspect_image(path: Path) -> str | None:
    if not path.exists():
        return "missing file"
    if not path.is_file():
        return "path is not a file"
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return f"unsupported image suffix: {path.suffix}"
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        return f"unreadable image: {exc}"
    return None


def inventory(
    root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"listed": 0, "readable": 0, "unreadable": 0}
    )

    for category in manifest["categories"]:
        name = category.get("name")
        label = category.get("label")
        paths = category.get("paths")
        if not isinstance(name, str) or not isinstance(label, str):
            errors.append({"path": "", "error": "category needs name and label"})
            continue
        if label not in VALID_LABELS:
            errors.append(
                {"path": name, "error": f"invalid label {label!r}"}
            )
        if not isinstance(paths, list):
            errors.append({"path": name, "error": "paths must be a list"})
            continue

        category_counts[name]["listed"] = len(paths)
        for relative_path in paths:
            if not isinstance(relative_path, str):
                errors.append({"path": name, "error": "image path is not a string"})
                continue
            image_path = root / Path(relative_path)
            error = inspect_image(image_path)
            record = {
                "path": relative_path,
                "category": name,
                "label": label,
                "readable": error is None,
            }
            entries.append(record)
            if error is None:
                category_counts[name]["readable"] += 1
            else:
                category_counts[name]["unreadable"] += 1
                errors.append({"path": relative_path, "error": error})

    required_missing = [
        category["name"]
        for category in manifest["categories"]
        if category.get("required") and not category.get("paths")
    ]
    report = {
        "category_counts": dict(sorted(category_counts.items())),
        "required_categories_missing_images": sorted(required_missing),
        "errors": errors,
        "totals": {
            "listed": len(entries),
            "readable": sum(entry["readable"] for entry in entries),
            "unreadable": sum(not entry["readable"] for entry in entries),
        },
    }
    return entries, report


def evaluate(root: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from app.maize_image_validator import (
        MaizeImageValidatorError,
        validate_maize_image,
    )

    results: list[dict[str, Any]] = []
    model_errors: list[dict[str, str]] = []
    for entry in entries:
        if not entry["readable"]:
            continue
        path = root / Path(entry["path"])
        try:
            with Image.open(path) as image:
                validation = validate_maize_image(image.convert("RGB"))
        except (OSError, UnidentifiedImageError, MaizeImageValidatorError) as exc:
            model_errors.append({"path": entry["path"], "error": str(exc)})
            continue
        results.append(
            {
                **entry,
                "accepted": validation.accepted,
                "maize_score": validation.maize_score,
                "non_maize_score": validation.non_maize_score,
                "score_margin": validation.score_margin,
            }
        )

    by_label: dict[str, dict[str, int]] = {}
    for label in sorted(VALID_LABELS):
        group = [result for result in results if result["label"] == label]
        accepted = sum(result["accepted"] for result in group)
        rejected = len(group) - accepted
        by_label[label] = {
            "total": len(group),
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": accepted / len(group) if group else None,
            "false_rejections": rejected if label == "maize" else 0,
            "false_acceptances": accepted if label == "non_maize" else 0,
        }

    return {
        "decision_rule": "best maize similarity > best non-maize similarity",
        "scores": "cosine image/text similarities; not calibrated probabilities",
        "by_label": by_label,
        "model_errors": model_errors,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    root = manifest_path.parent.parent
    try:
        manifest = load_manifest(manifest_path)
        entries, report = inventory(root, manifest)
        if not args.inventory_only:
            report["evaluation"] = evaluate(root, entries)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 1 if report["errors"] or report["required_categories_missing_images"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
