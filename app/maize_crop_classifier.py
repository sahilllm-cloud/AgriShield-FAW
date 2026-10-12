"""Lightweight, optional maize-vs-non-maize image gate.

The gate is disabled until a locally trained checkpoint is available. Enable it
with AGRISHIELD_MAIZE_CLASSIFIER_ENABLED=1 after training and evaluating the
checkpoint. It never replaces the existing CLIP validator or downstream models.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import models, transforms
from fastapi import HTTPException


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = BASE_DIR / "models" / "maize_crop_classifier.pth"
ARCHITECTURE = "mobilenet_v3_small"
IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.5
logger = logging.getLogger(__name__)
_model: torch.nn.Module | None = None
_threshold: float | None = None
_lock = threading.Lock()

_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class MaizeCropClassifierError(RuntimeError):
    """Raised when the optional crop-identity classifier cannot run."""


@dataclass(frozen=True)
class CropIdentityResult:
    accepted: bool
    maize_probability: float
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "maize_probability": self.maize_probability,
            "threshold": self.threshold,
        }


def classifier_enabled() -> bool:
    """Return whether the trained crop classifier gate is explicitly enabled."""
    return os.getenv("AGRISHIELD_MAIZE_CLASSIFIER_ENABLED", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _load_classifier() -> tuple[torch.nn.Module, float]:
    global _model, _threshold

    if _model is not None and _threshold is not None:
        return _model, _threshold

    with _lock:
        if _model is not None and _threshold is not None:
            return _model, _threshold

        if not CHECKPOINT_PATH.is_file():
            raise MaizeCropClassifierError(
                "Maize crop-classifier checkpoint is missing. Train "
                "training/train_maize_crop_classifier.py before enabling it."
            )

        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location="cpu",
                weights_only=True,
            )
            if checkpoint.get("architecture") != ARCHITECTURE:
                raise ValueError("Unexpected crop-classifier architecture.")
            if checkpoint.get("class_to_idx") != {"maize": 0, "non_maize": 1}:
                raise ValueError("Unexpected crop-classifier class mapping.")

            model = models.mobilenet_v3_small(weights=None)
            model.classifier[-1] = torch.nn.Linear(
                model.classifier[-1].in_features, 2
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            threshold = float(checkpoint["maize_threshold"])
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Invalid maize acceptance threshold.")
            model.eval()
        except Exception as exc:
            raise MaizeCropClassifierError(
                "Could not load the maize crop-classifier checkpoint."
            ) from exc

        _model = model
        _threshold = threshold
        return _model, _threshold


def predict_maize_identity(image: Image.Image) -> CropIdentityResult:
    """Return maize probability and the checkpoint-calibrated acceptance result."""
    try:
        model, threshold = _load_classifier()
        tensor = _PREPROCESS(image.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor), dim=1)[0]
        maize_probability = float(probabilities[0].item())
        result = CropIdentityResult(
            accepted=maize_probability >= threshold,
            maize_probability=maize_probability,
            threshold=threshold,
        )
        logger.info("Maize crop classifier: %s", result.as_dict())
        return result
    except MaizeCropClassifierError:
        raise
    except Exception as exc:
        raise MaizeCropClassifierError(
            "Maize crop-classifier inference failed."
        ) from exc


def enforce_maize_crop_identity(image: Image.Image, filename: str) -> None:
    """Enforce the optional gate, failing closed if it is enabled but unavailable."""
    if not classifier_enabled():
        return
    try:
        result = predict_maize_identity(image)
    except MaizeCropClassifierError as exc:
        logger.exception("Crop-identity classifier failed for %s", filename)
        raise HTTPException(
            status_code=503,
            detail="Maize crop validation is temporarily unavailable. Please try again.",
        ) from exc

    if not result.accepted:
        logger.info(
            "Rejected image before disease/risk inference filename=%s details=%s",
            filename,
            result.as_dict(),
        )
        raise HTTPException(
            status_code=422,
            detail=(
                "This image could not be verified as a maize leaf. "
                "Please upload a clear photo of a maize leaf."
            ),
        )
