from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


MODEL_NAME = "ViT-B-32"
PRETRAINED_WEIGHTS = "laion2b_s34b_b79k"

MAIZE_PROMPTS = (
    "a healthy maize leaf",
    "an infected maize leaf with visible pest or disease damage",
)

NON_MAIZE_PROMPTS = (
    "a leaf from an unrelated plant species",
    "a person",
    "a building or an urban structure",
    "a landscape or outdoor scenery",
    "a pet or other animal",
    "an unrelated everyday object",
)

logger = logging.getLogger(__name__)
_model_cache: tuple[Any, Any, Any] | None = None
_model_lock = threading.Lock()


class MaizeImageValidatorError(RuntimeError):
    """Raised when the maize validator cannot execute."""


@dataclass(frozen=True)
class MaizeValidationResult:
    accepted: bool
    maize_score: float
    non_maize_score: float
    score_margin: float
    model_name: str = MODEL_NAME
    pretrained_weights: str = PRETRAINED_WEIGHTS

    def as_log_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "maize_score": self.maize_score,
            "non_maize_score": self.non_maize_score,
            "score_margin": self.score_margin,
            "model_name": self.model_name,
            "pretrained_weights": self.pretrained_weights,
        }


def _load_model() -> tuple[Any, Any, Any]:
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    with _model_lock:
        if _model_cache is not None:
            return _model_cache

        try:
            previous_offline = {
                name: os.environ.get(name)
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                )
            }
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                import open_clip

                loaded = open_clip.create_model_and_transforms(
                    MODEL_NAME,
                    pretrained=PRETRAINED_WEIGHTS,
                    device="cpu",
                )
            finally:
                for name, value in previous_offline.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

            model, _, preprocess = loaded
            model.eval()
            tokenizer = open_clip.get_tokenizer(MODEL_NAME)
            text_tokens = tokenizer(
                list(MAIZE_PROMPTS + NON_MAIZE_PROMPTS)
            )
            with torch.inference_mode():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(
                    dim=-1,
                    keepdim=True,
                )

            _model_cache = (model, preprocess, text_features)
            return _model_cache
        except Exception as exc:
            raise MaizeImageValidatorError(
                "Maize image validator is unavailable."
            ) from exc


def validate_maize_image(image: Image.Image) -> MaizeValidationResult:
    """Classify an already-decoded image as maize or non-maize."""
    try:
        model, preprocess, text_features = _load_model()
        image_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

        with torch.inference_mode():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )
            scores = (
                image_features @ text_features.T
            ).squeeze(0).tolist()

        maize_score = max(scores[:len(MAIZE_PROMPTS)])
        non_maize_score = max(scores[len(MAIZE_PROMPTS):])
        result = MaizeValidationResult(
            accepted=maize_score > non_maize_score,
            maize_score=float(maize_score),
            non_maize_score=float(non_maize_score),
            score_margin=float(maize_score - non_maize_score),
        )
        logger.info("Maize image validation: %s", result.as_log_dict())
        return result
    except MaizeImageValidatorError:
        raise
    except Exception as exc:
        raise MaizeImageValidatorError(
            "Maize image validator failed during inference."
        ) from exc
