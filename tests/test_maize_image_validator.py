import asyncio
import io
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers
import torch
from PIL import Image

from app import maize_image_validator
from app import hybrid_predictor


def _image():
    return Image.new("RGB", (8, 8), color="green")


def _upload():
    buffer = io.BytesIO()
    _image().save(buffer, format="PNG")
    buffer.seek(0)
    return UploadFile(
        file=buffer,
        filename="leaf.png",
        headers=Headers({"content-type": "image/png"}),
    )


def _fake_model(scores):
    model = Mock()
    model.encode_image.return_value = torch.tensor([scores], dtype=torch.float32)
    preprocess = Mock(return_value=torch.zeros(3, 2, 2))
    text_features = torch.eye(len(scores), dtype=torch.float32)
    return model, preprocess, text_features


class MaizeImageValidatorTests(unittest.TestCase):
    def test_maize_image_is_accepted_when_maize_score_is_higher(self):
        model, preprocess, text_features = _fake_model(
            [0.9, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        )

        with patch.object(
            maize_image_validator,
            "_load_model",
            return_value=(model, preprocess, text_features),
        ):
            result = maize_image_validator.validate_maize_image(_image())

        self.assertTrue(result.accepted)
        self.assertGreater(result.maize_score, result.non_maize_score)

    def test_non_maize_image_is_rejected_when_non_maize_score_is_higher(self):
        model, preprocess, text_features = _fake_model(
            [0.2, 0.1, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
        )

        with patch.object(
            maize_image_validator,
            "_load_model",
            return_value=(model, preprocess, text_features),
        ):
            result = maize_image_validator.validate_maize_image(_image())

        self.assertFalse(result.accepted)
        self.assertGreater(result.non_maize_score, result.maize_score)

    def test_validator_failure_is_not_converted_to_acceptance(self):
        with patch.object(
            maize_image_validator,
            "_load_model",
            side_effect=maize_image_validator.MaizeImageValidatorError(
                "unavailable"
            ),
        ):
            with self.assertRaises(
                maize_image_validator.MaizeImageValidatorError
            ):
                maize_image_validator.validate_maize_image(_image())

    def test_hybrid_gate_rejects_before_swin_inference(self):
        validation = maize_image_validator.MaizeValidationResult(
            accepted=False,
            maize_score=0.2,
            non_maize_score=0.9,
            score_margin=-0.7,
        )

        with patch.object(
            hybrid_predictor,
            "validate_maize_image",
            return_value=validation,
        ), patch.object(
            hybrid_predictor,
            "load_swin_model",
        ) as load_model:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    hybrid_predictor.extract_image_features(_upload())
                )

        self.assertEqual(raised.exception.status_code, 422)
        load_model.assert_not_called()

    def test_hybrid_gate_fails_closed_when_validator_is_unavailable(self):
        with patch.object(
            hybrid_predictor,
            "validate_maize_image",
            side_effect=maize_image_validator.MaizeImageValidatorError(
                "unavailable"
            ),
        ), patch.object(
            hybrid_predictor,
            "load_swin_model",
        ) as load_model:
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(
                    hybrid_predictor.extract_image_features(_upload())
                )

        self.assertEqual(raised.exception.status_code, 503)
        load_model.assert_not_called()
