import unittest
from unittest.mock import patch

from fastapi import HTTPException
from PIL import Image

from app import maize_crop_classifier as classifier


class MaizeCropClassifierGateTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 16), "green")

    def test_gate_is_disabled_by_default_and_does_not_require_checkpoint(self):
        with patch.dict("os.environ", {}, clear=True), patch.object(
            classifier, "predict_maize_identity"
        ) as predict:
            classifier.enforce_maize_crop_identity(self.image, "leaf.png")
        predict.assert_not_called()

    def test_non_maize_prediction_is_rejected(self):
        result = classifier.CropIdentityResult(
            accepted=False, maize_probability=0.18, threshold=0.82
        )
        with patch.object(classifier, "classifier_enabled", return_value=True), patch.object(
            classifier, "predict_maize_identity", return_value=result
        ):
            with self.assertRaises(HTTPException) as raised:
                classifier.enforce_maize_crop_identity(self.image, "other-leaf.png")
        self.assertEqual(raised.exception.status_code, 422)

    def test_maize_prediction_is_allowed_to_continue(self):
        result = classifier.CropIdentityResult(
            accepted=True, maize_probability=0.96, threshold=0.82
        )
        with patch.object(classifier, "classifier_enabled", return_value=True), patch.object(
            classifier, "predict_maize_identity", return_value=result
        ):
            classifier.enforce_maize_crop_identity(self.image, "maize-leaf.png")

    def test_enabled_gate_fails_closed_when_checkpoint_is_unavailable(self):
        with patch.object(classifier, "classifier_enabled", return_value=True), patch.object(
            classifier,
            "predict_maize_identity",
            side_effect=classifier.MaizeCropClassifierError("checkpoint missing"),
        ):
            with self.assertRaises(HTTPException) as raised:
                classifier.enforce_maize_crop_identity(self.image, "leaf.png")
        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
