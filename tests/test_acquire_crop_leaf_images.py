import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from training import acquire_crop_leaf_images as acquisition


def image_bytes(color: str = "green") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def candidate(index: int, data: bytes, license_url: str = "https://creativecommons.org/licenses/by/4.0/") -> dict[str, str | int]:
    return {
        "image_url": f"https://images.example/{index}.png",
        "source_url": f"https://commons.example/File:{index}",
        "commons_filename": f"leaf-{index}.png",
        "license": "CC BY 4.0",
        "license_url": license_url,
        "creator": "Test creator",
        "api_sha1": hashlib.sha1(data).hexdigest(),
        "advertised_size": len(data),
    }


class AcquisitionTests(unittest.TestCase):
    def test_duplicate_handling_and_count_reaches_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "stage1_plant_leaves"
            existing = output / "existing" / "existing.png"
            existing.parent.mkdir(parents=True)
            duplicate = image_bytes("green")
            existing.write_bytes(duplicate)
            payloads = [duplicate, image_bytes("red"), image_bytes("blue")]
            candidates = [candidate(i, data) for i, data in enumerate(payloads)]
            with mock.patch.object(acquisition, "find_candidates", return_value=candidates):
                with mock.patch.object(
                    acquisition,
                    "fetch_bytes",
                    side_effect=payloads,
                ):
                    count = acquisition.acquire_crop(
                        "rice_leaf", 2, 1, output, output, False, acquisition.existing_hashes(root)
                    )
            self.assertEqual(count, 2)

    def test_unreadable_image_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "stage1_plant_leaves"
            valid = image_bytes()
            candidates = [candidate(1, b"bad"), candidate(2, valid)]
            with mock.patch.object(acquisition, "find_candidates", return_value=candidates):
                with mock.patch.object(
                    acquisition, "fetch_bytes", side_effect=[b"not-an-image", valid]
                ):
                    count = acquisition.acquire_crop(
                        "rice_leaf", 1, 1, output, output, False, set()
                    )
            self.assertEqual(count, 1)

    def test_sha1_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "stage1_plant_leaves"
            valid = image_bytes()
            mismatched = candidate(1, valid)
            mismatched["api_sha1"] = "0" * 40
            with mock.patch.object(acquisition, "find_candidates", return_value=[mismatched]):
                with mock.patch.object(acquisition, "fetch_bytes", return_value=valid):
                    count = acquisition.acquire_crop(
                        "rice_leaf", 1, 1, output, output, False, set()
                    )
            self.assertEqual(count, 0)
            manifest = json.loads((output / "rice_leaf_sources.json").read_text())
            self.assertEqual(manifest["status"], "partial")

    def test_missing_license_url_is_rejected(self) -> None:
        page = {
            "title": "File:Rice leaf.jpg",
            "imageinfo": [
                {
                    "url": "https://images.example/rice.jpg",
                    "descriptionurl": "https://commons.example/File:Rice_leaf.jpg",
                    "size": 10,
                    "sha1": "a" * 40,
                    "extmetadata": {
                        "LicenseShortName": {"value": "CC BY 4.0"},
                        "LicenseUrl": {"value": ""},
                        "ImageDescription": {"value": "rice leaf"},
                    },
                }
            ],
        }
        self.assertIsNone(acquisition.extract_candidate("rice_leaf", page, "rice leaf"))

    def test_http_image_url_is_rejected(self) -> None:
        page = {
            "title": "File:Rice leaf.jpg",
            "imageinfo": [
                {
                    "url": "http://images.example/rice.jpg",
                    "descriptionurl": "https://commons.example/File:Rice_leaf.jpg",
                    "size": 10,
                    "sha1": "a" * 40,
                    "extmetadata": {
                        "LicenseShortName": {"value": "CC BY 4.0"},
                        "LicenseUrl": {
                            "value": "https://creativecommons.org/licenses/by/4.0/"
                        },
                        "ImageDescription": {"value": "rice leaf"},
                    },
                }
            ],
        }
        self.assertIsNone(acquisition.extract_candidate("rice_leaf", page, "rice leaf"))

    def test_dry_run_does_not_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "stage1_plant_leaves"
            candidates = [candidate(1, image_bytes())]
            with mock.patch.object(acquisition, "find_candidates", return_value=candidates):
                with mock.patch.object(acquisition, "fetch_bytes") as download:
                    count = acquisition.acquire_crop(
                        "rice_leaf", 1, 1, output, output, True, set()
                    )
            self.assertEqual(count, 0)
            download.assert_not_called()

    def test_partial_collection_is_not_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "stage1_plant_leaves"
            valid = image_bytes()
            candidates = [candidate(1, valid)]
            with mock.patch.object(acquisition, "find_candidates", return_value=candidates):
                with mock.patch.object(acquisition, "fetch_bytes", return_value=valid):
                    count = acquisition.acquire_crop(
                        "rice_leaf", 2, 1, output, output, False, set()
                    )
            self.assertEqual(count, 1)
            manifest = json.loads((output / "rice_leaf_sources.json").read_text())
            self.assertEqual(manifest["status"], "partial")
            self.assertEqual(manifest["shortfall"], 1)


if __name__ == "__main__":
    unittest.main()
