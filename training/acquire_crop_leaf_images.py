"""Acquire individually licensed rice, wheat, and cotton leaf images.

The default mode downloads only individual Commons image URLs selected from
file metadata. Search snippets are treated as discovery only; every saved
image must have an explicit reusable license and is marked for visual review.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from urllib import error, parse, request

from PIL import Image, UnidentifiedImageError


API_URL = "https://commons.wikimedia.org/w/api.php"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "validator_dataset" / "stage1_plant_leaves"
DEFAULT_MANIFEST_ROOT = OUTPUT_ROOT
DEFAULT_TIMEOUT = 30.0
DEFAULT_COUNT = 20
USER_AGENT = "AgriShield-FAW crop-leaf acquisition/1.0"

SEARCHES = {
    "rice_leaf": (
        "incategory:Oryza leaves",
        "rice leaf photograph",
        "Oryza sativa leaf photograph",
    ),
    "wheat_leaf": (
        "incategory:Triticum leaves",
        "wheat leaf photograph",
        "Triticum aestivum leaf photograph",
    ),
    "cotton_leaf": (
        "incategory:Gossypium leaves",
        "cotton leaf photograph",
        "Gossypium leaf photograph",
    ),
}
CROP_TERMS = {
    "rice_leaf": ("rice", "oryza", "sativa"),
    "wheat_leaf": ("wheat", "triticum", "aestivum"),
    "cotton_leaf": ("cotton", "gossypium"),
}
EXCLUDED_TERMS = {
    "art",
    "artwork",
    "bottle gourd",
    "cake",
    "cartoon",
    "drawing",
    "flower",
    "food",
    "fruit",
    "germinated",
    "illustration",
    "insect",
    "meal",
    "painting",
    "poster",
    "rice dish",
    "screenshot",
    "seed",
    "sticker",
    "vegetable",
}


class AcquisitionError(RuntimeError):
    """Raised when a source response cannot be safely processed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire individually licensed crop-leaf images from Commons."
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="Directory receiving one metadata JSON file per crop.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search and validate metadata without downloading image bytes.",
    )
    return parser.parse_args()


def api_url(params: dict[str, str]) -> str:
    return f"{API_URL}?{parse.urlencode(params)}"


def fetch_json(url: str, timeout: float) -> dict[str, object]:
    try:
        with request.urlopen(
            request.Request(url, headers={"User-Agent": USER_AGENT}),
            timeout=timeout,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        if exc.code == 429:
            raise AcquisitionError(
                "Commons API rate limit (HTTP 429); no retry attempted."
            ) from exc
        raise AcquisitionError(f"Commons API HTTP error {exc.code}.") from exc
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"Could not read Commons API response: {exc}") from exc
    if not isinstance(payload, dict):
        raise AcquisitionError("Commons API returned a non-object response.")
    api_error = payload.get("error")
    if isinstance(api_error, dict):
        raise AcquisitionError(
            f"Commons API error: {api_error.get('info', 'unknown API error')}"
        )
    return payload


def fetch_bytes(url: str, timeout: float) -> bytes:
    try:
        with request.urlopen(
            request.Request(url, headers={"User-Agent": USER_AGENT}),
            timeout=timeout,
        ) as response:
            return response.read()
    except error.HTTPError as exc:
        if exc.code == 429:
            raise AcquisitionError(
                "Commons image request rate limit (HTTP 429); no retry attempted."
            ) from exc
        raise AcquisitionError(f"Commons image HTTP error {exc.code}.") from exc
    except (OSError, error.URLError) as exc:
        raise AcquisitionError(f"Could not download image: {exc}") from exc


def metadata_value(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, str):
            return " ".join(raw.split())
    return ""


def reusable_license(name: str) -> bool:
    normalized = " ".join(name.upper().replace("-", " ").split())
    return normalized == "CC0" or normalized.startswith(
        ("CC0 ", "CC BY ", "CC BY SA ")
    )


def valid_https_url(value: str) -> bool:
    parsed = parse.urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc)


def candidate_matches(crop: str, title: str, description: str) -> bool:
    searchable = f"{title} {description}".lower()
    if "leaf" not in searchable:
        return False
    if any(term in searchable for term in EXCLUDED_TERMS):
        return False
    return any(term in searchable for term in CROP_TERMS[crop])


def extract_candidate(
    crop: str, page: dict[str, object], search_term: str
) -> dict[str, str | int] | None:
    imageinfo = page.get("imageinfo")
    info = imageinfo[0] if isinstance(imageinfo, list) and imageinfo else {}
    if not isinstance(info, dict):
        return None
    metadata = info.get("extmetadata")
    if not isinstance(metadata, dict):
        return None
    title = page.get("title")
    image_url = info.get("url")
    source_url = info.get("descriptionurl")
    advertised_size = info.get("size")
    api_sha1 = info.get("sha1")
    license_name = metadata_value(metadata, "LicenseShortName")
    description = metadata_value(metadata, "ImageDescription")
    if (
        not isinstance(title, str)
        or not isinstance(image_url, str)
        or not isinstance(source_url, str)
        or not valid_https_url(image_url)
        or not valid_https_url(source_url)
        or not isinstance(advertised_size, int)
        or advertised_size <= 0
        or (
            api_sha1 is not None
            and (
                not isinstance(api_sha1, str)
                or len(api_sha1) != 40
                or any(character not in "0123456789abcdef" for character in api_sha1.lower())
            )
        )
        or not reusable_license(license_name)
        or not valid_https_url(metadata_value(metadata, "LicenseUrl"))
        or not candidate_matches(crop, title, description)
    ):
        return None
    return {
        "crop_category": crop,
        "commons_filename": title.removeprefix("File:"),
        "creator": metadata_value(metadata, "Artist"),
        "license": license_name,
        "license_url": metadata_value(metadata, "LicenseUrl"),
        "source_url": source_url,
        "image_url": image_url,
        "advertised_size": advertised_size,
        "api_sha1": api_sha1.lower() if isinstance(api_sha1, str) else "",
        "discovered_by": search_term,
        "manual_visual_verification_required": True,
    }


def find_candidates(
    crop: str, timeout: float, max_candidates: int | None = None
) -> list[dict[str, str | int]]:
    candidates: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for search_term in SEARCHES[crop]:
        offset = 0
        while True:
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": search_term,
                "gsrnamespace": "6",
                "gsrlimit": "50",
                "prop": "imageinfo",
                "iiprop": "url|size|sha1|extmetadata",
                "format": "json",
            }
            if offset:
                params["gsroffset"] = str(offset)
            payload = fetch_json(api_url(params), timeout)
            query = payload.get("query")
            pages = query.get("pages", {}) if isinstance(query, dict) else {}
            if not isinstance(pages, dict):
                break
            for page in pages.values():
                if not isinstance(page, dict):
                    continue
                candidate = extract_candidate(crop, page, search_term)
                if candidate is None:
                    continue
                identity = f"{candidate['image_url']}|{candidate['api_sha1']}"
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append(candidate)
                if max_candidates is not None and len(candidates) >= max_candidates:
                    break
            continuation = payload.get("continue")
            next_offset = continuation.get("gsroffset") if isinstance(continuation, dict) else None
            if (
                max_candidates is not None
                and len(candidates) >= max_candidates
            ) or not isinstance(next_offset, int):
                break
            if next_offset <= offset:
                break
            offset = next_offset
    return candidates if max_candidates is None else candidates[:max_candidates]


def existing_hashes(root: Path) -> set[str]:
    hashes: set[str] = set()
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes


def verify_image(data: bytes) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise AcquisitionError(f"Downloaded file is not a readable image: {exc}") from exc


def image_extension(url: str) -> str:
    suffix = Path(parse.urlparse(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"} else ".img"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def acquire_crop(
    crop: str,
    count: int,
    timeout: float,
    output_root: Path,
    manifest_root: Path,
    dry_run: bool,
    known_hashes: set[str],
) -> int:
    output_dir = output_root / crop
    manifest_path = manifest_root / f"{crop}_sources.json"
    result: dict[str, object] = {
        "dataset": "Wikimedia Commons",
        "crop_category": crop,
        "target_count": count,
        "status": "started",
        "images": [],
        "errors": [],
    }
    try:
        candidates = find_candidates(
            crop,
            timeout,
            max_candidates=count if dry_run else None,
        )
    except AcquisitionError as exc:
        message = str(exc)
        result["status"] = "rate_limit" if "429" in message else "network_error"
        result["errors"] = [message]
        write_json(manifest_path, result)
        print(f"{crop}: api_error: {exc}", file=sys.stderr)
        return 0

    if dry_run:
        result["status"] = "metadata_only"
        result["candidate_count"] = len(candidates)
        result["candidates"] = candidates
        write_json(manifest_path, result)
        print(f"{crop}: {len(candidates)} metadata candidates (dry run)")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        if isinstance(result["images"], list) and len(result["images"]) >= count:
            break
        try:
            data = fetch_bytes(str(candidate["image_url"]), timeout)
            verify_image(data)
        except AcquisitionError as exc:
            errors = result["errors"]
            if isinstance(errors, list):
                errors.append(f"{candidate['commons_filename']}: {exc}")
            if "429" in str(exc):
                break
            continue
        api_sha1 = str(candidate.get("api_sha1", ""))
        actual_sha1 = hashlib.sha1(data).hexdigest()
        if api_sha1 and actual_sha1 != api_sha1:
            errors = result["errors"]
            if isinstance(errors, list):
                errors.append(
                    f"{candidate['commons_filename']}: downloaded SHA-1 does not "
                    "match Commons metadata."
                )
            continue
        sha256 = hashlib.sha256(data).hexdigest()
        if sha256 in known_hashes:
            continue
        filename = (
            f"{len(result['images']) + 1:02d}_{sha256[:12]}"
            f"{image_extension(str(candidate['image_url']))}"
        )
        local_path = output_dir / filename
        if local_path.exists():
            continue
        local_path.write_bytes(data)
        known_hashes.add(sha256)
        images = result["images"]
        if isinstance(images, list):
            images.append(
                {
                    "local_filename": filename,
                    "original_source_url": candidate["source_url"],
                    "image_url": candidate["image_url"],
                    "commons_filename": candidate["commons_filename"],
                    "license": candidate["license"],
                    "license_url": candidate["license_url"],
                    "creator": candidate["creator"],
                    "file_size": len(data),
                    "sha256": sha256,
                    "api_sha1": candidate["api_sha1"],
                    "manual_visual_verification_required": True,
                }
            )
    result["verified_count"] = len(result["images"]) if isinstance(result["images"], list) else 0
    result["shortfall"] = max(0, count - int(result["verified_count"]))
    errors = result["errors"]
    if int(result["verified_count"]) >= count:
        result["status"] = "success"
    elif not candidates:
        result["status"] = "no_valid_candidates"
    elif isinstance(errors, list) and any(
        "rate limit" in str(error).lower() or "429" in str(error)
        for error in errors
    ):
        result["status"] = "rate_limit"
    elif (
        isinstance(errors, list)
        and errors
        and not result["verified_count"]
        and any(
            "http error" in str(error).lower()
            or "could not download" in str(error).lower()
            for error in errors
        )
    ):
        result["status"] = "network_error"
    else:
        result["status"] = "partial"
    write_json(manifest_path, result)
    print(f"{crop}: {result['verified_count']} verified image(s), shortfall {result['shortfall']}")
    return int(result["verified_count"])


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.timeout <= 0:
        print("ERROR: count and timeout must be positive.", file=sys.stderr)
        return 2
    known_hashes = existing_hashes(args.output_root.parent.parent)
    total = 0
    for crop in SEARCHES:
        total += acquire_crop(
            crop,
            args.count,
            args.timeout,
            args.output_root,
            args.manifest_root,
            args.dry_run,
            known_hashes,
        )
    print(f"Total verified images collected: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
