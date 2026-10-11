"""Download individually licensed banana-leaf photographs from Wikimedia Commons."""

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
SEARCH_TERMS = (
    "banana leaf close up filetype:bitmap",
    "banana leaf photograph filetype:bitmap",
    "banana leaves filetype:bitmap",
)
OUTPUT_DEFAULT = Path("validator_dataset/stage1_plant_leaves/banana_leaf")
MANIFEST_DEFAULT = Path(
    "validator_dataset/stage1_plant_leaves/banana_leaf_sources.json"
)
ACCEPTED_LICENSES = {
    "CC BY",
    "CC BY-SA",
    "CC0",
    "CC0 1.0",
}
EXCLUDED_TERMS = {
    "drawing",
    "dosa",
    "feast",
    "flower",
    "food",
    "girl",
    "heliconia",
    "man",
    "meal",
    "painting",
    "plate",
    "puris",
    "puto",
    "restaurant",
    "sambar",
    "satay",
    "chutney",
    "chicken",
    "spike",
    "idli",
}
MAX_TRANSFER_DEFAULT = 25 * 1024 * 1024
DEFAULT_TIMEOUT = 30.0
USER_AGENT = "AgriShield-FAW banana-leaf dataset acquisition/1.0"


class AcquisitionError(RuntimeError):
    """Raised when a candidate cannot be safely acquired."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download individually licensed banana-leaf photos from Commons."
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--max-transfer",
        type=int,
        default=MAX_TRANSFER_DEFAULT,
        help="Maximum total image bytes (default: 25 MiB).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def fetch_json(url: str, timeout: float) -> dict[str, object]:
    try:
        response = request.urlopen(
            request.Request(url, headers={"User-Agent": USER_AGENT}),
            timeout=timeout,
        )
        payload = json.loads(response.read().decode("utf-8"))
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"Could not read Commons API response: {exc}") from exc
    if not isinstance(payload, dict):
        raise AcquisitionError("Commons API returned a non-object response.")
    return payload


def api_url(params: dict[str, str]) -> str:
    return f"{API_URL}?{parse.urlencode(params)}"


def metadata_value(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, str):
            return raw
    return ""


def reusable_license(name: str) -> bool:
    normalized = " ".join(name.upper().replace("-", " ").split())
    return normalized == "CC0" or normalized.startswith(("CC BY", "CC BY SA"))


def find_candidates(count: int, timeout: float) -> list[dict[str, str | int]]:
    candidates: list[dict[str, str | int]] = []
    seen_urls: set[str] = set()
    for search_term in SEARCH_TERMS:
        continue_token = ""
        while len(candidates) < count:
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
            if continue_token:
                params["gsroffset"] = continue_token
            payload = fetch_json(api_url(params), timeout)
            query = payload.get("query")
            pages = query.get("pages", {}) if isinstance(query, dict) else {}
            if not isinstance(pages, dict):
                break
            for page in pages.values():
                if not isinstance(page, dict):
                    continue
                imageinfo = page.get("imageinfo")
                info = imageinfo[0] if isinstance(imageinfo, list) and imageinfo else {}
                if not isinstance(info, dict):
                    continue
                url = info.get("url")
                source_page = info.get("descriptionurl")
                metadata = info.get("extmetadata")
                if not isinstance(url, str) or not isinstance(source_page, str):
                    continue
                if not isinstance(metadata, dict):
                    continue
                license_name = metadata_value(metadata, "LicenseShortName")
                title = page.get("title")
                description = metadata_value(metadata, "ImageDescription")
                searchable = f"{title} {description}".lower()
                if (
                    not isinstance(title, str)
                    or not reusable_license(license_name)
                    or url in seen_urls
                    or "banana" not in searchable
                    or "leaf" not in searchable
                    or any(term in searchable for term in EXCLUDED_TERMS)
                ):
                    continue
                size = info.get("size")
                api_sha1 = info.get("sha1")
                if (
                    not isinstance(size, int)
                    or size <= 0
                    or not isinstance(api_sha1, str)
                    or len(api_sha1) != 40
                    or any(character not in "0123456789abcdef" for character in api_sha1.lower())
                ):
                    continue
                seen_urls.add(url)
                candidates.append(
                    {
                        "title": title,
                        "author": metadata_value(metadata, "Artist"),
                        "license": license_name,
                        "license_url": metadata_value(metadata, "LicenseUrl"),
                        "source_page": source_page,
                        "image_url": url,
                        "size": size,
                        "api_sha1": api_sha1.lower(),
                    }
                )
                if len(candidates) >= count:
                    break
            continuation = payload.get("continue")
            if len(candidates) >= count or not isinstance(continuation, dict):
                break
            offset = continuation.get("gsroffset")
            if not isinstance(offset, int):
                break
            continue_token = str(offset)
        if len(candidates) >= count:
            break
    if len(candidates) < count:
        raise AcquisitionError(
            f"Commons returned only {len(candidates)} acceptable licensed candidates."
        )
    return candidates


def verify_image(data: bytes, title: str) -> None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise AcquisitionError(f"Unreadable image for {title}: {exc}") from exc


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.count <= 0 or args.timeout <= 0 or args.max_transfer <= 0:
        print("ERROR: count, timeout, and max-transfer must be positive.", file=sys.stderr)
        return 2
    try:
        candidates = find_candidates(args.count * 5, args.timeout)
        candidates = sorted(candidates, key=lambda candidate: int(candidate["size"]))[
            : args.count
        ]
        estimated = sum(int(candidate["size"]) for candidate in candidates)
        if estimated > args.max_transfer:
            raise AcquisitionError(
                f"Estimated image transfer is {estimated} bytes, above the "
                f"{args.max_transfer}-byte limit."
            )
        print(f"Commons searches: {', '.join(SEARCH_TERMS)}")
        print(f"Selected {len(candidates)} individually licensed images.")
        print(f"Estimated image transfer: {estimated} bytes.")
        for candidate in candidates:
            print(
                f"PLAN {candidate['title']} | {candidate['license']} | "
                f"{candidate['size']} bytes | {candidate['image_url']}"
            )
        if args.dry_run:
            print("DRY RUN: no image data was downloaded.")
            return 0

        args.output.mkdir(parents=True, exist_ok=True)
        state_path = args.output / ".banana_download_state.json"
        state_images: dict[str, dict[str, str]] = {}
        if state_path.exists():
            try:
                state_payload = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AcquisitionError(
                    f"Could not read download state {state_path}: {exc}"
                ) from exc
            stored_images = (
                state_payload.get("images")
                if isinstance(state_payload, dict)
                else None
            )
            if not isinstance(stored_images, list):
                raise AcquisitionError(f"Invalid download state: {state_path}")
            for stored in stored_images:
                if isinstance(stored, dict) and isinstance(stored.get("image_url"), str):
                    state_images[stored["image_url"]] = {
                        str(key): str(value) for key, value in stored.items()
                    }

        hashes: set[str] = {
            image["sha256"] for image in state_images.values() if "sha256" in image
        }
        images: list[dict[str, str]] = []
        for index, candidate in enumerate(candidates, start=1):
            image_url = str(candidate["image_url"])
            resumed = state_images.get(image_url)
            if resumed is not None:
                destination = Path(resumed.get("local_path", ""))
                if (
                    destination.is_file()
                    and resumed.get("api_sha1") == str(candidate["api_sha1"])
                    and resumed.get("sha256")
                    and hashlib.sha256(destination.read_bytes()).hexdigest()
                    == resumed["sha256"]
                ):
                    images.append(resumed)
                    continue
                raise AcquisitionError(
                    f"Download state does not match an intact file: {destination}"
                )
            suffix = Path(parse.urlparse(image_url).path).suffix.lower()
            suffix = suffix if suffix in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
            temporary = args.output / f".{index:02d}_{candidate['api_sha1']}.part"
            try:
                response = request.urlopen(
                    request.Request(
                        image_url,
                        headers={"User-Agent": USER_AGENT},
                    ),
                    timeout=args.timeout,
                )
                data = response.read()
            except (OSError, error.URLError) as exc:
                raise AcquisitionError(
                    f"Could not download {candidate['title']}: {exc}"
                ) from exc
            if len(data) != int(candidate["size"]):
                raise AcquisitionError(
                    f"Downloaded size mismatch for {candidate['title']}: "
                    f"expected {candidate['size']}, got {len(data)}."
                )
            api_sha1 = hashlib.sha1(data).hexdigest()
            if api_sha1 != str(candidate["api_sha1"]):
                raise AcquisitionError(
                    f"API SHA-1 mismatch for {candidate['title']}: "
                    f"expected {candidate['api_sha1']}, got {api_sha1}."
                )
            verify_image(data, str(candidate["title"]))
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                raise AcquisitionError(f"Duplicate image content: {candidate['title']}")
            hashes.add(digest)
            filename = f"{index:02d}_{digest[:12]}{suffix}"
            destination = args.output / filename
            if destination.exists():
                raise AcquisitionError(f"Refusing to overwrite {destination}.")
            temporary.write_bytes(data)
            temporary.replace(destination)
            record = {
                "title": str(candidate["title"]),
                "author": str(candidate["author"]),
                "license": str(candidate["license"]),
                "license_url": str(candidate["license_url"]),
                "source_page": str(candidate["source_page"]),
                "image_url": image_url,
                "api_sha1": api_sha1,
                "local_path": str(destination),
                "sha256": digest,
            }
            images.append(record)
            state_images[image_url] = record
            atomic_write_json(state_path, {"images": list(state_images.values())})

        if args.manifest.exists():
            raise AcquisitionError(f"Refusing to overwrite existing manifest {args.manifest}.")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            args.manifest,
            {
                "source": "Wikimedia Commons",
                "search_terms": list(SEARCH_TERMS),
                "category": "banana_leaf",
                "images": images,
            },
        )
        if state_path.exists():
            state_path.unlink()
        print(f"Downloaded {len(images)} images to {args.output}.")
        return 0
    except AcquisitionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
