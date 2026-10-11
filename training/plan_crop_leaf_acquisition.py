"""Plan selective, metadata-only acquisition of crop-leaf images from Commons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib import error, parse, request


API_URL = "https://commons.wikimedia.org/w/api.php"
PLAN_DEFAULT = Path("validator_dataset/crop_leaf_acquisition_plan.json")
DEFAULT_TIMEOUT = 30.0
USER_AGENT = "AgriShield-FAW crop-leaf metadata planner/1.0"
TARGET_COUNT = 10

# These searches deliberately mix Commons category-oriented terms with precise
# file searches. Search text is only a discovery aid; it is not identity proof.
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


class PlanningError(RuntimeError):
    """Raised when Commons metadata cannot be safely processed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a metadata-only Commons crop-leaf acquisition plan."
    )
    parser.add_argument("--count", type=int, default=TARGET_COUNT)
    parser.add_argument("--plan", type=Path, default=PLAN_DEFAULT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly confirm metadata-only operation (the default behavior).",
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
            raise PlanningError("Commons API rate limit (HTTP 429); no retry attempted.") from exc
        raise PlanningError(f"Commons API HTTP error {exc.code}.") from exc
    except (OSError, error.URLError, json.JSONDecodeError) as exc:
        raise PlanningError(f"Could not read Commons API response: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlanningError("Commons API returned a non-object response.")
    api_error = payload.get("error")
    if isinstance(api_error, dict):
        info = api_error.get("info", "unknown API error")
        raise PlanningError(f"Commons API error: {info}")
    return payload


def metadata_value(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, str):
            return " ".join(raw.split())
    return ""


def reusable_license(name: str) -> bool:
    normalized = " ".join(name.upper().replace("-", " ").split())
    return normalized == "CC0" or normalized.startswith(("CC BY ", "CC BY SA "))


def candidate_matches(crop: str, title: str, description: str) -> bool:
    searchable = f"{title} {description}".lower()
    if any(term in searchable for term in EXCLUDED_TERMS):
        return False
    return any(term in searchable for term in CROP_TERMS[crop]) and "leaf" in searchable


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
    source_page = info.get("descriptionurl")
    size = info.get("size")
    api_sha1 = info.get("sha1")
    description = metadata_value(metadata, "ImageDescription")
    license_name = metadata_value(metadata, "LicenseShortName")
    if (
        not isinstance(title, str)
        or not isinstance(image_url, str)
        or not isinstance(source_page, str)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(api_sha1, str)
        or len(api_sha1) != 40
        or not reusable_license(license_name)
        or not candidate_matches(crop, title, description)
    ):
        return None
    return {
        "crop_category": crop,
        "commons_filename": title.removeprefix("File:"),
        "author": metadata_value(metadata, "Artist"),
        "license": license_name,
        "license_url": metadata_value(metadata, "LicenseUrl"),
        "source_page": source_page,
        "image_url": image_url,
        "advertised_size": size,
        "api_sha1": api_sha1.lower(),
        "discovered_by": search_term,
        "manual_visual_verification_required": True,
    }


def find_candidates(
    crop: str, count: int, timeout: float
) -> tuple[list[dict[str, str | int]], int]:
    candidates: list[dict[str, str | int]] = []
    seen: set[str] = set()
    requests_made = 0
    for search_term in SEARCHES[crop]:
        offset = 0
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
            if offset:
                params["gsroffset"] = str(offset)
            payload = fetch_json(api_url(params), timeout)
            requests_made += 1
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
                if len(candidates) >= count:
                    break
            continuation = payload.get("continue")
            if len(candidates) >= count or not isinstance(continuation, dict):
                break
            next_offset = continuation.get("gsroffset")
            if not isinstance(next_offset, int) or next_offset <= offset:
                break
            offset = next_offset
    return candidates[:count], requests_made


def write_plan(path: Path, plan: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.count <= 0 or args.timeout <= 0:
        print("ERROR: count and timeout must be positive.", file=sys.stderr)
        return 2

    categories: dict[str, object] = {}
    total_size = 0
    for crop in SEARCHES:
        try:
            candidates, requests_made = find_candidates(
                crop, args.count, args.timeout
            )
        except PlanningError as exc:
            categories[crop] = {
                "status": "api_error",
                "requested": args.count,
                "selected": 0,
                "shortfall": args.count,
                "metadata_requests": 0,
                "advertised_total_bytes": 0,
                "candidates": [],
                "error": str(exc),
                "availability_assessed": False,
            }
            print(f"ERROR [{crop}]: {exc}", file=sys.stderr)
            continue
        crop_size = sum(int(item["advertised_size"]) for item in candidates)
        total_size += crop_size
        categories[crop] = {
            "status": "completed_metadata_search",
            "requested": args.count,
            "selected": len(candidates),
            "shortfall": max(0, args.count - len(candidates)),
            "metadata_requests": requests_made,
            "advertised_total_bytes": crop_size,
            "candidates": candidates,
            "error": None,
            "availability_assessed": False,
        }

    plan = {
        "version": 1,
        "source": "Wikimedia Commons",
        "api_url": API_URL,
        "mode": "metadata-only",
        "manual_visual_verification_required": True,
        "categories": categories,
        "advertised_total_bytes": total_size,
    }
    write_plan(args.plan, plan)
    print(f"Plan written: {args.plan}")
    for crop, details in categories.items():
        assert isinstance(details, dict)
        print(
            f"{crop}: {details['status']}; "
            f"{details['selected']}/{details['requested']} selected; "
            f"{details['advertised_total_bytes']} bytes; "
            f"shortfall={details['shortfall']}"
        )
        candidates = details["candidates"]
        assert isinstance(candidates, list)
        for candidate in candidates:
            assert isinstance(candidate, dict)
            print(
                f"  {candidate['commons_filename']} | "
                f"{candidate['license']} | {candidate['advertised_size']} bytes"
            )
    print(f"Total advertised download size: {total_size} bytes")
    print("No image data was downloaded; all candidates require manual visual verification.")
    return 1 if any(
        isinstance(details, dict) and "error" in details
        for details in categories.values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
