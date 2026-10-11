"""Download a small tomato subset from PlantVillage without fetching data.zip."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image, UnidentifiedImageError


REPOSITORY_URL = "https://huggingface.co/datasets/mohanty/PlantVillage"
ARCHIVE_URL = f"{REPOSITORY_URL}/resolve/main/data.zip"
LICENSE = "CC BY-SA 3.0"
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_TRANSFER = 50 * 1024 * 1024
EOCD_SIGNATURE = b"PK\x05\x06"
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"


class DownloadError(RuntimeError):
    """Raised when selective retrieval cannot be completed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selectively download tomato images from PlantVillage."
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validator_dataset/stage1_plant_leaves/tomato_leaf"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "validator_dataset/stage1_plant_leaves/tomato_leaf_sources.json"
        ),
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--max-transfer",
        type=int,
        default=DEFAULT_MAX_TRANSFER,
        help="Maximum total response bytes, including metadata and image data.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan retrieval and estimate bytes without downloading image data.",
    )
    return parser.parse_args()


def split_url(split: str) -> str:
    return f"{REPOSITORY_URL}/raw/main/splits/color_{split}.txt"


def request_bytes(
    session: Any,
    url: str,
    timeout: float,
    *,
    byte_range: tuple[int, int] | None = None,
) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    try:
        response = request.urlopen(
            request.Request(url, headers=headers), timeout=timeout
        )
        status = response.status
        content = response.read()
    except (OSError, error.URLError) as exc:
        raise DownloadError(f"Request failed for {url}: {exc}") from exc

    if byte_range is not None:
        if status != 206:
            raise DownloadError(
                f"Range request was not honored for {url} "
                f"(HTTP {status}); stopping safely."
            )
        expected = byte_range[1] - byte_range[0] + 1
        if len(content) != expected:
            raise DownloadError(
                f"Range response for {url} returned {len(content)} "
                f"bytes; expected {expected}."
            )
    return content


def check_range_support(
    session: Any, timeout: float
) -> tuple[int, int]:
    try:
        response = request.urlopen(
            request.Request(ARCHIVE_URL, method="HEAD"),
            timeout=timeout,
        )
    except (OSError, error.URLError) as exc:
        raise DownloadError(f"Could not inspect archive headers: {exc}") from exc

    accept_ranges = response.headers.get("Accept-Ranges", "").lower()
    content_length = response.headers.get("Content-Length")
    if accept_ranges != "bytes" or not content_length:
        raise DownloadError(
            "The archive does not confirm byte-range support and was not downloaded."
        )
    try:
        size = int(content_length)
    except ValueError as exc:
        raise DownloadError("Archive Content-Length is invalid.") from exc
    if size <= 0:
        raise DownloadError("Archive Content-Length is not positive.")

    # A one-byte probe confirms that the redirected storage endpoint honors Range.
    probe = request_bytes(session, ARCHIVE_URL, timeout, byte_range=(0, 0))
    if len(probe) != 1:
        raise DownloadError("Archive range probe returned an unexpected size.")
    return size, 1


def read_tomato_paths(
    session: requests.Session, split: str, timeout: float
) -> tuple[list[str], int]:
    data = request_bytes(session, split_url(split), timeout)
    paths: list[str] = []
    seen: set[str] = set()
    for raw_line in data.decode("utf-8").splitlines():
        path = raw_line.strip()
        if (
            path.startswith("raw/color/Tomato___")
            and path not in seen
            and "/" in path
        ):
            seen.add(path)
            paths.append(path)
    if not paths:
        raise DownloadError(f"No tomato image paths found in the {split} split.")
    return paths, len(data)


def parse_central_directory(
    data: bytes, entries_total: int
) -> dict[str, tuple[int, int, int]]:
    if len(data) < 46:
        raise DownloadError("ZIP central-directory data is incomplete.")
    cursor = 0
    records: dict[str, tuple[int, int, int]] = {}
    for _ in range(entries_total):
        if data[cursor : cursor + 4] != CENTRAL_SIGNATURE:
            raise DownloadError("Invalid ZIP central-directory entry.")
        if len(data) - cursor < 46:
            raise DownloadError("Truncated ZIP central-directory entry.")
        fields = struct.unpack_from("<4s6H3L5H2L", data, cursor)
        flags = fields[3]
        compression = fields[4]
        compressed_size = fields[8]
        filename_size = fields[10]
        extra_size = fields[11]
        comment_size = fields[12]
        local_offset = fields[16]
        start = cursor + 46
        end = start + filename_size
        name = data[start:end].decode("utf-8")
        cursor = end + extra_size + comment_size
        if flags & 0x08 or compressed_size == 0xFFFFFFFF or local_offset == 0xFFFFFFFF:
            raise DownloadError(f"Unsupported ZIP entry format for {name}.")
        records[name] = (local_offset, compressed_size, compression)
    return records


def parse_end_of_central_directory(
    data: bytes, archive_size: int
) -> tuple[int, int, int]:
    eocd_offset = data.rfind(EOCD_SIGNATURE)
    if eocd_offset < 0 or len(data) - eocd_offset < 22:
        raise DownloadError("Could not locate the ZIP end-of-central-directory record.")
    fields = struct.unpack_from("<4s4H2LH", data, eocd_offset)
    _, disk, disk_start, entries_disk, entries_total, directory_size, directory_offset, _ = fields
    if (
        entries_total == 0xFFFF
        or entries_disk == 0xFFFF
        or directory_offset == 0xFFFFFFFF
        or directory_size == 0xFFFFFFFF
    ):
        zip64_offset = data.rfind(ZIP64_EOCD_SIGNATURE, 0, eocd_offset)
        if zip64_offset < 0 or len(data) - zip64_offset < 56:
            raise DownloadError("ZIP64 central-directory metadata is unavailable.")
        zip64 = struct.unpack_from("<4sQ2H2L4Q", data, zip64_offset)
        (
            _, _, _, _, zip64_disk, zip64_start, zip64_entries_disk,
            zip64_entries_total, directory_size, directory_offset,
        ) = zip64
        disk = zip64_disk
        disk_start = zip64_start
        entries_disk = zip64_entries_disk
        entries_total = zip64_entries_total
    if disk != 0 or disk_start != 0 or entries_disk != entries_total:
        raise DownloadError("Multi-disk ZIP archives are not supported.")
    if directory_offset + directory_size > archive_size:
        raise DownloadError("ZIP central-directory bounds are invalid.")
    return entries_total, directory_offset, directory_size


def read_archive_records(
    session: requests.Session,
    archive_size: int,
    selected: list[str],
    timeout: float,
) -> tuple[dict[str, tuple[int, int, int]], int]:
    tail_size = min(65557, archive_size)
    tail_start = archive_size - tail_size
    tail = request_bytes(
        session, ARCHIVE_URL, timeout, byte_range=(tail_start, archive_size - 1)
    )
    entries_total, directory_offset, directory_size = parse_end_of_central_directory(
        tail, archive_size
    )
    directory_end = directory_offset + directory_size - 1
    if directory_offset >= tail_start and directory_end < archive_size:
        directory = tail[directory_offset - tail_start : directory_end - tail_start + 1]
    else:
        directory = request_bytes(
            session,
            ARCHIVE_URL,
            timeout,
            byte_range=(directory_offset, directory_end),
        )
    records = parse_central_directory(directory, entries_total)
    missing = [path for path in selected if path not in records]
    if missing:
        raise DownloadError(f"Selected paths are absent from the archive: {missing}")
    return {path: records[path] for path in selected}, tail_size + directory_size


def image_data(
    session: requests.Session,
    record: tuple[int, int, int],
    timeout: float,
) -> tuple[bytes, int]:
    local_offset, compressed_size, compression = record
    header = request_bytes(
        session,
        ARCHIVE_URL,
        timeout,
        byte_range=(local_offset, local_offset + 29),
    )
    if header[:4] != LOCAL_SIGNATURE:
        raise DownloadError("Invalid ZIP local-file header.")
    local_fields = struct.unpack("<4s5H3L2H", header)
    flags = local_fields[2]
    method = local_fields[3]
    filename_size = local_fields[9]
    extra_size = local_fields[10]
    if flags & 0x08 or method != compression:
        raise DownloadError("Unsupported ZIP entry flags or compression method.")
    data_start = local_offset + 30 + filename_size + extra_size
    compressed = request_bytes(
        session,
        ARCHIVE_URL,
        timeout,
        byte_range=(data_start, data_start + compressed_size - 1),
    )
    if compression == 0:
        return compressed, 30 + filename_size + extra_size + compressed_size
    if compression == 8:
        import zlib

        try:
            return zlib.decompress(compressed, -15), (
                30 + filename_size + extra_size + compressed_size
            )
        except zlib.error as exc:
            raise DownloadError("Could not decompress ZIP image data.") from exc
    raise DownloadError(f"Unsupported ZIP compression method: {compression}.")


def safe_filename(index: int, archive_path: str) -> str:
    suffix = Path(archive_path).suffix.lower() or ".jpg"
    return f"{index:02d}_{hashlib.sha256(archive_path.encode()).hexdigest()[:12]}{suffix}"


def main() -> int:
    args = parse_args()
    if args.count <= 0 or args.max_transfer <= 0 or args.timeout <= 0:
        print("ERROR: count, max-transfer, and timeout must be positive.", file=sys.stderr)
        return 2

    session = None
    try:
        archive_size, probe_bytes = check_range_support(session, args.timeout)
        paths, split_bytes = read_tomato_paths(session, args.split, args.timeout)
        selected = paths[: args.count]
        records, archive_metadata_bytes = read_archive_records(
            session, archive_size, selected, args.timeout
        )
        image_transfer = sum(30 + len(Path(path).name.encode()) + size for path, (_, size, _) in records.items())
        estimated = probe_bytes + split_bytes + archive_metadata_bytes + image_transfer
        if estimated > args.max_transfer:
            raise DownloadError(
                f"Estimated transfer is {estimated} bytes, above the "
                f"{args.max_transfer}-byte limit."
            )
        print(f"Selected {len(selected)} unique tomato paths from {args.split}.")
        print(f"Estimated maximum transfer: {estimated} bytes.")
        if args.dry_run:
            for path in selected:
                print(f"PLAN {path} -> {safe_filename(selected.index(path) + 1, path)}")
            return 0

        args.output.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        hashes: set[str] = set()
        for index, archive_path in enumerate(selected, start=1):
            data, _ = image_data(session, records[archive_path], args.timeout)
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise DownloadError(f"Downloaded image is unreadable: {archive_path}") from exc
            digest = hashlib.sha256(data).hexdigest()
            if digest in hashes:
                raise DownloadError(f"Duplicate image content detected: {archive_path}")
            hashes.add(digest)
            filename = safe_filename(index, archive_path)
            destination = args.output / filename
            if destination.exists():
                raise DownloadError(f"Refusing to overwrite existing file: {destination}")
            destination.write_bytes(data)
            results.append({
                "local_path": str(destination),
                "original_path": archive_path,
                "label": archive_path.split("/")[2],
                "source_url": ARCHIVE_URL,
                "license": LICENSE,
                "sha256": digest,
            })
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "dataset": "mohanty/PlantVillage",
                    "license": LICENSE,
                    "split": args.split,
                    "archive_url": ARCHIVE_URL,
                    "split_url": split_url(args.split),
                    "images": results,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Downloaded {len(results)} images to {args.output}.")
        return 0
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
