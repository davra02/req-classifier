"""Download pinned sources into data/raw/ and verify their checksums.

Archives are never extracted to disk: parsers read members directly from the tar/zip.
"""

import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from reqcore.paths import data_dir

from reqbench.data.sources import SOURCES, Source


class ChecksumMismatchError(RuntimeError):
    pass


def raw_path(source: Source) -> Path:
    return data_dir() / "raw" / source.name / source.filename


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(source: Source, force: bool = False) -> Path:
    """Return the verified local copy of `source`, downloading it if needed."""
    path = raw_path(source)
    if path.exists() and not force and sha256_file(path) == source.sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(source.url, headers={"User-Agent": "reqbench/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    actual = sha256_file(tmp)
    if actual != source.sha256:
        tmp.unlink()
        raise ChecksumMismatchError(f"{source.name}: expected {source.sha256}, got {actual}")
    tmp.replace(path)
    return path


def download_all(force: bool = False) -> Path:
    """Fetch every source and write data/manifests/sources.json."""
    entries = []
    for source in SOURCES.values():
        path = fetch(source, force=force)
        entries.append(
            {
                "name": source.name,
                "url": source.url,
                "sha256": source.sha256,
                "bytes": path.stat().st_size,
                "license": source.license,
                "citation": source.citation,
                "verified_at": datetime.now(UTC).date().isoformat(),
            }
        )
    manifest = data_dir() / "manifests" / "sources.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    return manifest
