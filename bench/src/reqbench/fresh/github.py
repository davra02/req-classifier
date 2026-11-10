"""GitHub access through the authenticated `gh` CLI, with an on-disk snapshot.

The search results drift over time, so the first run freezes them in
`data/raw/fresh/snapshot.json` (repository metadata, HEAD commit and file list). Blobs are
cached by their git SHA, which makes them immutable. Later runs read only the snapshot and the
blob cache, so the preview and the `--write` run select exactly the same requirements.
Downloaded content is untrusted data: it is only parsed with regular expressions.
"""

import json
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reqcore.paths import data_dir

CREATED_SINCE = "2025-01-01"
# GitHub license keys -> SPDX IDs. CC BY-SA and similar share-alike licenses are not accepted.
ALLOWED_LICENSES = {
    "mit": "MIT",
    "apache-2.0": "Apache-2.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc0-1.0": "CC0-1.0",
}
QUERIES = (
    '"software requirements specification" in:name,description',
    '"requirements specification" in:name,description',
    '"software requirements" in:name,description',
    '"system requirements" in:name,description',
    "srs in:name",
)
# Code search finds SRS files in repositories named after the product, which the repository
# search misses. It has no date or license qualifiers: those are checked on the repository.
CODE_QUERIES = (
    'shall "functional requirements" filename:SRS.md',
    'shall "functional requirements" filename:requirements.md',
    'shall "non-functional requirements" path:docs extension:md',
    "shall filename:SRS.txt",
)
SEARCH_PAUSE_S = 2.2  # the search API allows 30 requests per minute
CODE_SEARCH_PAUSE_S = 6.5  # code search allows 10 requests per minute
MAX_SEARCH_PAGES = 10  # GitHub returns at most 1000 results per query
WORKERS = 8
RATE_LIMIT_WAIT_S = 65 * 60  # the core limit (5000 requests) resets every hour
SAVE_EVERY = 100

DOC_EXTENSIONS = (".md", ".markdown", ".txt")
DOC_PATH = re.compile(r"srs|requirement|specification|spec\b|specs/", re.IGNORECASE)
# Third-party copies, blogs, test fixtures, examples, templates and AI-agent instructions are
# not the project's own requirements (or are machine-written).
SKIP_PATH = re.compile(
    r"(^|/)(node_modules|vendor|3rd ?party|third[_-]?party|\.github|site-packages|dist|build|"
    r"blogs?|posts|_?_?fixtures?_?_?|examples?|samples?|tests?|[^/]*templates?[^/]*|prompts?|skills|"
    r"\.agents?|\.claude|\.cursor|\.kiro|ai-sessions)/|"
    r"(^|/)(changelog|license|code_of_conduct|contributing|security|skill|agents|claude|"
    r"gemini|copilot-instructions)[^/]*$",
    re.IGNORECASE,
)
MAX_FILE_BYTES = 400_000
MAX_FILES_PER_REPO = 20


def fresh_raw_dir() -> Path:
    return data_dir() / "raw" / "fresh"


def snapshot_path() -> Path:
    return fresh_raw_dir() / "snapshot.json"


class RateLimitedError(RuntimeError):
    pass


def gh_api(endpoint: str, *args: str, raw: bool = False) -> Any:
    """Call `gh api`; when the rate limit is hit, wait and retry (up to RATE_LIMIT_WAIT_S)."""
    command = ["gh", "api", endpoint, *args]
    if raw:
        command[2:2] = ["-H", "Accept: application/vnd.github.raw"]
    waited = 0
    while True:
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode == 0:
            return result.stdout if raw else json.loads(result.stdout)
        message = result.stderr.decode(errors="replace")
        if "rate limit" not in message.lower():
            raise RuntimeError(f"gh api {endpoint}: {message[:300]}")
        if waited >= RATE_LIMIT_WAIT_S:
            raise RateLimitedError(f"gh api {endpoint}: still rate limited after {waited} s")
        if waited == 0:
            print(f"GitHub rate limit hit; waiting up to {RATE_LIMIT_WAIT_S // 60} min.",
                  file=sys.stderr, flush=True)  # fmt: skip
        time.sleep(60)
        waited += 60


def search_repositories() -> list[dict]:
    """Union of all repository queries x licenses and code queries, deduplicated by ID."""
    found: dict[int, dict] = {}
    for query in QUERIES:
        for key in ALLOWED_LICENSES:
            q = f"{query} created:>={CREATED_SINCE} license:{key} fork:false archived:false"
            for page in range(1, MAX_SEARCH_PAGES + 1):
                time.sleep(SEARCH_PAUSE_S)
                result = gh_api(
                    "search/repositories", "-X", "GET", "-f", f"q={q}",
                    "-f", "per_page=100", "-f", f"page={page}", "-f", "sort=created",
                    "-f", "order=asc",
                )  # fmt: skip
                for item in result["items"]:
                    entry = found.setdefault(item["id"], _repo_entry(item))
                    entry["queries"] = sorted({*entry["queries"], query})
                if len(result["items"]) < 100:
                    break
    by_name = {r["full_name"]: r for r in found.values()}
    for query in CODE_QUERIES:
        for page in range(1, MAX_SEARCH_PAGES + 1):
            time.sleep(CODE_SEARCH_PAUSE_S)
            result = gh_api(
                "search/code", "-X", "GET", "-f", f"q={query}", "-f", "per_page=100",
                "-f", f"page={page}",
            )  # fmt: skip
            for item in result["items"]:
                name = item["repository"]["full_name"]
                by_name.setdefault(name, {"full_name": name, "queries": []})
                by_name[name]["queries"] = sorted({*by_name[name]["queries"], query})
            if len(result["items"]) < 100:
                break
    missing = [r for r in by_name.values() if "created_at" not in r]
    with ThreadPoolExecutor(WORKERS) as pool:
        for repo, item in zip(missing, pool.map(_repo_metadata, missing), strict=True):
            if item is not None:
                repo.update(_repo_entry(item), queries=repo["queries"])
    repos = [r for r in by_name.values() if "created_at" in r]
    return sorted(repos, key=lambda r: r["full_name"].lower())


def _repo_metadata(repo: dict) -> dict | None:
    try:
        return gh_api(f"repos/{repo['full_name']}")
    except RuntimeError:  # deleted or made private since it was indexed
        return None


def _repo_entry(item: dict) -> dict:
    return {
        "full_name": item["full_name"],
        "owner": item["owner"]["login"],
        "html_url": item["html_url"],
        "description": item.get("description") or "",
        "created_at": item["created_at"],
        "license": (item.get("license") or {}).get("spdx_id"),
        "default_branch": item["default_branch"],
        "fork": item["fork"],
        "queries": [],
    }


def is_document_path(path: str, size: int) -> bool:
    if not path.lower().endswith(DOC_EXTENSIONS) or size > MAX_FILE_BYTES or SKIP_PATH.search(path):
        return False
    return path.lower() == "readme.md" or DOC_PATH.search(path) is not None


def select_files(nodes: list[dict]) -> list[dict]:
    """Candidate document files of a tree; the root README only when there is nothing else."""
    files = [
        {"path": n["path"], "sha": n["sha"], "size": n["size"]}
        for n in nodes
        if n["type"] == "blob" and is_document_path(n["path"], n["size"])
    ]
    documents = [f for f in files if f["path"].lower() != "readme.md"]
    return sorted(documents or files, key=lambda f: f["path"])[:MAX_FILES_PER_REPO]


def resolve_files(repo: dict) -> None:
    """Pin HEAD of the default branch and list the candidate document files at that commit."""
    try:
        head = gh_api(f"repos/{repo['full_name']}/commits/{repo['default_branch']}")
        tree = gh_api(f"repos/{repo['full_name']}/git/trees/{head['sha']}?recursive=1")
    except RateLimitedError:
        raise
    except RuntimeError as error:  # empty or deleted repository
        repo["error"] = str(error)[:200]
        repo["files"] = []
        return
    repo["head_sha"] = head["sha"]
    repo["head_date"] = head["commit"]["committer"]["date"]
    repo["tree_truncated"] = tree.get("truncated", False)
    repo["files"] = select_files(tree["tree"])


def _pending(repo: dict) -> bool:
    if "head_sha" in repo:
        return False
    return "error" not in repo or "rate limit" in repo["error"].lower()


def _save(snapshot: dict) -> None:
    path = snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)


def load_snapshot(wanted: Callable[[dict], bool], refresh: bool = False) -> dict:
    """Search once, then pin HEAD and list files of the `wanted` repositories (resumable)."""
    path = snapshot_path()
    if path.exists() and not refresh:
        snapshot = json.loads(path.read_text())
    else:
        snapshot = {
            "retrieved_at": datetime.now(UTC).date().isoformat(),
            "created_since": CREATED_SINCE,
            "licenses": sorted(ALLOWED_LICENSES.values()),
            "queries": list(QUERIES),
            "code_queries": list(CODE_QUERIES),
            "repos": search_repositories(),
        }
        _save(snapshot)
    pending = [r for r in snapshot["repos"] if wanted(r) and _pending(r)]
    for start in range(0, len(pending), SAVE_EVERY):
        for repo in pending[start : start + SAVE_EVERY]:
            repo.pop("error", None)
        with ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(resolve_files, pending[start : start + SAVE_EVERY]))
        _save(snapshot)
    return snapshot


def read_blob(full_name: str, sha: str) -> str:
    path = fresh_raw_dir() / "blobs" / sha
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{threading.get_ident()}.part")
        tmp.write_bytes(gh_api(f"repos/{full_name}/git/blobs/{sha}", raw=True))
        tmp.replace(path)
    return path.read_bytes().decode("utf-8", errors="replace")


def prefetch_blobs(repos: list[dict]) -> None:
    jobs = [(r["full_name"], f["sha"]) for r in repos for f in r.get("files", [])]
    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(lambda job: read_blob(*job), jobs))


def first_commit_date(full_name: str, path: str, head_sha: str) -> str:
    """Author date of the oldest commit that touched `path` up to `head_sha` (cached)."""
    cache = fresh_raw_dir() / "first_commit.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    key = f"{full_name}@{head_sha}:{path}"
    if key not in known:
        pages = gh_api(
            f"repos/{full_name}/commits", "-X", "GET", "-f", f"path={path}",
            "-f", f"sha={head_sha}", "-f", "per_page=100", "--paginate", "--slurp",
        )  # fmt: skip
        commits = [c for page in pages for c in page]
        known[key] = min(c["commit"]["author"]["date"] for c in commits)
        cache.write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    return known[key]
