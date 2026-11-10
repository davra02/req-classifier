"""Collect the fresh evaluation set from the GitHub snapshot (plan §6.9).

Pipeline (all deterministic, no model involved):
1. Repository filters: created since 2025, allowed license, not a fork, not a generator,
   template or LLM tool (their text is boilerplate or machine-written).
2. Extraction rules from `reqbench.fresh.extract` on each candidate document file.
3. Near-duplicates (>= REF_NEAR_DUP) of PROMISE or Dalpiaz texts are dropped; a repository with
   TEMPLATE_HITS or more of them is dropped whole, since it probably copied an old template.
4. A repository needs MIN_DOC_CANDIDATES candidates to count as a requirements document.
5. Seeded draw: repositories in a stable hash order (adding repositories later does not reorder
   the others), at most one per owner, every used file first committed since 2025, up to
   `per_doc` sentences each, no near-duplicates across the set, until `target` is reached.
6. Rows are shuffled with the seed, so annotators do not read a document in order.
"""

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd
from reqcore.hashing import content_hash
from reqcore.normalize import dedup_key, text_hash
from reqcore.paths import data_dir

from reqbench.data.dalpiaz import read_dalpiaz_texts
from reqbench.data.download import fetch
from reqbench.data.promise import read_promise
from reqbench.data.sources import SOURCES
from reqbench.fresh import github
from reqbench.fresh.extract import Candidate, extract_candidates

FRESH_VERSION = "fresh_v1"
SEED = 20261010
TARGET = 170
PER_DOC = 15
MAX_PER_DOC = 40
MIN_DOCS = 5
MIN_DOC_CANDIDATES = 10
REF_NEAR_DUP = 0.85
NEAR_DUP = 0.90
TEMPLATE_HITS = 3
MIRROR_SHARE = 0.5  # a document sharing this share of sentences with a chosen one is a copy
_TOKEN_OVERLAP = 0.4  # prefilter before the (slower) character similarity

EXCLUDE_REPO = re.compile(
    r"\b(generators?|generat(e|ion|ing)|templates?|boilerplate|awesome|llms?|gpt|chatgpt|"
    r"prompts?|copilot|skills?|agents?|agentic|plugins?)\b",
    re.IGNORECASE,
)
EXAMPLE_REPO = re.compile(r"\b(examples?|samples?|demos?|tutorials?)\b", re.IGNORECASE)
# Kiro-style generated specs: "THE Documentation_System SHALL document ...".
GENERATED_STYLE = re.compile(r"\bTHE\s+\S+(?:\s+\S+){0,3}\s+SHALL\b")
GENERATED_STYLE_SHARE = 0.5
DECLARED_GENERATED = re.compile(
    r"\b(generated|written|drafted|produced) (by|with|using) (chatgpt|gpt-?\d|claude|gemini|"
    r"copilot|an? (llm|ai)|ai)\b",
    re.IGNORECASE,
)
ANNOTATION_COLUMNS = [
    "annotator_a", "annotator_b", "annotator_c", "second_annotator", "adjudicated",
    "adjudication_note",
]  # fmt: skip


class NearDuplicateIndex:
    """Find near-duplicates by character similarity of `dedup_key`, prefiltered by tokens."""

    def __init__(self, texts: list[str] = ()) -> None:
        self.keys: list[str] = []
        self.tokens: list[set[str]] = []
        self.by_token: dict[str, list[int]] = {}
        for text in texts:
            self.add(text)

    def add(self, text: str) -> None:
        key = dedup_key(text)
        tokens = set(key.split())
        for token in tokens:
            self.by_token.setdefault(token, []).append(len(self.keys))
        self.keys.append(key)
        self.tokens.append(tokens)

    def match(self, text: str, threshold: float) -> tuple[int, float] | None:
        key = dedup_key(text)
        tokens = set(key.split())
        shared = Counter(i for t in tokens for i in self.by_token.get(t, ()))
        for i, n in shared.most_common():
            if n / max(len(tokens), len(self.tokens[i])) < _TOKEN_OVERLAP:
                break
            ratio = SequenceMatcher(None, key, self.keys[i]).ratio()
            if ratio >= threshold:
                return i, ratio
        return None


@dataclass
class Document:
    repo: dict
    candidates: list[Candidate]
    reference_hits: list[dict] = field(default_factory=list)
    declared_generated: bool = False

    @property
    def name(self) -> str:
        return self.repo["full_name"]


def _rank(seed: int, value: str) -> str:
    return content_hash([seed, value])


def repo_exclusion(repo: dict) -> str | None:
    reason = metadata_exclusion(repo)
    if reason is None and not repo.get("files"):
        return "no_document_files"
    return reason


def metadata_exclusion(repo: dict) -> str | None:
    """Exclusion decided from repository metadata alone, before listing its files."""
    if repo["fork"]:
        return "fork"
    if repo["license"] not in github.ALLOWED_LICENSES.values():
        return "license"
    if repo["created_at"][:10] < github.CREATED_SINCE:
        return "created_before_2025"
    name = re.sub(r"[-_.]", " ", repo["full_name"].split("/", 1)[1])
    if EXCLUDE_REPO.search(f"{name} {repo['description']}"):
        return "tool_or_template"
    if EXAMPLE_REPO.search(name):
        return "example_repo"
    return None


def reference_index() -> NearDuplicateIndex:
    promise = [r.text for r in read_promise(fetch(SOURCES["promise_nfr"]))]
    dalpiaz = read_dalpiaz_texts(fetch(SOURCES["dalpiaz_re19"]))
    return NearDuplicateIndex(promise + dalpiaz)


def build_document(repo: dict, files: dict[str, str], refs: NearDuplicateIndex) -> Document:
    """Extract candidates from every file of a repository and drop reference near-duplicates."""
    seen: set[str] = set()
    doc = Document(repo, [])
    doc.declared_generated = any(DECLARED_GENERATED.search(text) for text in files.values())
    for path in sorted(files):
        for candidate in extract_candidates(path, files[path]):
            key = dedup_key(candidate.text)
            if key in seen:
                continue
            seen.add(key)
            hit = refs.match(candidate.text, REF_NEAR_DUP)
            if hit is None:
                doc.candidates.append(candidate)
            else:
                doc.reference_hits.append(
                    {"text": candidate.text, "reference": refs.keys[hit[0]],
                     "similarity": round(hit[1], 3)}
                )  # fmt: skip
    return doc


def document_exclusion(doc: Document) -> str | None:
    if doc.declared_generated:
        return "declared_ai_generated"
    styled = sum(GENERATED_STYLE.search(c.text) is not None for c in doc.candidates)
    if doc.candidates and styled >= GENERATED_STYLE_SHARE * len(doc.candidates):
        return "generated_spec_style"
    if len(doc.reference_hits) >= TEMPLATE_HITS:
        return "template_copy"
    if len(doc.candidates) < MIN_DOC_CANDIDATES:
        return "few_candidates"
    return None


@dataclass
class Selection:
    documents: list[tuple[Document, list[Candidate], str]]  # (doc, chosen, published date)
    skipped: list[dict]

    @property
    def n_requirements(self) -> int:
        return sum(len(chosen) for _, chosen, _ in self.documents)


def select(
    documents: list[Document],
    published: Callable[[Document, list[Candidate]], str],
    *,
    seed: int = SEED,
    target: int = TARGET,
    per_doc: int = PER_DOC,
) -> Selection:
    """Seeded draw of documents and sentences; `published` returns the ISO date of the files."""
    if not 1 <= per_doc <= MAX_PER_DOC:
        raise ValueError(f"per_doc must be in 1..{MAX_PER_DOC}")
    selection = Selection([], [])
    owners: set[str] = set()
    taken = NearDuplicateIndex()
    for doc in sorted(documents, key=lambda d: _rank(seed, d.name)):
        if selection.n_requirements >= target:
            break
        if doc.repo["owner"].lower() in owners:
            selection.skipped.append({"repo": doc.name, "reason": "same_owner"})
            continue
        fresh = [c for c in doc.candidates if taken.match(c.text, NEAR_DUP) is None]
        if len(fresh) <= (1 - MIRROR_SHARE) * len(doc.candidates):
            selection.skipped.append({"repo": doc.name, "reason": "mirror_of_selected"})
            continue
        quota = min(per_doc, target - selection.n_requirements)
        chosen = sorted(fresh, key=lambda c: _rank(seed, text_hash(c.text)))[:quota]
        date = published(doc, chosen)
        if date < github.CREATED_SINCE:
            selection.skipped.append({"repo": doc.name, "reason": "file_before_2025"})
            continue
        for candidate in chosen:
            taken.add(candidate.text)
        owners.add(doc.repo["owner"].lower())
        selection.documents.append((doc, chosen, date))
    if len(selection.documents) < MIN_DOCS:
        raise RuntimeError(f"Only {len(selection.documents)} documents; need >= {MIN_DOCS}.")
    return selection


def published_on_github(doc: Document, chosen: list[Candidate]) -> str:
    paths = sorted({c.path for c in chosen})
    repo = doc.repo
    return min(
        github.first_commit_date(repo["full_name"], p, repo["head_sha"]) for p in paths
    )[:10]


def to_tables(selection: Selection, retrieved_at: str, seed: int) -> tuple[pd.DataFrame, ...]:
    doc_rows, req_rows = [], []
    for i, (doc, chosen, date) in enumerate(selection.documents, start=1):
        doc_id = f"fresh-doc-{i:02d}"
        repo = doc.repo
        paths = sorted({c.path for c in chosen})
        doc_rows.append(
            {
                "doc_id": doc_id,
                "title": repo["description"] or repo["full_name"],
                "url": f"https://github.com/{repo['full_name']}/tree/{repo['head_sha']}",
                "published_date": date,
                "license": repo["license"],
                "retrieved_at": retrieved_at,
                "version_or_commit": repo["head_sha"],
                "max_requirements_taken": len(chosen),
                "notes": (
                    f"repo {repo['full_name']} created {repo['created_at'][:10]}; "
                    f"files: {', '.join(paths)}; {len(doc.candidates)} candidates, "
                    f"{len(doc.reference_hits)} PROMISE/Dalpiaz near-duplicates dropped"
                ),
            }
        )
        req_rows += [{"doc_id": doc_id, "location": c.location, "text": c.text} for c in chosen]
    requirements = pd.DataFrame(req_rows)
    order = requirements["text"].map(lambda t: _rank(seed, "order:" + text_hash(t)))
    requirements = requirements.iloc[order.argsort(kind="stable")].reset_index(drop=True)
    requirements.insert(0, "req_id", [f"fresh-{i:04d}" for i in range(1, len(requirements) + 1)])
    for column in ANNOTATION_COLUMNS:
        requirements[column] = ""
    return pd.DataFrame(doc_rows), requirements


@dataclass
class Collection:
    snapshot: dict
    repo_exclusions: Counter
    doc_exclusions: Counter
    eligible: list[Document]
    excluded_docs: list[tuple[Document, str]]
    selection: Selection
    documents: pd.DataFrame
    requirements: pd.DataFrame


def collect(
    refresh: bool = False, seed: int = SEED, target: int = TARGET, per_doc: int = PER_DOC
) -> Collection:
    snapshot = github.load_snapshot(lambda r: metadata_exclusion(r) is None, refresh=refresh)
    refs = reference_index()
    repo_exclusions: Counter = Counter()
    doc_exclusions: Counter = Counter()
    eligible: list[Document] = []
    excluded_docs: list[tuple[Document, str]] = []
    github.prefetch_blobs([r for r in snapshot["repos"] if repo_exclusion(r) is None])
    for repo in snapshot["repos"]:
        reason = repo_exclusion(repo)
        if reason:
            repo_exclusions[reason] += 1
            continue
        # Re-apply the path rules: the snapshot may predate a rule change.
        kept = github.select_files([f | {"type": "blob"} for f in repo["files"]])
        if not kept:
            repo_exclusions["no_document_files"] += 1
            continue
        files = {f["path"]: github.read_blob(repo["full_name"], f["sha"]) for f in kept}
        doc = build_document(repo, files, refs)
        reason = document_exclusion(doc)
        if reason:
            doc_exclusions[reason] += 1
            excluded_docs.append((doc, reason))
        else:
            eligible.append(doc)
    selection = select(eligible, published_on_github, seed=seed, target=target, per_doc=per_doc)
    documents, requirements = to_tables(selection, snapshot["retrieved_at"], seed)
    return Collection(
        snapshot, repo_exclusions, doc_exclusions, eligible, excluded_docs, selection,
        documents, requirements,
    )  # fmt: skip


def write_collection(result: Collection, seed: int, target: int, per_doc: int) -> dict:
    """Write data/fresh/*.csv and the manifest; never overwrite started annotations."""
    fresh = data_dir() / "fresh"
    existing = fresh / "requirements.csv"
    if existing.exists():
        current = pd.read_csv(existing, dtype=str).fillna("")
        if (current.reindex(columns=ANNOTATION_COLUMNS, fill_value="") != "").any().any():
            raise RuntimeError(f"{existing} already has annotations; refusing to overwrite it.")
    result.documents.to_csv(fresh / "documents.csv", index=False)
    result.requirements.to_csv(existing, index=False)
    manifest = {
        "fresh_version": FRESH_VERSION,
        "retrieved_at": result.snapshot["retrieved_at"],
        "search": {
            "queries": result.snapshot["queries"],
            "code_queries": result.snapshot.get("code_queries", []),
            "created_since": result.snapshot["created_since"],
            "licenses": result.snapshot["licenses"],
        },
        "params": {
            "seed": seed, "target": target, "per_doc": per_doc,
            "min_doc_candidates": MIN_DOC_CANDIDATES, "ref_near_dup": REF_NEAR_DUP,
            "near_dup": NEAR_DUP, "template_hits": TEMPLATE_HITS, "mirror_share": MIRROR_SHARE,
            "generated_style_share": GENERATED_STYLE_SHARE,
        },  # fmt: skip
        "counts": {
            "repos_found": len(result.snapshot["repos"]),
            "repos_excluded": dict(result.repo_exclusions),
            "documents_excluded": dict(result.doc_exclusions),
            "documents_eligible": len(result.eligible),
            "candidates_eligible": sum(len(d.candidates) for d in result.eligible),
            "documents_selected": len(result.documents),
            "requirements_selected": len(result.requirements),
        },
        "skipped_during_draw": result.selection.skipped,
        "content_hash": content_hash(
            result.requirements[["req_id", "doc_id", "location", "text"]].values.tolist()
        ),
    }
    path = data_dir() / "manifests" / f"{FRESH_VERSION}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest
