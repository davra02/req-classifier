"""Build the cleaned PROMISE dataset: normalize, map to the taxonomy and deduplicate.

Duplicate policy (documented in docs/DATA_CARD.md):
- Exact duplicates after aggressive normalization: keep the first occurrence.
- Near-duplicates within the same project (similarity >= NEAR_EXACT): keep the first.
- Near-duplicates across projects (similarity >= NEAR_DUP): kept, but recorded so the split
  step can drop the test-side member when the pair straddles dev and test.
"""

import difflib
import json
from dataclasses import dataclass, field

import pandas as pd
from reqcore.hashing import content_hash
from reqcore.normalize import dedup_key, normalize_text, text_hash
from reqcore.paths import data_dir
from reqcore.questions import load_taxonomy

from reqbench.data.download import fetch
from reqbench.data.promise import read_promise
from reqbench.data.sources import SOURCES

NEAR_EXACT = 0.98
NEAR_DUP = 0.90
DATASET_VERSION = "promise_v1"


@dataclass
class BuildReport:
    removed: list[dict] = field(default_factory=list)
    cross_project_pairs: list[tuple[str, str, float]] = field(default_factory=list)


def _similarity(a: str, b: str) -> float:
    matcher = difflib.SequenceMatcher(None, a, b)
    return matcher.ratio() if matcher.quick_ratio() >= NEAR_DUP else 0.0


def build_promise() -> tuple[pd.DataFrame, BuildReport]:
    taxonomy = load_taxonomy()
    rows = read_promise(fetch(SOURCES["promise_nfr"]))
    df = pd.DataFrame(
        {
            "req_id": [f"promise-{r.row_index:04d}" for r in rows],
            "project_id": [r.project_id for r in rows],
            "text": [normalize_text(r.text) for r in rows],
            "label_promise": [r.label for r in rows],
        }
    )
    df["leaf"] = df["label_promise"].map(taxonomy.promise_mapping)
    df["level1"] = df["leaf"].map(taxonomy.level1_of)
    df["text_hash"] = df["text"].map(text_hash)
    keys = df["text"].map(dedup_key).tolist()

    report = BuildReport()
    drop: set[int] = set()
    first_seen: dict[str, int] = {}
    for i, key in enumerate(keys):
        if key in first_seen:
            drop.add(i)
            report.removed.append(_removal(df, i, first_seen[key], "exact_duplicate", 1.0))
        else:
            first_seen[key] = i

    kept = [i for i in range(len(df)) if i not in drop]
    for pos, i in enumerate(kept):
        for j in kept[pos + 1 :]:
            if j in drop or abs(len(keys[i]) - len(keys[j])) > 40:
                continue
            sim = _similarity(keys[i], keys[j])
            if sim < NEAR_DUP:
                continue
            same_project = df.at[i, "project_id"] == df.at[j, "project_id"]
            if same_project and sim >= NEAR_EXACT:
                drop.add(j)
                report.removed.append(_removal(df, j, i, "near_exact_duplicate", sim))
            elif not same_project:
                report.cross_project_pairs.append(
                    (df.at[i, "req_id"], df.at[j, "req_id"], round(sim, 3))
                )

    clean = df.drop(index=sorted(drop)).reset_index(drop=True)
    return clean, report


def _removal(df: pd.DataFrame, i: int, kept: int, reason: str, sim: float) -> dict:
    return {
        "removed": df.at[i, "req_id"],
        "kept": df.at[kept, "req_id"],
        "reason": reason,
        "similarity": round(sim, 3),
        "removed_label": df.at[i, "label_promise"],
        "kept_label": df.at[kept, "label_promise"],
    }


def write_build(clean: pd.DataFrame, report: BuildReport) -> dict:
    """Write the processed CSV (git-ignored) and its manifest (tracked)."""
    processed = data_dir() / "processed" / f"{DATASET_VERSION}.csv"
    processed.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed, index=False)
    manifest = {
        "dataset_version": DATASET_VERSION,
        "source": "promise_nfr",
        "n_raw": len(clean) + len(report.removed),
        "n_clean": len(clean),
        "content_hash": content_hash(clean[["req_id", "text_hash", "leaf"]].values.tolist()),
        "removed": report.removed,
        "cross_project_near_duplicates": report.cross_project_pairs,
        "leaf_counts": clean["leaf"].value_counts().sort_index().to_dict(),
        "promise_counts": clean["label_promise"].value_counts().sort_index().to_dict(),
        "projects": int(clean["project_id"].nunique()),
    }
    path = data_dir() / "manifests" / f"{DATASET_VERSION}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def load_processed() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "processed" / f"{DATASET_VERSION}.csv")
