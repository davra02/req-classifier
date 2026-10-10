"""Frozen, project-grouped splits for the phase-1 benchmark (plan §6.1).

- Locked test: whole projects, chosen by a seeded draw among the project combinations that
  hold 27-33 % of requirements, have >= MIN_PER_LEAF examples of every leaf, and are among
  the TOP_K closest (KL divergence) to the overall leaf distribution. No one hand-picks it.
- Cross-project near-duplicate pairs that straddle dev/test lose their test-side member.
- Dev: StratifiedGroupKFold(5) by project for model selection, calibration and thresholds.
"""

import itertools
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from reqcore.hashing import content_hash
from reqcore.paths import data_dir
from sklearn.model_selection import StratifiedGroupKFold

SEED = 20261010
TEST_FRACTION = (0.27, 0.33)
MIN_PER_LEAF = 10
TOP_K = 50
N_FOLDS = 5
SPLIT_VERSION = "promise_v1_split_v1"


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    return sum(p[c] * math.log(p[c] / max(q.get(c, 0.0), 1e-9)) for c in p if p[c] > 0)


def candidate_test_sets(df: pd.DataFrame) -> list[tuple[float, tuple[int, ...]]]:
    leaves = sorted(df["leaf"].unique())
    total = len(df)
    overall = {c: n / total for c, n in Counter(df["leaf"]).items()}
    per_project = {p: Counter(g["leaf"]) for p, g in df.groupby("project_id")}
    projects = sorted(per_project)
    candidates = []
    for size in range(2, len(projects)):
        for combo in itertools.combinations(projects, size):
            counts: Counter[str] = Counter()
            for p in combo:
                counts.update(per_project[p])
            n = sum(counts.values())
            if not TEST_FRACTION[0] * total <= n <= TEST_FRACTION[1] * total:
                continue
            if min(counts.get(c, 0) for c in leaves) < MIN_PER_LEAF:
                continue
            candidates.append((_kl(overall, {c: k / n for c, k in counts.items()}), combo))
    candidates.sort()
    return candidates


def make_splits(df: pd.DataFrame, cross_project_pairs: list[list]) -> tuple[pd.DataFrame, dict]:
    candidates = candidate_test_sets(df)
    if not candidates:
        raise ValueError("No project combination satisfies the test-set constraints.")
    rng = np.random.default_rng(SEED)
    pick = int(rng.integers(min(TOP_K, len(candidates))))
    test_projects = set(candidates[pick][1])

    out = df[["req_id", "text_hash", "project_id", "leaf"]].copy()
    out["split"] = np.where(out["project_id"].isin(test_projects), "test", "dev")

    split_of = dict(zip(out["req_id"], out["split"], strict=True))
    excluded = []
    for a, b, _sim in cross_project_pairs:
        if {split_of[a], split_of[b]} == {"dev", "test"}:
            test_side = a if split_of[a] == "test" else b
            excluded.append(test_side)
    out.loc[out["req_id"].isin(excluded), "split"] = "excluded"

    out["fold"] = -1
    dev = out[out["split"] == "dev"]
    folds = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, idx) in enumerate(folds.split(dev, dev["leaf"], dev["project_id"])):
        out.loc[dev.index[idx], "fold"] = fold

    meta = {
        "split_version": SPLIT_VERSION,
        "seed": SEED,
        "n_candidates": len(candidates),
        "picked_rank": pick,
        "test_projects": sorted(int(p) for p in test_projects),
        "excluded_from_test": excluded,
        "counts": {s: int(n) for s, n in out["split"].value_counts().items()},
        "test_leaf_counts": out[out["split"] == "test"]["leaf"].value_counts().to_dict(),
        "fold_sizes": out[out["fold"] >= 0]["fold"].value_counts().sort_index().to_dict(),
    }
    return out, meta


def write_splits(splits: pd.DataFrame, meta: dict) -> dict:
    """Write split assignments (IDs and hashes only, no text) and their manifest; both tracked."""
    path = data_dir() / "splits" / f"{SPLIT_VERSION}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["req_id", "text_hash", "project_id", "split", "fold"]
    splits[columns].to_csv(path, index=False)
    meta["content_hash"] = content_hash(splits[columns].values.tolist())
    manifest = data_dir() / "manifests" / f"{SPLIT_VERSION}.json"
    manifest.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=int) + "\n")
    return meta


def load_splits() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "splits" / f"{SPLIT_VERSION}.csv")
