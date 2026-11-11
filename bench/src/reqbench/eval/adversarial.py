"""Adversarial suite `adversarial_v1` (plan §6.3 robustness (c); GO criterion 3).

Variants are generated from dev requirements by the fixed templates of
questions/adversarial_v1.yaml; nothing is selected or edited by hand. Each system's flip rate
is the share of variants whose prediction differs from its prediction on the original text
(the original predictions are the dev ones: Jev rotation 0 sample 0, B5's k = 5 vote).
"""

import numpy as np
import pandas as pd
from reqcore.paths import questions_dir
from reqcore.questions import load_yaml

from reqbench.eval import cv
from reqbench.models.baselines import LABELS

VERSION = "adversarial_v1"


def spec() -> dict:
    return load_yaml(VERSION, questions_dir())


def _allocation(counts: pd.Series, n: int, minimum: int = 2) -> dict[str, int]:
    share = counts / counts.sum() * n
    alloc = {k: max(minimum, int(np.floor(v))) for k, v in share.items()}
    remainders = sorted(share.items(), key=lambda kv: -(kv[1] - np.floor(kv[1])))
    i = 0
    while sum(alloc.values()) < n:
        alloc[remainders[i % len(remainders)][0]] += 1
        i += 1
    return alloc


def variants() -> pd.DataFrame:
    """One row per variant: req_id, variant_id, attack, target, gold, original text, text."""
    s = spec()
    dev = cv.load_dev()
    rows = dev[~dev["is_example"]].sort_values("req_id").reset_index(drop=True)
    rng = np.random.default_rng(s["seed"])
    alloc = _allocation(rows["leaf"].value_counts().reindex(LABELS), s["n_requirements"])
    picked = []
    for leaf in LABELS:
        pool = rows[rows["leaf"] == leaf]
        picked.append(pool.iloc[np.sort(rng.choice(len(pool), alloc[leaf], replace=False))])
    base = pd.concat(picked).sort_values("req_id")
    out = []
    for r in base.itertuples():
        target = LABELS[(LABELS.index(r.leaf) + s["target_offset"]) % len(LABELS)]
        for attack, template in s["templates"].items():
            out.append({
                "req_id": r.req_id, "variant_id": f"{r.req_id}:{attack}", "attack": attack,
                "target": target, "gold": r.leaf, "original": r.text,
                "text": template.format(text=r.text, target=target).strip(),
            })  # fmt: skip
    return pd.DataFrame(out)


def flip_rates(adv: pd.DataFrame, original_pred: pd.Series, variant_pred: pd.Series) -> dict:
    """original_pred indexed by req_id; variant_pred indexed by variant_id."""
    orig = original_pred.loc[adv["req_id"]].to_numpy()
    new = variant_pred.loc[adv["variant_id"]].to_numpy()
    flipped = orig != new
    success = new == adv["target"].to_numpy()
    by_attack = {
        a: {"flip_rate": float(flipped[m].mean()), "attack_success": float(success[m].mean())}
        for a in adv["attack"].unique() for m in [adv["attack"].to_numpy() == a]
    }  # fmt: skip
    return {"n_variants": len(adv), "flip_rate": float(flipped.mean()),
            "attack_success": float(success.mean()), "by_attack": by_attack}  # fmt: skip
