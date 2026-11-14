"""Fresh-set analyses fixed in docs/PREREGISTRATION_ADDENDUM.md (written before the run).

Reads only saved predictions: runs/phase1/fresh/pred_*.csv (written by
`reqbench heldout fresh --run`) and runs/phase1/test/pred_*.csv. No API calls.

- κ of every system against the gold (6 leaves), next to the human κ;
- the PROMISE-to-fresh drop of every system;
- both gold sets: primary (all non-discard rows) and secondary (double-annotated rows);
- R1: macro-F1(J3) − macro-F1(B5x) ≥ 0.10 with paired-bootstrap CI lower bound > 0;
- R2: drop(B5x) − mean drop(B4 seeds) ≥ 0.10 with CI lower bound > 0 (test and fresh
  resampled independently);
- criterion 2(c) as preregistered: J3 − B5 ≥ δ with CI lower bound > 0.
"""

import json

import numpy as np
import pandas as pd
from reqcore.paths import data_dir
from sklearn.metrics import cohen_kappa_score

from reqbench.eval import bootstrap, cv, heldout, metrics
from reqbench.models.baselines import LABELS

REOPEN_THRESHOLD = 0.10
HUMAN_KAPPA = {"leaves_with_discard": (0.31, [0.18, 0.43]), "level1": 0.29}
JEV, LLM1, LLM5 = "J3_jev_lr", "B5x_haiku_single", "B5_haiku_k5"
B4 = heldout.B4_SEEDS
REPORTED = ("B0_majority", "B1_keywords", "B2_tfidf_lr", "B3_bge_small_lr", *B4,
            "J1_jev_choice", "J2_jev_choice_rot_avg", "J3_jev_lr", "J4_jev_hierarchical",
            LLM5, LLM1, "B6_opus")  # fmt: skip


def kappa(frame: pd.DataFrame) -> float:
    return float(cohen_kappa_score(frame["gold"], frame["pred"], labels=list(LABELS)))


def _f1(frame: pd.DataFrame, idx: np.ndarray) -> float:
    return metrics.macro_f1(frame["gold"].to_numpy()[idx], frame["pred"].to_numpy()[idx], LABELS)


def r1(fresh: dict[str, pd.DataFrame], n_resamples: int) -> dict:
    d = cv.compare(fresh[JEV], fresh[LLM1], n_resamples)["macro_f1"]
    return {"diff": d, "met": bool(d["value"] >= REOPEN_THRESHOLD and d["ci95"][0] > 0)}


def r2(test: dict[str, pd.DataFrame], fresh: dict[str, pd.DataFrame], n_resamples: int,
       seed: int = bootstrap.SEED) -> dict:  # fmt: skip
    def gap(it: np.ndarray, iff: np.ndarray) -> float:
        drop_llm = _f1(test[LLM1], it) - _f1(fresh[LLM1], iff)
        drop_b4 = np.mean([_f1(test[s], it) - _f1(fresh[s], iff) for s in B4])
        return float(drop_llm - drop_b4)

    nt, nf = len(test[LLM1]), len(fresh[LLM1])
    rng = np.random.default_rng(seed)
    values = np.array([gap(rng.integers(0, nt, nt), rng.integers(0, nf, nf))
                       for _ in range(n_resamples)])  # fmt: skip
    point = gap(np.arange(nt), np.arange(nf))
    ci = [float(x) for x in np.percentile(values, [2.5, 97.5])]
    return {"value": point, "ci95": ci,
            "met": bool(point >= REOPEN_THRESHOLD and ci[0] > 0)}  # fmt: skip


def _load(set_name: str) -> dict[str, pd.DataFrame]:
    d = heldout.out_dir(set_name)
    return {n: pd.read_csv(d / f"pred_{n}.csv").sort_values("req_id").reset_index(drop=True)
            for n in REPORTED}  # fmt: skip


def _double_annotated() -> set[str]:
    df = pd.read_csv(data_dir() / "fresh" / "requirements.csv", dtype=str, keep_default_na=False)
    return set(df.loc[df["second_annotator"] != "", "req_id"])


def analyse(fresh: dict[str, pd.DataFrame], test: dict[str, pd.DataFrame], double: set[str],
            n_resamples: int = bootstrap.N_RESAMPLES) -> dict:  # fmt: skip
    secondary = {n: f[f["req_id"].isin(double)].reset_index(drop=True) for n, f in fresh.items()}
    per_system = {}
    for n in REPORTED:
        per_system[n] = {
            "test_macro_f1": _f1(test[n], np.arange(len(test[n]))),
            "fresh_macro_f1": _f1(fresh[n], np.arange(len(fresh[n]))),
            "fresh_kappa": kappa(fresh[n]),
            "fresh_double_annotated_macro_f1": _f1(secondary[n], np.arange(len(secondary[n]))),
            "fresh_double_annotated_kappa": kappa(secondary[n]),
        }  # fmt: skip
        per_system[n]["drop"] = per_system[n]["test_macro_f1"] - per_system[n]["fresh_macro_f1"]
    c2c = cv.compare(fresh[JEV], fresh[LLM5], n_resamples)["macro_f1"]
    return {
        "n_fresh": len(fresh[LLM1]), "n_double_annotated": len(secondary[LLM1]),
        "gold_counts": fresh[LLM1]["gold"].value_counts().to_dict(),
        "human_kappa": HUMAN_KAPPA, "per_system": per_system,
        "R1_jev_clearly_better": r1(fresh, n_resamples),
        "R2_contamination_signal": r2(test, fresh, n_resamples),
        "criterion_2c": {"diff": c2c, "met": bool(c2c["value"] >= heldout.DELTA
                                                  and c2c["ci95"][0] > 0)},
        "R1_secondary_gold": r1(secondary, n_resamples),
    }  # fmt: skip


def run(n_resamples: int = bootstrap.N_RESAMPLES) -> dict:
    out = analyse(_load("fresh"), _load("test"), _double_annotated(), n_resamples)
    out["reopen_adr_0002"] = out["R1_jev_clearly_better"]["met"] or \
        out["R2_contamination_signal"]["met"]  # fmt: skip
    path = heldout.out_dir("fresh") / "analysis.json"
    path.write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out
