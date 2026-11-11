"""Phase 1, step 6a-b: calibration (plan §6.4) and human-review thresholds (§6.5) on dev.

Inputs are the out-of-fold dev predictions in runs/phase1/dev/ (examples_v1 excluded); the
locked test is never read. For every system with probabilities:

1. `calibration.nested_oof` picks a calibrator by pooled log loss over the project-grouped
   folds and, in an outer loop, produces calibrated dev probabilities that no calibrator (or
   choice of calibrator) fitted on their own fold.
2. On those probabilities: calibrated ECE with a bootstrap CI (operative criterion ≤ 0.05),
   log loss, Brier, AURC and coverage at 95 % precision, next to the raw values.
3. Review threshold = the lowest calibrated confidence at which the automatic decisions keep
   precision ≥ 0.95 on dev (block-aware, as `metrics.coverage_at_precision`).

Jev is calibrated per question: J1/J2 (`req_type`), J4 = `is_functional` (Noul, binary
candidates, all rows) + `nfr_subtype` (Choice, fitted on gold-NFR rows only), recombined as
in `models.jev.hierarchical_probs`. J3 and the baselines are calibrated as whole systems.
The gate Nouls (`is_requirement`, `addresses_ai`) have no gold label in PROMISE and are not
calibrated here.

Frozen outputs (versioned): frozen/phase1/calibrators.json and frozen/phase1/thresholds.json.
Report: runs/phase1/dev/calibration_report.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from reqcore.hashing import content_hash
from reqcore.paths import repo_root

from reqbench.eval import bootstrap, calibration, cv, metrics
from reqbench.models import jev
from reqbench.models.baselines import LABELS

B4_SEEDS = tuple(f"B4_deberta_v3_small_s{k}" for k in range(1, 6))
SYSTEMS = ("B2_tfidf_lr", "B3_bge_small_lr", *B4_SEEDS, "B5_haiku_k5",
           "J1_jev_choice", "J2_jev_choice_rot_avg", "J3_jev_lr")  # fmt: skip
J4 = "J4_jev_hierarchical"
NFR = jev.NFR


def frozen_dir() -> Path:
    return repo_root() / "frozen" / "phase1"


def _rows(oof: pd.DataFrame) -> pd.DataFrame:
    return oof[~oof["is_example"]].sort_values("req_id").reset_index(drop=True)


def _codes(values, labels) -> np.ndarray:
    return np.array([list(labels).index(v) for v in values])


def _with_probs(rows: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    out = rows.copy()
    out["pred"] = [LABELS[i] for i in probs.argmax(axis=1)]
    for k, label in enumerate(LABELS):
        out[f"p_{label}"] = probs[:, k]
    return out


def _metrics(gold, probs, n_resamples: int) -> dict:
    pred = np.array([LABELS[i] for i in probs.argmax(axis=1)])
    gold = np.asarray(gold)

    def ece(i):
        return metrics.ece_top_label(gold[i], probs[i], LABELS)

    return {
        "macro_f1": metrics.macro_f1(gold, pred, LABELS),
        "accuracy": metrics.accuracy(gold, pred),
        "log_loss": metrics.log_loss(gold, probs, LABELS),
        "brier": metrics.brier(gold, probs, LABELS),
        "ece": bootstrap.ci(ece, len(gold), n_resamples),
        "aurc": metrics.aurc(gold, probs, LABELS),
        "coverage_at_95": metrics.coverage_at_precision(gold, probs, LABELS),
    }


def _system(name: str, n_resamples: int) -> tuple[dict, dict, pd.DataFrame]:
    rows = _rows(pd.read_csv(cv.runs_dir() / f"oof_{name}.csv"))
    raw = cv.probabilities(rows)
    y, folds = _codes(rows["gold"], LABELS), rows["fold"].to_numpy()
    res = calibration.nested_oof(raw, y, folds)
    frozen = {"questions": {"system": {
        "labels": list(LABELS), **calibration.to_dict(res.final),
        "cv_log_loss": res.final_scores, "chosen_per_outer_fold": res.chosen_per_fold,
    }}, "input_hash": content_hash(rows[["req_id"] + [f"p_{k}" for k in LABELS]]
                                   .round(10).to_dict("list"))}  # fmt: skip
    return frozen, {"raw": _metrics(rows["gold"], raw, n_resamples),
                    "calibrated": _metrics(rows["gold"], res.calibrated, n_resamples)}, \
        _with_probs(rows, res.calibrated)  # fmt: skip


def _j4(n_resamples: int) -> tuple[dict, dict, pd.DataFrame]:
    rows = _rows(pd.read_csv(cv.runs_dir() / f"oof_{J4}.csv"))
    answers = pd.read_csv(cv.runs_dir() / "jev_answers.csv")
    r0 = answers[(answers["rotation"] == 0) & (answers["sample_index"] == 0)]
    r0 = r0.set_index("req_id").loc[rows["req_id"]]
    folds, gold = rows["fold"].to_numpy(), rows["gold"].to_numpy()
    p_f = r0["nf_is_functional"].to_numpy()
    is_f = calibration.nested_oof(np.column_stack([1 - p_f, p_f]),
                                  (gold == "functional").astype(int), folds)  # fmt: skip
    sub = jev._renorm(r0[[f"ns_{k}" for k in NFR]].to_numpy())
    nfr_mask = gold != "functional"
    y_sub = np.array([NFR.index(g) if g in NFR else 0 for g in gold])
    ns = calibration.nested_oof(sub, y_sub, folds, fit_mask=nfr_mask)
    combined = combine_j4(is_f.calibrated[:, 1], ns.calibrated)
    raw = cv.probabilities(rows)
    frozen = {"questions": {
        "is_functional": {"labels": ["false", "true"], **calibration.to_dict(is_f.final),
                          "cv_log_loss": is_f.final_scores,
                          "chosen_per_outer_fold": is_f.chosen_per_fold},
        "nfr_subtype": {"labels": list(NFR), **calibration.to_dict(ns.final),
                        "cv_log_loss": ns.final_scores, "fit_rows": "gold NFR only",
                        "chosen_per_outer_fold": ns.chosen_per_fold},
    }, "input_hash": content_hash({"p_f": np.round(p_f, 10).tolist(),
                                   "sub": np.round(sub, 10).tolist()})}  # fmt: skip
    report = {"raw": _metrics(gold, raw, n_resamples),
              "calibrated": _metrics(gold, combined, n_resamples)}  # fmt: skip
    return frozen, report, _with_probs(rows, combined)


def combine_j4(p_functional: np.ndarray, sub: np.ndarray) -> np.ndarray:
    probs = np.zeros((len(p_functional), len(LABELS)))
    probs[:, LABELS.index("functional")] = p_functional
    for j, leaf in enumerate(NFR):
        probs[:, LABELS.index(leaf)] = (1 - p_functional) * sub[:, j]
    return probs


def apply(spec: dict, system: str, raw: dict[str, np.ndarray]) -> np.ndarray:
    """Calibrated probabilities with a frozen spec. `raw` holds the system's probability
    matrix under "system" or, for J4, the `is_functional` p ("is_functional", 1-D) and the
    renormalized `nfr_subtype` matrix ("nfr_subtype")."""
    questions = spec[system]["questions"]
    if system == J4:
        p = raw["is_functional"]
        p_f = calibration.from_dict(questions["is_functional"]).transform(
            np.column_stack([1 - p, p]))[:, 1]  # fmt: skip
        sub = calibration.from_dict(questions["nfr_subtype"]).transform(raw["nfr_subtype"])
        return combine_j4(p_f, sub)
    return calibration.from_dict(questions["system"]).transform(raw["system"])


def run_calibration_dev(n_resamples: int = bootstrap.N_RESAMPLES, write: bool = True) -> dict:
    frozen, report, thresholds = {}, {}, {}
    for name in (*SYSTEMS, J4):
        spec, rep, calibrated = _j4(n_resamples) if name == J4 else _system(name, n_resamples)
        gold, probs = calibrated["gold"].to_numpy(), cv.probabilities(calibrated)
        t = metrics.threshold_at_precision(gold, probs, LABELS)
        sel = metrics.selective(gold, calibrated["pred"].to_numpy(), probs, t)
        frozen[name] = spec
        thresholds[name] = {"target_precision": metrics.TARGET_PRECISION, **sel,
                            "set_on": "dev nested-CV calibrated probabilities"}  # fmt: skip
        report[name] = rep | {"threshold_dev": thresholds[name]}
        if write:
            calibrated.to_csv(cv.runs_dir() / f"cal_{name}.csv", index=False)
    if write:
        out = frozen_dir()
        out.mkdir(parents=True, exist_ok=True)
        meta = {"eps": calibration.EPS, "vector_l2": calibration.VECTOR_L2,
                "selection": "pooled log loss over the 5 project-grouped dev folds"}  # fmt: skip
        (out / "calibrators.json").write_text(
            json.dumps({"meta": meta, "systems": frozen}, indent=1) + "\n")
        (out / "thresholds.json").write_text(json.dumps(thresholds, indent=1) + "\n")
        (cv.runs_dir() / "calibration_report.json").write_text(
            json.dumps(report, indent=2) + "\n")
    return report
