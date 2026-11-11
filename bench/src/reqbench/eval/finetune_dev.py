"""Phase 1: B4 (fine-tuned DeBERTa-v3-small) on dev, one CV run per seed (plan §6.2).

Each seed is a full 5-fold CV run saved as `oof_B4_deberta_v3_small_s<k>.csv` (k = 1..5) with
its summary; finished seeds are skipped on re-runs, so an interrupted run resumes. The report
(`B4_report.json`) gives the mean, SD and range of the metrics over seeds and the paired
bootstrap of every seed against J1 (Jev) and B3. The mean of the seeds' probabilities
(`B4e`, a seed ensemble) is reported as exploratory only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reqbench.eval import bootstrap, cv
from reqbench.models.baselines import LABELS
from reqbench.models.finetune import SEEDS, DebertaClassifier, Recipe

NAME = DebertaClassifier.name
REFERENCES = ("J1_jev_choice", "B3_bge_small_lr")
SEED_METRICS = ("macro_f1", "level1_f1", "level2_macro_f1", "accuracy", "log_loss", "ece",
                "aurc", "coverage_at_95")  # fmt: skip


def _path(k: int) -> Path:
    return cv.runs_dir() / f"oof_{NAME}_s{k}.csv"


def run_finetune_dev(seeds=SEEDS, recipe: Recipe | None = None,
                     n_resamples: int = bootstrap.N_RESAMPLES, log=print) -> dict:  # fmt: skip
    dev = cv.load_dev()
    summaries = {}
    for k, seed in enumerate(seeds, 1):
        name = f"{NAME}_s{k}"
        if _path(k).exists():
            log(f"{name}: already done, skipped")
        else:
            log(f"{name}: seed {seed}")
            oof, params = cv.run_cv(lambda seed=seed: DebertaClassifier(seed, recipe, log), dev)
            cv.save(name, oof, params, cv.evaluate(oof, n_resamples=n_resamples))
        summaries[name] = json.loads((cv.runs_dir() / f"summary_{name}.json").read_text())

    oofs = {name: pd.read_csv(cv.runs_dir() / f"oof_{name}.csv") for name in summaries}
    ensemble = _ensemble(list(oofs.values()))
    ensemble_summary = cv.evaluate(ensemble, n_resamples=n_resamples)
    cv.save(f"{NAME}_seed_ensemble", ensemble, [], ensemble_summary)

    references = {ref: pd.read_csv(cv.runs_dir() / f"oof_{ref}.csv") for ref in REFERENCES
                  if (cv.runs_dir() / f"oof_{ref}.csv").exists()}  # fmt: skip
    report = {
        "seeds": list(seeds),
        "per_seed": {n: {m: s.get(m) for m in SEED_METRICS} | {
            "epochs": [int(p["epochs"]) for p in s["fold_params"]]} for n, s in summaries.items()},
        "over_seeds": {
            m: _spread([s[m] for s in summaries.values() if m in s]) for m in SEED_METRICS
        },
        "ensemble_exploratory": {m: ensemble_summary.get(m) for m in SEED_METRICS},
        "comparisons": {
            n: {ref: cv.compare(oofs[n], r, n_resamples) for ref, r in references.items()}
            for n in oofs
        },
        "fit_seconds_total": sum(float(p["fit_seconds"]) for s in summaries.values()
                                 for p in s["fold_params"]),  # fmt: skip
    }
    (cv.runs_dir() / "B4_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _spread(values: list[float]) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}  # fmt: skip


def _ensemble(oofs: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [f"p_{label}" for label in LABELS]
    base = oofs[0].drop(columns=columns + ["pred"]).copy()
    mean = np.mean([o.sort_values("req_id")[columns].to_numpy() for o in oofs], axis=0)
    base = base.sort_values("req_id").reset_index(drop=True)
    base["pred"] = [LABELS[i] for i in mean.argmax(axis=1)]
    for k, column in enumerate(columns):
        base[column] = mean[:, k]
    return base
