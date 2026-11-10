"""Phase 1: Jev variants J1-J4 on dev, compared with the cheap baselines (plan §6.2-6.4).

`run_jev_dev(run=False)` is a dry run: it counts the calls missing from the inference cache
and estimates their cost; it makes no API call. With `run=True` it fills the cache, then
evaluates every variant with the same code as B0-B3 (reqbench.eval.cv) and writes:

- runs/phase1/dev/jev_answers.csv: one row per Jev call (probabilities, model, usage, latency);
- runs/phase1/dev/oof_J*.csv and summary_J*.json, in the layout of the baselines;
- runs/phase1/dev/jev_report.json: paired comparisons, stability, option order, cost.

When every call is already cached, the dry run also evaluates (it costs nothing).
"""

import json

import numpy as np
import pandas as pd
from reqcore.clients.jev import JevClient

from reqbench.eval import bootstrap, cv, metrics
from reqbench.models import jev
from reqbench.models.baselines import LABELS

REFERENCES = ("B3_bge_small_lr", "B2_tfidf_lr")
PRIMARY_REFERENCE = "B3_bge_small_lr"


def _variants(answers: pd.DataFrame, rows: pd.DataFrame, rotations) -> dict:
    """name -> (oof, fold params). J4_rot_avg is exploratory (not preregistered)."""
    out = {
        "J1_jev_choice": (jev.to_oof(jev.flat_probs(answers, (0,)), rows), []),
        "J2_jev_choice_rot_avg": (jev.to_oof(jev.flat_probs(answers, rotations), rows), []),
        "J4_jev_hierarchical": (jev.to_oof(jev.hierarchical_probs(answers, (0,)), rows), []),
        "J4x_jev_hierarchical_rot_avg": (
            jev.to_oof(jev.hierarchical_probs(answers, rotations), rows), []),
    }  # fmt: skip
    table = jev.features(answers, rotations)
    table.index = rows.set_index("req_id").loc[table.index, "text"]
    oof, params = cv.run_cv(lambda: jev.JevLR(table), rows)
    out["J3_jev_lr"] = (oof, params)
    return out


def _aligned(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = a[~a["is_example"]], b[~b["is_example"]]
    ids = sorted(set(a["req_id"]) & set(b["req_id"]))
    a = a.set_index("req_id").loc[ids].reset_index()
    b = b.set_index("req_id").loc[ids].reset_index()
    if not (a["gold"].to_numpy() == b["gold"].to_numpy()).all():
        raise RuntimeError("gold labels differ between OOF files")
    return a, b


def compare(oof_a: pd.DataFrame, oof_b: pd.DataFrame, n_resamples: int) -> dict:
    """Paired bootstrap of a - b: macro-F1, level-1 F1, accuracy, AURC (lower is better)."""
    a, b = _aligned(oof_a, oof_b)
    gold, n = a["gold"].to_numpy(), len(a)
    pa, pb = a["pred"].to_numpy(), b["pred"].to_numpy()
    out = {"n": n}
    out["macro_f1"] = bootstrap.paired_difference(
        lambda i: metrics.macro_f1(gold[i], pa[i], LABELS),
        lambda i: metrics.macro_f1(gold[i], pb[i], LABELS), n, n_resamples)
    out["level1_f1"] = bootstrap.paired_difference(
        lambda i: metrics.level1_f1(gold[i], pa[i]),
        lambda i: metrics.level1_f1(gold[i], pb[i]), n, n_resamples)
    out["accuracy"] = bootstrap.paired_difference(
        lambda i: metrics.accuracy(gold[i], pa[i]),
        lambda i: metrics.accuracy(gold[i], pb[i]), n, n_resamples)  # fmt: skip
    qa, qb = cv.probabilities(a), cv.probabilities(b)
    if qa is not None and qb is not None:
        out["aurc"] = bootstrap.paired_difference(
            lambda i: metrics.aurc(gold[i], qa[i], LABELS),
            lambda i: metrics.aurc(gold[i], qb[i], LABELS), n, n_resamples)  # fmt: skip
    return out


def run_jev_dev(run: bool = False, rotations=jev.ROTATIONS, n_samples: int = jev.N_SAMPLES,
                workers: int = 8, n_resamples: int = bootstrap.N_RESAMPLES) -> dict:  # fmt: skip
    dev = cv.load_dev()
    rows = dev[~dev["is_example"]].reset_index(drop=True)
    qsets = jev.question_sets(rotations)
    client = JevClient()
    calls = jev.plan_calls(rows, rotations, n_samples)
    estimate = jev.estimate(calls, client, qsets)
    if not run and estimate["calls_to_make"]:
        return {"dry_run": estimate}

    answers = jev.collect(calls, client, qsets, run=run, workers=workers)
    out_dir = cv.runs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    answers.to_csv(out_dir / "jev_answers.csv", index=False)

    systems = _variants(answers, rows, tuple(rotations))
    summaries = {}
    for name, (oof, params) in systems.items():
        summary = cv.evaluate(oof, n_resamples=n_resamples)
        cv.save(name, oof, params, summary)
        summaries[name] = summary

    references = {
        ref: pd.read_csv(out_dir / f"oof_{ref}.csv") for ref in REFERENCES
        if (out_dir / f"oof_{ref}.csv").exists()
    }
    comparisons = {
        name: {ref: compare(oof, ref_oof, n_resamples) for ref, ref_oof in references.items()}
        for name, (oof, _) in systems.items()
    }
    preregistered = [n for n in systems if not n.startswith("J4x")]
    if PRIMARY_REFERENCE in references:
        adjusted = bootstrap.holm({
            n: comparisons[n][PRIMARY_REFERENCE]["macro_f1"]["p_two_sided"] for n in preregistered
        })  # fmt: skip
        for name, p in adjusted.items():
            comparisons[name][PRIMARY_REFERENCE]["macro_f1"]["p_holm"] = p

    report = {
        "estimate": estimate,
        "usage": jev.usage_summary(answers),
        "stability": jev.stability(answers),
        "order": jev.order_sensitivity(answers, tuple(rotations)),
        "comparisons": comparisons,
        "summaries": {n: {k: s[k] for k in ("n", "ci", "per_fold_macro_f1")}
                      for n, s in summaries.items()},  # fmt: skip
        "confusion_J2": metrics.confusion(
            systems["J2_jev_choice_rot_avg"][0]["gold"],
            systems["J2_jev_choice_rot_avg"][0]["pred"], LABELS).tolist(),
        "labels": list(LABELS),
    }
    (out_dir / "jev_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n")  # fmt: skip
    return report


def _json_default(value):
    if isinstance(value, np.integer | np.floating):
        return value.item()
    raise TypeError(type(value))
