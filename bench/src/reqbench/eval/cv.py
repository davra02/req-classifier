"""Cross-validation on dev with the frozen project-grouped folds (plan §6.1).

Only `split == "dev"` rows are ever loaded here; the locked test is not read. Out-of-fold
predictions (and probabilities) are saved to runs/phase1/dev/ for calibration and thresholds.
Metrics exclude the few-shot example IDs of examples_v1, which the LLM and Jev see in their
prompts.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from reqcore.paths import data_dir, questions_dir, repo_root
from reqcore.questions import load_taxonomy, load_yaml

from reqbench.data.build import DATASET_VERSION
from reqbench.data.splits import SPLIT_VERSION
from reqbench.eval import bootstrap, metrics
from reqbench.models.baselines import LABELS

EXAMPLES_VERSION = "examples_v1"
CI_METRICS = ("macro_f1", "level1_f1", "level2_macro_f1", "accuracy")
CI_PROB_METRICS = ("log_loss", "brier", "ece", "aurc", "coverage_at_95")


def runs_dir() -> Path:
    return repo_root() / "runs" / "phase1" / "dev"


def load_dev() -> pd.DataFrame:
    splits = pd.read_csv(data_dir() / "splits" / f"{SPLIT_VERSION}.csv")
    processed = pd.read_csv(data_dir() / "processed" / f"{DATASET_VERSION}.csv")
    dev = splits[splits["split"] == "dev"].merge(
        processed[["req_id", "text", "leaf"]], on="req_id", how="left", validate="one_to_one"
    )
    if dev["text"].isna().any() or (dev["fold"] < 0).any():
        raise RuntimeError("dev rows without text or fold; rebuild with `reqbench data all`.")
    examples = load_yaml(EXAMPLES_VERSION, questions_dir())["source_ids"]
    dev["is_example"] = dev["req_id"].isin({i for ids in examples.values() for i in ids})
    return dev.reset_index(drop=True)


def run_cv(make_model: Callable[[], object], dev: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Out-of-fold predictions for every dev row, plus the parameters chosen in each fold."""
    parts, params = [], []
    for fold in sorted(dev["fold"].unique()):
        train, held = dev[dev["fold"] != fold], dev[dev["fold"] == fold]
        model = make_model()
        model.fit(train["text"].tolist(), train["leaf"].tolist(), train["fold"].tolist())
        part = held[["req_id", "project_id", "fold", "leaf", "is_example"]].copy()
        part = part.rename(columns={"leaf": "gold"})
        if model.has_proba:
            proba = model.predict_proba(held["text"].tolist())
            part["pred"] = [LABELS[i] for i in proba.argmax(axis=1)]
            for k, label in enumerate(LABELS):
                part[f"p_{label}"] = proba[:, k]
        else:
            part["pred"] = model.predict(held["text"].tolist())
        parts.append(part)
        params.append({"fold": int(fold), **{k: str(v) for k, v in model.params_.items()}})
    return pd.concat(parts).sort_values("req_id").reset_index(drop=True), params


def probabilities(oof: pd.DataFrame) -> np.ndarray | None:
    columns = [f"p_{label}" for label in LABELS]
    return oof[columns].to_numpy() if columns[0] in oof else None


def evaluate(oof: pd.DataFrame, n_resamples: int = bootstrap.N_RESAMPLES) -> dict:
    """Point metrics, bootstrap CIs and per-fold macro-F1 on the non-example rows."""
    nfr = load_taxonomy().level1["non_functional"]
    rows = oof[~oof["is_example"]].reset_index(drop=True)
    gold, pred, probs = rows["gold"].to_numpy(), rows["pred"].to_numpy(), probabilities(rows)
    summary = metrics.summarize(gold, pred, probs, LABELS, nfr)
    names = CI_METRICS + (CI_PROB_METRICS if probs is not None else ())
    summary["ci"] = {
        name: bootstrap.ci(_fast_metric(name, gold, pred, probs, nfr), len(rows), n_resamples)
        for name in names
    }
    summary["per_fold_macro_f1"] = {
        int(f): metrics.macro_f1(g["gold"], g["pred"], LABELS) for f, g in rows.groupby("fold")
    }
    return summary


def _fast_metric(name, gold, pred, probs, nfr) -> Callable[[np.ndarray], float]:
    """A metric of a resample (row indices) without recomputing the whole summary."""
    functions = {
        "macro_f1": lambda i: metrics.macro_f1(gold[i], pred[i], LABELS),
        "level1_f1": lambda i: metrics.level1_f1(gold[i], pred[i]),
        "level2_macro_f1": lambda i: metrics.level2_macro_f1(gold[i], pred[i], nfr),
        "accuracy": lambda i: metrics.accuracy(gold[i], pred[i]),
        "log_loss": lambda i: metrics.log_loss(gold[i], probs[i], LABELS),
        "brier": lambda i: metrics.brier(gold[i], probs[i], LABELS),
        "ece": lambda i: metrics.ece_top_label(gold[i], probs[i], LABELS),
        "aurc": lambda i: metrics.aurc(gold[i], probs[i], LABELS),
        "coverage_at_95": lambda i: metrics.coverage_at_precision(gold[i], probs[i], LABELS),
    }
    return functions[name]


def save(name: str, oof: pd.DataFrame, params: list, summary: dict) -> Path:
    out = runs_dir()
    out.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out / f"oof_{name}.csv", index=False)
    (out / f"summary_{name}.json").write_text(
        json.dumps({"system": name, "fold_params": params, **summary}, indent=2) + "\n"
    )
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
    qa, qb = probabilities(a), probabilities(b)
    if qa is not None and qb is not None:
        out["aurc"] = bootstrap.paired_difference(
            lambda i: metrics.aurc(gold[i], qa[i], LABELS),
            lambda i: metrics.aurc(gold[i], qb[i], LABELS), n, n_resamples)  # fmt: skip
    return out
