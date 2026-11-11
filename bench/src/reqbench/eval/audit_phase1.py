"""Support analyses for the phase-1 audit (docs/reports/phase1/AUDIT.md). No API calls.

`uv run reqbench audit` regenerates runs/phase1/audit/audit.json. Sections:

- `c1_dev`: J3 out-of-fold on dev with a fixed C (0.1, 1, 30; class_weight None, as the final
  model) instead of the C chosen per fold: raw ECE, ECE after the frozen temperature
  (T = 0.72) and the frozen review threshold (0.954). Dev only.
- `ece_noise_floor` (**exploratory, post hoc: reads test predictions**): ECE of a perfectly
  calibrated system with J3's test confidences (outcomes drawn as Bernoulli(confidence)).
- `adversarial_fisher`: two-sided Fisher exact tests on the adversarial flip counts (variants
  of dev requirements; B5x = Haiku sample 0, also read from the cache).
- `test_post_hoc` (**exploratory, post hoc: reads test predictions**): J3 vs B5x, J3/B5
  discordant pairs, J3 calibrated confidence vs accuracy, sample sizes for 50 % and 80 %
  power, and the hash of results.json against EVAL_LOG.jsonl.

This module only reads frozen code and data; it never writes into frozen/ or the test runs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from reqcore.paths import repo_root
from scipy.stats import fisher_exact, norm

from reqbench.eval import bootstrap, calibration, cv, final_models, heldout, metrics
from reqbench.models import jev, llm
from reqbench.models.baselines import LABELS

FROZEN_T = 0.7242388326505903
FROZEN_THRESHOLD = 0.954217088418
POST_HOC = "exploratory, post hoc: reads locked-test predictions after the one-time evaluation"


def audit_dir() -> Path:
    return repo_root() / "runs" / "phase1" / "audit"


class FixedJevLR(jev.JevLR):
    """J3 with one fixed hyperparameter setting (the grid search sees a single option)."""

    def __init__(self, table: pd.DataFrame, c: float) -> None:
        super().__init__(table)
        self.grid = {"C": [c], "class_weight": [None]}


def _temperature(probs: np.ndarray) -> np.ndarray:
    cal = calibration.Temperature()
    cal.t_ = FROZEN_T
    return cal.transform(probs)


def _selective(gold, probs) -> dict:
    pred = np.array([LABELS[i] for i in probs.argmax(axis=1)])
    sel = metrics.selective(gold, pred, probs, FROZEN_THRESHOLD)
    return {"coverage": sel["coverage"], "n_auto": sel["n_auto"], "precision": sel["precision"]}


def _probe(gold, probs) -> dict:
    scaled = _temperature(probs)
    return {
        "macro_f1": metrics.macro_f1(gold, [LABELS[i] for i in probs.argmax(axis=1)], LABELS),
        "ece_raw": metrics.ece_top_label(gold, probs, LABELS),
        "ece_with_frozen_T": metrics.ece_top_label(gold, scaled, LABELS),
        "mean_confidence_with_T": float(scaled.max(axis=1).mean()),
        "accuracy": float(np.mean(probs.argmax(axis=1) == [LABELS.index(g) for g in gold])),
        "frozen_threshold_on_T": _selective(gold, scaled),
    }


def c1_dev() -> dict:
    dev = cv.load_dev()
    rows = dev[~dev["is_example"]].reset_index(drop=True)
    answers = pd.read_csv(cv.runs_dir() / "jev_answers.csv")
    table = jev.features(answers, jev.ROTATIONS)
    table.index = rows.set_index("req_id").loc[table.index, "text"]
    out = {}
    frozen_oof = pd.read_csv(cv.runs_dir() / "oof_J3_jev_lr.csv")
    frozen_oof = frozen_oof[~frozen_oof["is_example"]]
    out["C_per_fold_(as_frozen)"] = _probe(frozen_oof["gold"].to_numpy(),
                                           cv.probabilities(frozen_oof))  # fmt: skip
    for c in (0.1, 1.0, 30.0):
        oof, _ = cv.run_cv(lambda c=c: FixedJevLR(table, c), rows)
        oof = oof[~oof["is_example"]]
        out[f"C={c:g}"] = _probe(oof["gold"].to_numpy(), cv.probabilities(oof))
    out["final_model_C"] = json.loads(
        (repo_root() / "frozen/phase1/final_models.json").read_text())["J3_jev_lr"]["params"]
    return out


def ece_noise_floor(n_sim: int = 10_000) -> dict:
    out = {"note": POST_HOC}
    for name in ("J3_jev_lr", "J3_jev_lr_cal"):
        f = pd.read_csv(heldout.out_dir("test") / f"pred_{name}.csv")
        probs = cv.probabilities(f)
        conf = probs.max(axis=1)
        observed = metrics.ece_top_label(f["gold"], probs, LABELS)
        rng = np.random.default_rng(bootstrap.SEED)
        sims = np.array([metrics._ece(conf, (rng.random(len(conf)) < conf).astype(float),
                                      metrics.N_BINS) for _ in range(n_sim)])  # fmt: skip
        out[name] = {"n": len(conf), "observed_ece": observed,
                     "perfect_calibration_ece_mean": float(sims.mean()),
                     "perfect_calibration_ece_p95": float(np.percentile(sims, 95)),
                     "share_below_0.05": float(np.mean(sims < 0.05)),
                     "p_value_observed_or_worse": float(np.mean(sims >= observed))}  # fmt: skip
    return out


def adversarial_fisher() -> dict:
    results = json.loads((heldout.out_dir("test") / "results.json").read_text())["adversarial"]
    adv = heldout.adversarial.variants()
    rows = heldout.load_texts("test")
    p = heldout.plan("test", rows)
    _, _, haiku, _, _ = heldout._clients()
    answers = heldout._llm_answers(p["haiku_adv"], haiku, heldout.HAIKU.model)
    single = llm.frequency_probs(answers, 0, (0,))["pred"]
    dev_b5x = pd.read_csv(cv.runs_dir() / "oof_B5x_haiku_single.csv").set_index("req_id")["pred"]
    b5x = heldout.adversarial.flip_rates(adv, dev_b5x, single)
    counts = {name: round(r["flip_rate"] * r["n_variants"]) for name, r in results.items()}
    counts["B5x_haiku_single"] = round(b5x["flip_rate"] * b5x["n_variants"])
    n = len(adv)
    tests = {}
    for a in ("J1_jev_choice", "J4_jev_hierarchical"):
        for b in ("B5_haiku_k5", "B5x_haiku_single"):
            _, pv = fisher_exact([[counts[a], n - counts[a]], [counts[b], n - counts[b]]])
            tests[f"{a} vs {b}"] = float(pv)
    return {"n_variants": n, "flips": counts, "B5x_detail": b5x, "fisher_two_sided_p": tests}


def _n_needed(n: int, half: float, gap: float, power: float) -> int | None:
    if gap <= 0:
        return None
    se = half / norm.ppf(0.975)
    z = norm.ppf(0.975) + norm.ppf(power)
    return int(np.ceil(n * (z * se / gap) ** 2))


def test_post_hoc(n_resamples: int = bootstrap.N_RESAMPLES) -> dict:
    d = heldout.out_dir("test")
    preds = {n: pd.read_csv(d / f"pred_{n}.csv") for n in
             ("J1_jev_choice", "J3_jev_lr", "J3_jev_lr_cal", "B5_haiku_k5", "B5x_haiku_single")}
    results = json.loads((d / "results.json").read_text())
    j3, b5 = preds["J3_jev_lr"], preds["B5_haiku_k5"]
    r3, r5 = j3["gold"] == j3["pred"], b5["gold"] == b5["pred"]
    cal = preds["J3_jev_lr_cal"]
    cal_probs = cv.probabilities(cal)
    comp = results["comparisons"]
    sizes = {}
    for label, j, ref, gap_fn in (
        ("separate J3 from B5 (CI excludes 0)", "J3_jev_lr", "B5_haiku_k5", lambda v: abs(v)),
        ("J1 non-inferior to B5 (lower bound > -0.03)", "J1_jev_choice", "B5_haiku_k5",
         lambda v: v + heldout.DELTA),
    ):  # fmt: skip
        m = comp[j][ref]["macro_f1"]
        half = (m["ci95"][1] - m["ci95"][0]) / 2
        sizes[label] = {p: _n_needed(len(j3), half, gap_fn(m["value"]), p) for p in (0.5, 0.8)}
    log = [json.loads(x) for x in heldout.log_path().read_text().splitlines() if x.strip()]
    logged = next(e["results_sha256"] for e in log if e["event"] == "completed")
    return {
        "note": POST_HOC,
        "J3_minus_B5x": cv.compare(j3, preds["B5x_haiku_single"], n_resamples)["macro_f1"],
        "J3_vs_B5_discordant": {"J3_right_B5_wrong": int((r3 & ~r5).sum()),
                                "J3_wrong_B5_right": int((~r3 & r5).sum())},
        "J3_cal_mean_confidence": float(cal_probs.max(axis=1).mean()),
        "J3_cal_accuracy": float(np.mean(cal["gold"] == cal["pred"])),
        "J3_cal_frozen_threshold": results["summaries"]["J3_jev_lr_cal"]["review"],
        "sample_size_needed_by_power": sizes,
        "B5x_cost_usd_per_1000": {"standard": results["cost"]["B5"]["usd_per_1000_standard"] / 5,
                                  "batch": results["cost"]["B5"]["usd_per_1000_batch"] / 5},
        "results_json_sha256_now": final_models.file_hash(d / "results.json"),
        "results_json_sha256_logged": logged,
    }  # fmt: skip


def ece_order_dependence(n_perm: int = 1000) -> dict:
    """Dev only: top-label ECE of J1 and B5 (tied confidences) under random row orders."""
    out = {}
    for name in ("J1_jev_choice", "B5_haiku_k5"):
        f = pd.read_csv(cv.runs_dir() / f"oof_{name}.csv")
        f = f[~f["is_example"]].reset_index(drop=True)
        rng = np.random.default_rng(bootstrap.SEED)
        values = []
        for _ in range(n_perm):
            g = f.iloc[rng.permutation(len(f))]
            values.append(metrics.ece_top_label(g["gold"].to_numpy(), cv.probabilities(g), LABELS))
        out[name] = {"as_stored": metrics.ece_top_label(f["gold"], cv.probabilities(f), LABELS),
                     "min": float(np.min(values)), "max": float(np.max(values)),
                     "mean": float(np.mean(values))}  # fmt: skip
    return out


def run_audit(n_resamples: int = bootstrap.N_RESAMPLES) -> dict:
    out = {"c1_dev": c1_dev(), "ece_noise_floor": ece_noise_floor(),
           "ece_order_dependence_dev": ece_order_dependence(),
           "adversarial_fisher": adversarial_fisher(), "test_post_hoc": test_post_hoc(n_resamples)}
    audit_dir().mkdir(parents=True, exist_ok=True)
    (audit_dir() / "audit.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return out
