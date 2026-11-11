"""Tables and figures of the phase-1 report, regenerated from runs/phase1/<set>/results.json.

Writes docs/reports/phase1/figures/*.png and docs/reports/phase1/TEST_TABLES.md (generated;
REPORT.md is the hand-written narrative that cites them). Static PNGs, light surface; colors
from the reference categorical palette in fixed slot order, with a legend and direct labels.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from reqcore.paths import repo_root

from reqbench.eval import cv, heldout, metrics
from reqbench.models.baselines import LABELS

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
RELIABILITY = ("J3_jev_lr", "J1_jev_choice", "J4_jev_hierarchical", "B5_haiku_k5",
               "B4_deberta_v3_small_s1")  # fmt: skip
RISK = ("J3_jev_lr_cal", "J1_jev_choice_cal", "B5_haiku_k5_cal", "B4_deberta_v3_small_s1_cal",
        "B3_bge_small_lr_cal")  # fmt: skip
NAMES = {
    "B0_majority": "B0 majority", "B1_keywords": "B1 keywords", "B2_tfidf_lr": "B2 TF-IDF+LR",
    "B3_bge_small_lr": "B3 bge-small+LR", "B5_haiku_k5": "B5 Haiku 5.5 (k=5)",
    "B5x_haiku_single": "B5x Haiku, 1 sample", "B6_opus": "B6 Opus 5.5 (ceiling)",
    "B4x_deberta_seed_ensemble": "B4x DeBERTa seed ensemble",
    "J1_jev_choice": "J1 Jev Choice", "J2_jev_choice_rot_avg": "J2 Jev 3-order avg",
    "J3_jev_lr": "J3 LR on Jev", "J4_jev_hierarchical": "J4 Jev hierarchical",
    "J4x_jev_hierarchical_rot_avg": "J4x hierarchical, 3 orders",
}  # fmt: skip


def name(system: str) -> str:
    base, cal = (system[:-4], " + cal.") if system.endswith("_cal") else (system, "")
    if base.startswith("B4_deberta_v3_small_s"):
        return f"B4 DeBERTa seed {base[-1]}{cal}"
    return NAMES.get(base, base) + cal


def report_dir() -> Path:
    return repo_root() / "docs" / "reports" / "phase1"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = metrics.N_BINS):
    order = np.argsort(conf, kind="stable")
    chunks = [c for c in np.array_split(order, min(n_bins, len(conf))) if len(c)]
    return (np.array([conf[c].mean() for c in chunks]),
            np.array([correct[c].mean() for c in chunks]))  # fmt: skip


def reliability_figure(frames: dict[str, pd.DataFrame], out: Path, title: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    systems = [s for s in RELIABILITY if s in frames]
    fig, axes = plt.subplots(1, len(systems), figsize=(3.2 * len(systems), 3.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, system in zip(np.atleast_1d(axes), systems, strict=True):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
        for variant, color, label in ((system, MUTED, "raw"),
                                      (f"{system}_cal", SLOTS[0], "calibrated")):  # fmt: skip
            if variant not in frames:
                continue
            f = frames[variant]
            probs = cv.probabilities(f)
            correct = (probs.argmax(axis=1) == np.array(
                [LABELS.index(g) for g in f["gold"]])).astype(float)  # fmt: skip
            x, y = _bins(probs.max(axis=1), correct)
            ece = metrics.ece_top_label(f["gold"], probs, LABELS)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4,
                    label=f"{label} (ECE {ece:.3f})")  # fmt: skip
        ax.set_title(name(system), fontsize=9, color=INK)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("confidence (top label)", fontsize=8, color=MUTED)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("accuracy", fontsize=8, color=MUTED)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def risk_coverage_figure(frames: dict[str, pd.DataFrame], thresholds: dict, out: Path,
                         title: str) -> Path:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhline(1 - metrics.TARGET_PRECISION, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.01, 1 - metrics.TARGET_PRECISION + 0.005, "5 % error (95 % precision)",
            fontsize=7, color=MUTED)  # fmt: skip
    ends: list[float] = []
    for color, system in zip(SLOTS, [s for s in RISK if s in frames], strict=False):
        f = frames[system]
        probs = cv.probabilities(f)
        risk = metrics.risk_coverage(f["gold"], probs, LABELS)
        coverage = np.arange(1, len(risk) + 1) / len(risk)
        aurc = float(risk.mean())
        ax.plot(coverage, risk, color=color, linewidth=2,
                label=f"{name(system)} (AURC {aurc:.3f})")  # fmt: skip
        y = float(risk[-1])
        while any(abs(y - e) < 0.012 for e in ends):  # nudge end labels apart
            y += 0.012
        ends.append(y)
        ax.annotate(name(system).split(" ")[0], (1.0, risk[-1]), xytext=(1.012, y),
                    textcoords="data", fontsize=7, color=INK, va="center")  # fmt: skip
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, None)
    ax.set_xlabel("coverage (share decided automatically, most confident first)",
                  fontsize=8, color=MUTED)  # fmt: skip
    ax.set_ylabel("error rate of the decided share", fontsize=8, color=MUTED)
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def _fmt(ci: dict) -> str:
    v, (lo, hi) = ci["value"], ci["ci95"]
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def tables(results: dict) -> str:
    s, set_name = results["summaries"], results["set"]
    lines = [f"# Phase 1, {set_name}: generated tables", "",
             f"Generated by `uv run reqbench reproduce` from `runs/phase1/{set_name}/results.json`"
             f" (n = {results['n']}). Do not edit by hand.", "",
             "## All systems", "",
             "| System | Macro-F1 [95 % CI] | Level-1 F1 | Level-2 F1 | Accuracy | ECE | AURC "
             "| Cov. @ 95 % | Review coverage (precision) |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for system, v in s.items():
        ci = v["ci"]
        review = v.get("review")
        rev = "—" if review is None else (
            "0 (no threshold)" if review["threshold"] is None
            else f"{review['coverage']:.3f} ({review['precision']:.3f})"
            if review["precision"] is not None else f"{review['coverage']:.3f}")  # fmt: skip
        lines.append(
            f"| {name(system)} | {_fmt(ci['macro_f1'])} | {v['level1_f1']:.3f} "
            f"| {v['level2_macro_f1']:.3f} | {v['accuracy']:.3f} "
            f"| {v['ece']:.3f} | {v['aurc']:.3f} | {v['coverage_at_95']:.3f} | {rev} |"
            if "ece" in v else
            f"| {name(system)} | {_fmt(ci['macro_f1'])} | {v['level1_f1']:.3f} "
            f"| {v['level2_macro_f1']:.3f} | {v['accuracy']:.3f} | — | — | — | — |")  # fmt: skip
    lines += ["", "## Paired differences: Jev variant − system (macro-F1, 10,000 resamples)", "",
              "| Jev | vs | Δ macro-F1 [95 % CI] | Holm p | Δ AURC [95 % CI] | McNemar p |",
              "|---|---|---|---|---|---|"]  # fmt: skip
    for j, comps in results["comparisons"].items():
        for other, d in comps.items():
            aurc = _fmt(d["aurc"]) if "aurc" in d else "—"
            lines.append(f"| {name(j)} | {name(other)} | {_fmt(d['macro_f1'])} "
                         f"| {d['macro_f1'].get('p_holm', float('nan')):.4f} | {aurc} "
                         f"| {d['mcnemar']['p_exact']:.4f} |")  # fmt: skip
    lines += ["", "## Per-class F1", "", "| System | " + " | ".join(LABELS) + " |",
              "|---|" + "---|" * len(LABELS)]  # fmt: skip
    for system, v in s.items():
        if system.endswith("_cal"):
            continue
        lines.append(f"| {name(system)} | " + " | ".join(
            f"{v['per_class'][k]['f1']:.3f}" for k in LABELS) + " |")  # fmt: skip
    return "\n".join(lines) + "\n"


def write_report(set_name: str = "test") -> Path:
    path = heldout.out_dir(set_name) / "results.json"
    out = report_dir()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    dev_frames = {}
    for system in RELIABILITY:
        dev_frames[system] = pd.read_csv(cv.runs_dir() / f"oof_{system}.csv")
        dev_frames[system] = dev_frames[system][~dev_frames[system]["is_example"]]
        cal = cv.runs_dir() / f"cal_{system}.csv"
        if cal.exists():
            dev_frames[f"{system}_cal"] = pd.read_csv(cal)
    reliability_figure(dev_frames, figs / "reliability_dev.png",
                       "Reliability on dev (nested-CV calibration, n = 437)")  # fmt: skip
    if not path.exists():
        return figs
    results = json.loads(path.read_text())
    frames = {p.stem[5:]: pd.read_csv(p) for p in heldout.out_dir(set_name).glob("pred_*.csv")}
    thresholds = json.loads((repo_root() / "frozen/phase1/thresholds.json").read_text())
    reliability_figure(frames, figs / f"reliability_{set_name}.png",
                       f"Reliability on {set_name} (frozen calibrators, n = {results['n']})")
    risk_coverage_figure(frames, thresholds, figs / f"risk_coverage_{set_name}.png",
                         f"Risk-coverage on {set_name} (calibrated, n = {results['n']})")
    target = out / f"{set_name.upper()}_TABLES.md"
    target.write_text(tables(results))
    return target
