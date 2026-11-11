"""B5: Claude Haiku 5.5 zero-shot on dev (plan §6.2) and its robustness measures (§6.3).

Prompt `llm_type_v1` with the examples of examples_v1 (the same definitions and examples Jev
sees), thinking disabled, effort low, output constrained to the 6-label enum.

Calls per dev requirement (examples excluded), mirroring Jev's 3,059:

- rotation 0 (taxonomy order), samples 0-4: B5. Haiku 5.5 rejects non-default temperatures,
  so the probabilities are the label frequencies over the k = 5 samples. The prediction is
  the most frequent label; a tie goes to the tied label drawn first (lowest sample index).
- rotations 1 and 2, sample 0: definition-order sensitivity, compared with the run-to-run
  flip rate of rotation 0 (the noise floor of a sampled model).

All of these go through the Message Batches API. Latency is measured apart, with standard
calls (sample indices `LATENCY_SAMPLES`) on a small random subset of dev.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from reqcore.clients.claude import HAIKU, BatchItem, ClaudeClassifier, ClaudeResult
from reqcore.questions import render_llm_prompt

from reqbench.data.examples import EXAMPLES_VERSION
from reqbench.models.baselines import LABELS

PROMPT = "llm_type_v1"
ROTATIONS: tuple[int, ...] = (0, 1, 2)
K = 5
LATENCY_SAMPLES: tuple[int, ...] = (100, 101, 102, 103, 104)  # never mixed with 0..K-1


@dataclass(frozen=True)
class Call:
    req_id: str
    text: str
    rotation: int
    sample_index: int


def classifiers(rotations: Sequence[int] = ROTATIONS, **kwargs) -> dict[int, ClaudeClassifier]:
    return {
        r: ClaudeClassifier(HAIKU, render_llm_prompt(PROMPT, EXAMPLES_VERSION, rotation=r),
                            **kwargs)  # fmt: skip
        for r in rotations
    }


def plan_calls(rows: pd.DataFrame, rotations: Sequence[int] = ROTATIONS, k: int = K) -> list[Call]:
    calls = []
    for row in rows.itertuples():
        calls += [Call(row.req_id, row.text, 0, s) for s in range(k)]
        calls += [Call(row.req_id, row.text, r, 0) for r in rotations if r != 0]
    return calls


def batch_items(calls: list[Call]) -> dict[int, list[BatchItem]]:
    out: dict[int, list[BatchItem]] = {}
    for c in calls:
        out.setdefault(c.rotation, []).append(BatchItem(c.text, c.sample_index))
    return out


def to_row(call: Call, clf: ClaudeClassifier, result: ClaudeResult) -> dict:
    return {
        "req_id": call.req_id, "rotation": call.rotation, "sample_index": call.sample_index,
        "label": result.label, "stop_reason": result.stop_reason,
        "model": result.model_reported, "question_set_hash": clf.question_set_hash,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd, "latency_ms": result.latency_ms, "batch": result.batch,
    }  # fmt: skip


# ----- B5 probabilities -------------------------------------------------------------------


def frequency_probs(answers: pd.DataFrame, rotation: int = 0,
                    samples: Sequence[int] = tuple(range(K))) -> pd.DataFrame:  # fmt: skip
    """Rows = req_id; columns = LABELS (frequencies) + `pred` + `n_valid`.

    Samples without a valid label (refusal, truncation) are left out of the denominator; a
    requirement with no valid sample gets uniform probabilities and the first label.
    """
    sel = answers[(answers["rotation"] == rotation) & answers["sample_index"].isin(samples)]
    records = {}
    for req_id, group in sel.sort_values("sample_index").groupby("req_id"):
        labels = [lab for lab in group["label"] if isinstance(lab, str) and lab in LABELS]
        counts = np.array([labels.count(lab) for lab in LABELS], dtype=float)
        if counts.sum() == 0:
            probs, pred = np.full(len(LABELS), 1 / len(LABELS)), LABELS[0]
        else:
            probs = counts / counts.sum()
            tied = {LABELS[i] for i in np.flatnonzero(counts == counts.max())}
            pred = next(lab for lab in labels if lab in tied)  # first drawn among the tied
        records[req_id] = {**dict(zip(LABELS, probs, strict=True)), "pred": pred,
                           "n_valid": len(labels)}  # fmt: skip
    return pd.DataFrame.from_dict(records, orient="index")


def to_oof(probs: pd.DataFrame, rows: pd.DataFrame, with_probs: bool = True) -> pd.DataFrame:
    """OOF frame in the layout of reqbench.eval.cv.run_cv."""
    out = rows[["req_id", "project_id", "fold", "leaf", "is_example"]].rename(
        columns={"leaf": "gold"})  # fmt: skip
    table = probs.loc[out["req_id"]]
    out["pred"] = table["pred"].to_numpy()
    if with_probs:
        for label in LABELS:
            out[f"p_{label}"] = table[label].to_numpy(dtype=float)
    return out.sort_values("req_id").reset_index(drop=True)


def sample_labels(answers: pd.DataFrame, rotation: int, sample_index: int) -> pd.Series:
    sel = answers[(answers["rotation"] == rotation) & (answers["sample_index"] == sample_index)]
    return sel.set_index("req_id")["label"].sort_index()


# ----- robustness -------------------------------------------------------------------------


def stability(answers: pd.DataFrame, k: int = K) -> dict:
    """Run-to-run variation of the k samples of rotation 0 (plan §6.3 robustness (a))."""
    labels = pd.DataFrame({s: sample_labels(answers, 0, s) for s in range(k)})
    stack = labels.to_numpy().T  # (k, n)
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    probs = frequency_probs(answers, 0, tuple(range(k)))
    return {
        "n_samples": k, "n_requirements": int(labels.shape[0]),
        "flip_rate_vs_sample0": float(np.mean([stack[s] != stack[0] for s in range(1, k)])),
        "flip_rate_pairwise": float(np.mean([stack[a] != stack[b] for a, b in pairs])),
        "share_rows_any_change": float(np.mean((stack != stack[0]).any(axis=0))),
        "share_unanimous": float(np.mean(probs[list(LABELS)].max(axis=1) == 1.0)),
        "top_frequency_histogram": {
            f"{v:.1f}": int(c) for v, c in
            probs[list(LABELS)].max(axis=1).round(1).value_counts().sort_index().items()
        },
    }  # fmt: skip


def order_sensitivity(answers: pd.DataFrame, rotations: Sequence[int] = ROTATIONS) -> dict:
    """Definition-order effects over the rotations, sample 0 (plan §6.3 robustness (b)).

    The flip rate between rotations includes sampling noise; `noise_floor` is the flip rate
    between samples 0 and 1 of rotation 0, i.e. what two draws with the *same* order give.
    `first_position`: for the label that leads the list in each rotation, the share of
    requirements predicted as that label when it is first vs. in the other rotations.
    """
    labels = {r: sample_labels(answers, r, 0) for r in rotations}
    pairs = [(a, b) for i, a in enumerate(rotations) for b in rotations[i + 1:]]
    stack = np.stack([labels[r].to_numpy() for r in rotations])
    first = {r: render_llm_prompt(PROMPT, EXAMPLES_VERSION, rotation=r).labels[0]
             for r in rotations}  # fmt: skip
    return {
        "rotations": list(rotations),
        "first_label": {int(r): lab for r, lab in first.items()},
        "flip_rate_pairwise": float(np.mean([labels[a].to_numpy() != labels[b].to_numpy()
                                             for a, b in pairs])),  # fmt: skip
        "share_rows_any_change": float(np.mean((stack != stack[0]).any(axis=0))),
        "noise_floor_same_order": float(np.mean(
            sample_labels(answers, 0, 0).to_numpy() != sample_labels(answers, 0, 1).to_numpy())),
        "first_position": {
            lab: {
                "first_in_rotation": int(r),
                "share_when_first": float(np.mean(labels[r] == lab)),
                "share_otherwise": float(np.mean([np.mean(labels[o] == lab)
                                                  for o in rotations if o != r])),
            } for r, lab in first.items()
        },
    }  # fmt: skip
