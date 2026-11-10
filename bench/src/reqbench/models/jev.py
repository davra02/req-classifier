"""Jev variants J1-J4 on dev (plan §3.2, §6.2) and their robustness measures (§6.3).

One request per (requirement, option rotation, sample) carries every question of
`jev_type_v1` (req_type, nfr_subtype, is_functional and the two gate Nouls), so J1 and J4 come
from the same call:

- J1: `req_type` at rotation 0, sample 0 (Jev "as is").
- J2: `req_type` probabilities averaged over rotations 0, 1, 2 (sample 0).
- J4: hierarchical, rotation 0: P(functional) = p, P(k) = (1 - p) * P(k | NFR), with p the
  `is_functional` Noul and P(k | NFR) the `nfr_subtype` Choice.
- J3: logistic regression on every Jev output averaged over the three rotations, trained with
  the same project-grouped dev folds and grid as B3 (Jev itself is never trained).

Rotation 0 is also sampled `n_samples` times (sample_index 1.. carry a `uid` in the state, so
they are independent draws) to measure run-to-run stability. The few-shot example rows of
examples_v1 are not queried: Jev sees them in its criteria and they are excluded from metrics.
Only `split == "dev"` rows ever reach this module.
"""

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
from reqcore.clients.jev import PINNED_MODEL, JevClient, JevResult, ModelVersionMismatchError
from reqcore.pricing import PRICES, cost_usd
from reqcore.questions import JevQuestionSet, load_taxonomy, render_jev_questions

from reqbench.data.examples import EXAMPLES_VERSION
from reqbench.models.baselines import LABELS, _GridLR

QUESTION_SET = "jev_type_v1"
ROTATIONS: tuple[int, ...] = (0, 1, 2)
N_SAMPLES = 5  # sample 0 + 4 repeats of rotation 0
NFR: tuple[str, ...] = load_taxonomy().level1["non_functional"]
NOULS = ("is_functional", "is_requirement", "addresses_ai")
J3_FEATURES = (
    [f"rt_{label}" for label in LABELS] + [f"ns_{label}" for label in NFR]
    + [f"nf_{name}" for name in NOULS]
)  # fmt: skip


@dataclass(frozen=True)
class Call:
    req_id: str
    text: str
    rotation: int
    sample_index: int

    @property
    def state(self) -> dict:
        return {"requirement": self.text}

    @property
    def nonce(self) -> str | None:
        return None if self.sample_index == 0 else f"s{self.sample_index}"


def question_sets(rotations: Iterable[int] = ROTATIONS) -> dict[int, JevQuestionSet]:
    return {
        r: render_jev_questions(QUESTION_SET, rotation=r, examples=EXAMPLES_VERSION)
        for r in rotations
    }


def plan_calls(rows: pd.DataFrame, rotations: Sequence[int] = ROTATIONS,
               n_samples: int = N_SAMPLES) -> list[Call]:  # fmt: skip
    calls = []
    for row in rows.itertuples():
        calls += [Call(row.req_id, row.text, r, 0) for r in rotations]
        calls += [Call(row.req_id, row.text, 0, s) for s in range(1, n_samples)]
    return calls


# ----- running ----------------------------------------------------------------------------


def _flatten(call: Call, qset: JevQuestionSet, result: JevResult) -> dict:
    if result.model_reported != PINNED_MODEL:
        raise ModelVersionMismatchError(
            f"{call.req_id}: cached/returned model {result.model_reported!r} != {PINNED_MODEL}"
        )
    a = result.answers
    row = {
        "req_id": call.req_id, "rotation": call.rotation, "sample_index": call.sample_index,
        "question_set_hash": qset.hash, "model": result.model_reported,
        "input_tokens": result.input_tokens, "latency_ms": result.latency_ms,
        "cached": result.cached, "rt_choice": a["req_type"]["choice"],
        "ns_choice": a["nfr_subtype"]["choice"],
    }  # fmt: skip
    row |= {f"rt_{k}": float(a["req_type"]["probabilities"][k]) for k in LABELS}
    row |= {f"ns_{k}": float(a["nfr_subtype"]["probabilities"][k]) for k in NFR}
    row |= {f"nf_{name}": float(a[name]["noul"]) for name in NOULS}
    return row


def estimate(calls: list[Call], client: JevClient, qsets: dict[int, JevQuestionSet],
             tokens_per_call: float | None = None) -> dict:  # fmt: skip
    """Dry run: how many calls are missing from the cache and what they would cost."""
    missing, tokens_seen = [], []
    for c in calls:
        hit = client.cached(c.state, qsets[c.rotation], c.sample_index, c.nonce)
        if hit is None:
            missing.append(c)
        elif hit.input_tokens:
            tokens_seen.append(hit.input_tokens)
    if tokens_per_call is None and tokens_seen:
        tokens_per_call = float(np.mean(tokens_seen))  # real usage of the cached calls
    worst = sum(client.estimate_cost(c.state, qsets[c.rotation]) for c in missing)
    out = {
        "question_set": QUESTION_SET, "model": client.model,
        "question_set_hashes": {r: q.hash for r, q in qsets.items()},
        "requirements": len({c.req_id for c in calls}), "calls_total": len(calls),
        "calls_cached": len(calls) - len(missing), "calls_to_make": len(missing),
        "worst_case_usd": worst, "budget_remaining_usd": client.budget.remaining(),
        "budget_bucket": PRICES[client.model].budget_bucket,
    }  # fmt: skip
    if tokens_per_call:
        tokens = tokens_per_call * len(missing)
        out |= {"tokens_per_call": tokens_per_call, "expected_input_tokens": int(tokens),
                "expected_usd": cost_usd(client.model, int(tokens), 0)}  # fmt: skip
    return out


def collect(calls: list[Call], client: JevClient, qsets: dict[int, JevQuestionSet],
            run: bool, workers: int = 8, progress: bool = True) -> pd.DataFrame:  # fmt: skip
    """One row per call. Without `run`, only cached calls are returned (no API traffic).

    Any answer from a model other than the pinned one aborts the whole run.
    """
    rows, todo = [], []
    for call in calls:
        hit = client.cached(call.state, qsets[call.rotation], call.sample_index, call.nonce)
        if hit is not None:
            rows.append(_flatten(call, qsets[call.rotation], hit))
        else:
            todo.append(call)
    if todo and run:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(client.ask, c.state, qsets[c.rotation], c.sample_index, c.nonce): c
                for c in todo
            }
            try:
                for done, future in enumerate(as_completed(futures), 1):
                    call = futures[future]
                    rows.append(_flatten(call, qsets[call.rotation], future.result()))
                    if progress and done % 250 == 0:
                        print(f"  {done}/{len(todo)} calls")
            except BaseException:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["req_id", "rotation", "sample_index"]).reset_index(drop=True)


# ----- variants ---------------------------------------------------------------------------


def _select(answers: pd.DataFrame, rotations: Sequence[int], sample_index: int = 0):
    keep = answers["rotation"].isin(rotations) & (answers["sample_index"] == sample_index)
    return answers[keep]


def _renorm(probs: np.ndarray) -> np.ndarray:
    total = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, total, out=np.full_like(probs, 1 / probs.shape[1]),
                     where=total > 0)  # fmt: skip


def flat_probs(answers: pd.DataFrame, rotations: Sequence[int] = (0,),
               sample_index: int = 0) -> pd.DataFrame:  # fmt: skip
    """J1 (rotation 0) / J2 (rotations 0-2): mean `req_type` probabilities, rows = req_id."""
    columns = [f"rt_{k}" for k in LABELS]
    mean = _select(answers, rotations, sample_index).groupby("req_id")[columns].mean()
    return pd.DataFrame(_renorm(mean.to_numpy()), index=mean.index, columns=list(LABELS))


def hierarchical_probs(answers: pd.DataFrame, rotations: Sequence[int] = (0,),
                       sample_index: int = 0) -> pd.DataFrame:  # fmt: skip
    """J4: P(functional) = p; P(k) = (1 - p) * P(k | NFR), averaged over `rotations`."""
    columns = ["nf_is_functional"] + [f"ns_{k}" for k in NFR]
    mean = _select(answers, rotations, sample_index).groupby("req_id")[columns].mean()
    p = mean["nf_is_functional"].to_numpy()[:, None]
    sub = _renorm(mean[[f"ns_{k}" for k in NFR]].to_numpy())
    probs = np.zeros((len(mean), len(LABELS)))
    probs[:, LABELS.index("functional")] = p[:, 0]
    for j, leaf in enumerate(NFR):
        probs[:, LABELS.index(leaf)] = (1 - p[:, 0]) * sub[:, j]
    return pd.DataFrame(probs, index=mean.index, columns=list(LABELS))


def features(answers: pd.DataFrame, rotations: Sequence[int] = ROTATIONS) -> pd.DataFrame:
    """J3 inputs: every Jev output averaged over the rotations (sample 0), rows = req_id."""
    return _select(answers, rotations).groupby("req_id")[J3_FEATURES].mean()


def to_oof(probs: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """OOF frame in the layout of reqbench.eval.cv.run_cv (argmax ties -> taxonomy order)."""
    out = rows[["req_id", "project_id", "fold", "leaf", "is_example"]].rename(
        columns={"leaf": "gold"})  # fmt: skip
    matrix = probs.loc[out["req_id"], list(LABELS)].to_numpy()
    out["pred"] = [LABELS[i] for i in matrix.argmax(axis=1)]
    for k, label in enumerate(LABELS):
        out[f"p_{label}"] = matrix[:, k]
    return out.sort_values("req_id").reset_index(drop=True)


class JevLR(_GridLR):
    """J3: logistic regression on Jev outputs; same grid and inner CV as B3."""

    name = "J3_jev_lr"
    grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0, 30.0], "class_weight": [None, "balanced"]}

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table  # rows indexed by requirement text

    def _estimator(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=5000)

    def _features(self, texts):
        return self.table.loc[list(texts)].to_numpy()


# ----- robustness -------------------------------------------------------------------------


def _labels(probs: np.ndarray, names: Sequence[str]) -> np.ndarray:
    return np.array(names)[probs.argmax(axis=1)]


def stability(answers: pd.DataFrame) -> dict:
    """Run-to-run variation of rotation 0 over its samples (plan §6.3 robustness (a))."""
    r0 = answers[answers["rotation"] == 0]
    samples = sorted(r0["sample_index"].unique())
    if len(samples) < 2:
        return {"n_samples": len(samples)}
    per_sample = {
        "J1": {s: _labels(flat_probs(r0, (0,), s).to_numpy(), LABELS) for s in samples},
        "J4": {s: _labels(hierarchical_probs(r0, (0,), s).to_numpy(), LABELS) for s in samples},
    }
    out: dict = {"n_samples": len(samples),
                 "n_requirements": len(per_sample["J1"][samples[0]])}  # fmt: skip
    for name, labels in per_sample.items():
        base = labels[samples[0]]
        stack = np.stack([labels[s] for s in samples])
        pairs = [(a, b) for i, a in enumerate(samples) for b in samples[i + 1:]]
        out[name] = {
            "flip_rate_vs_sample0": float(np.mean([labels[s] != base for s in samples[1:]])),
            "flip_rate_pairwise": float(np.mean([labels[a] != labels[b] for a, b in pairs])),
            "share_rows_any_change": float(np.mean((stack != stack[0]).any(axis=0))),
        }
    wide = r0.set_index(["req_id", "sample_index"])
    out["prob_std_mean"] = {
        column: float(wide[column].groupby("req_id").std(ddof=1).mean())
        for column in [f"rt_{k}" for k in LABELS] + [f"nf_{n}" for n in NOULS]
    }
    out["is_functional_std_mean"] = out["prob_std_mean"]["nf_is_functional"]
    return out


def order_sensitivity(answers: pd.DataFrame, rotations: Sequence[int] = ROTATIONS) -> dict:
    """Option-order effects over the rotations (plan §6.3 robustness (b)).

    - flip rates of the req_type (J1-style) and J4 labels between rotations;
    - first-position lift: for each class that leads the option list in some rotation, the
      mean probability it gets when listed first minus when not (paired by requirement).
    """
    qsets = question_sets(rotations)
    flat = {r: _labels(flat_probs(answers, (r,)).to_numpy(), LABELS) for r in rotations}
    hier = {r: _labels(hierarchical_probs(answers, (r,)).to_numpy(), LABELS) for r in rotations}
    pairs = [(a, b) for i, a in enumerate(rotations) for b in rotations[i + 1:]]
    out: dict = {"rotations": list(rotations)}
    for name, labels in (("req_type", flat), ("J4", hier)):
        stack = np.stack([labels[r] for r in rotations])
        out[name] = {
            "flip_rate_pairwise": float(np.mean([labels[a] != labels[b] for a, b in pairs])),
            "share_rows_any_change": float(np.mean((stack != stack[0]).any(axis=0))),
        }
    lifts = {}
    for question, prefix in (("req_type", "rt_"), ("nfr_subtype", "ns_")):
        first = {r: next(iter(qsets[r].questions[question]["criteria"])) for r in rotations}
        sample0 = answers[answers["sample_index"] == 0]
        wide = sample0.pivot(index="req_id", columns="rotation")
        lifts[question] = {}
        for r, label in first.items():
            when_first = wide[(f"{prefix}{label}", r)]
            others = wide[[(f"{prefix}{label}", o) for o in rotations if o != r]].mean(axis=1)
            chosen = sample0.loc[sample0["rotation"] == r, f"{prefix}choice"]
            share_first = float(np.mean(chosen == label))
            lifts[question][label] = {
                "first_in_rotation": int(r),
                "mean_prob_when_first": float(when_first.mean()),
                "mean_prob_otherwise": float(others.mean()),
                "lift": float((when_first - others).mean()),
                "share_chosen_when_first": share_first,
            }
    out["first_position"] = lifts
    return out


def usage_summary(answers: pd.DataFrame) -> dict:
    fresh = answers[~answers["cached"]]
    tokens = float(answers["input_tokens"].mean())
    per_call = cost_usd(PINNED_MODEL, int(round(tokens)), 0)
    return {
        "calls": len(answers), "calls_made_this_run": len(fresh),
        "models_reported": sorted(answers["model"].unique()),
        "mean_input_tokens": tokens,
        "latency_ms_p50": float(fresh["latency_ms"].quantile(0.5)) if len(fresh) else None,
        "latency_ms_p95": float(fresh["latency_ms"].quantile(0.95)) if len(fresh) else None,
        "latency_ms_p50_all": float(answers["latency_ms"].quantile(0.5)),
        "latency_ms_p95_all": float(answers["latency_ms"].quantile(0.95)),
        "usd_per_1000_requirements": {"J1": 1000 * per_call, "J4": 1000 * per_call,
                                      "J2": 3000 * per_call, "J3": 3000 * per_call},
        "choice_vs_argmax_agreement": float(np.mean(
            answers["rt_choice"] == _labels(answers[[f"rt_{k}" for k in LABELS]].to_numpy(),
                                            LABELS))),
    }  # fmt: skip
