"""Phase 1: B5 (Claude Haiku 5.5 zero-shot, k = 5) on dev, compared with J1, B4 and B3.

`run_haiku_dev(run=False)` is a dry run: it counts the calls missing from the inference cache
and estimates their cost (with and without the Batch API); it makes no API call. With
`run=True` it submits the missing calls as one Message Batch per prompt rotation, measures
latency with standard calls while the batches run, ingests the results and evaluates:

- runs/phase1/dev/haiku_answers.csv: one row per call (label, model, usage, latency);
- runs/phase1/dev/oof_B5*.csv and summary_B5*.json, in the layout of the other systems;
- runs/phase1/dev/haiku_report.json: comparisons, stability, order, cost, latency.

When every call is already cached, the dry run also evaluates (it costs nothing).
"""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from reqcore.budget import Budget
from reqcore.cache import InferenceCache
from reqcore.clients.claude import HAIKU
from reqcore.pricing import AS_OF, PRICES, cost_usd

from reqbench.data.splits import SEED
from reqbench.eval import bootstrap, cv, metrics
from reqbench.models import llm
from reqbench.models.baselines import LABELS

B4_SEEDS = tuple(f"B4_deberta_v3_small_s{k}" for k in range(1, 6))
REFERENCES = ("J1_jev_choice", "B3_bge_small_lr") + B4_SEEDS
MAIN = "B5_haiku_k5"
SINGLE = "B5x_haiku_single"  # exploratory: one sample, hard label (what k = 5 adds)
# Mean tokens of the 20 smoke-test Haiku calls (2026-10-10), used only if nothing is cached.
SMOKE_TOKENS = (1209.0, 11.0)


def latency_rows(rows: pd.DataFrame, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 2)
    return rows.iloc[np.sort(rng.choice(len(rows), size=min(n, len(rows)), replace=False))]


def latency_calls(rows: pd.DataFrame) -> list[llm.Call]:
    return [llm.Call(r.req_id, r.text, 0, s) for r in rows.itertuples()
            for s in llm.LATENCY_SAMPLES]  # fmt: skip


def estimate_llm(calls, lat_calls, clfs, budget, model: str = HAIKU.model,
                 fallback_tokens: tuple[float, float] = SMOKE_TOKENS) -> dict:  # fmt: skip
    """Dry run for Claude calls: cache misses and their cost with and without the Batch API.
    Tokens per call = mean of the cached calls of this plan, else `fallback_tokens`."""
    missing = [c for c in calls if clfs[c.rotation].cached(c.text, c.sample_index) is None]
    lat_missing = [c for c in lat_calls if clfs[0].cached(c.text, c.sample_index) is None]
    seen = [clfs[c.rotation].cached(c.text, c.sample_index) for c in calls + lat_calls]
    seen = [r for r in seen if r is not None and r.input_tokens]
    tokens_in, tokens_out = (
        (float(np.mean([r.input_tokens for r in seen])),
         float(np.mean([r.output_tokens for r in seen]))) if seen else fallback_tokens
    )  # fmt: skip

    def usd(n_calls: int, batch: bool) -> float:
        return n_calls * cost_usd(model, round(tokens_in), round(tokens_out), batch)

    pending = [s["batch_id"] for c in clfs.values() for s in c.pending_batches()]
    return {
        "model": model, "prompt": {r: c.prompt.version for r, c in clfs.items()},
        "question_set_hashes": {r: c.question_set_hash for r, c in clfs.items()},
        "requirements": len({c.req_id for c in calls}),
        "batch_calls_total": len(calls), "batch_calls_cached": len(calls) - len(missing),
        "batch_calls_to_make": len(missing),
        "latency_requirements": len({c.req_id for c in lat_calls}),
        "latency_calls_total": len(lat_calls), "latency_calls_to_make": len(lat_missing),
        "pending_batches": pending,
        "tokens_per_call": {"input": tokens_in, "output": tokens_out,
                            "source": "cache" if seen else "fallback"},
        "expected_usd": {
            "batch_calls_with_batch_api": usd(len(missing), True),
            "batch_calls_without_batch_api": usd(len(missing), False),
            "latency_calls_standard": usd(len(lat_missing), False),
            "total": usd(len(missing), True) + usd(len(lat_missing), False),
        },
        "worst_case_usd": (sum(clfs[c.rotation].estimate_cost(c.text, batch=True)
                               for c in missing)
                           + sum(clfs[0].estimate_cost(c.text) for c in lat_missing)),
        "prices_as_of": AS_OF, "budget_bucket": PRICES[model].budget_bucket,
        "budget_remaining_usd": budget.remaining(),
    }  # fmt: skip


def _measure_latency(lat_calls, clf, log=print) -> None:
    """Standard calls, one requirement at a time, its K samples concurrently (as a service
    computing B5 would). Cached calls are skipped, so latencies are never cache hits."""
    by_req: dict[str, list[llm.Call]] = {}
    for c in lat_calls:
        by_req.setdefault(c.req_id, []).append(c)
    with ThreadPoolExecutor(max_workers=len(llm.LATENCY_SAMPLES)) as pool:
        for i, group in enumerate(by_req.values(), 1):
            todo = [c for c in group if clf.cached(c.text, c.sample_index) is None]
            list(pool.map(lambda c: clf.classify(c.text, c.sample_index), todo))
            if i % 10 == 0:
                log(f"  latency: {i}/{len(by_req)} requirements")


def _collect(calls, clfs) -> pd.DataFrame:
    rows, missing = [], 0
    for c in calls:
        result = clfs[c.rotation].cached(c.text, c.sample_index)
        if result is None:
            missing += 1
        else:
            rows.append(llm.to_row(c, clfs[c.rotation], result))
    if missing:
        raise RuntimeError(
            f"{missing} calls are still missing from the cache (errored or expired batch "
            "requests); re-run with --run to resubmit them."
        )
    frame = pd.DataFrame(rows)
    bad = sorted(set(frame["model"]) - {HAIKU.model})
    if bad:
        raise RuntimeError(f"answers from unexpected models: {bad}")
    return frame.sort_values(["req_id", "rotation", "sample_index"]).reset_index(drop=True)


def _usage(answers: pd.DataFrame, latency: pd.DataFrame, k: int) -> dict:
    model = HAIKU.model
    tin, tout = float(answers["input_tokens"].mean()), float(answers["output_tokens"].mean())
    per_call = {b: cost_usd(model, round(tin), round(tout), b) for b in (False, True)}
    per_req = latency.groupby("req_id")["latency_ms"].max()  # K concurrent calls
    return {
        "calls": len(answers), "calls_via_batch": int(answers["batch"].sum()),
        "models_reported": sorted(answers["model"].unique()),
        "stop_reasons": answers["stop_reason"].value_counts().to_dict(),
        "invalid_labels": int(answers["label"].isna().sum()),
        "mean_input_tokens": tin, "mean_output_tokens": tout,
        "spent_usd_recorded": float(answers["cost_usd"].sum() + latency["cost_usd"].sum()),
        "usd_per_1000_requirements": {
            "B5_k5_standard": 1000 * k * per_call[False],
            "B5_k5_batch": 1000 * k * per_call[True],
            "single_call_standard": 1000 * per_call[False],
            "single_call_batch": 1000 * per_call[True],
            "prices_as_of": AS_OF,
        },
        "latency": {
            "calls": len(latency), "requirements": int(per_req.size),
            "per_call_ms_p50": float(latency["latency_ms"].quantile(0.5)),
            "per_call_ms_p95": float(latency["latency_ms"].quantile(0.95)),
            "per_requirement_k5_concurrent_ms_p50": float(per_req.quantile(0.5)),
            "per_requirement_k5_concurrent_ms_p95": float(per_req.quantile(0.95)),
            "note": "standard calls, never cache hits; per requirement = slowest of the K "
                    "concurrent calls",
        },
    }  # fmt: skip


def run_haiku_dev(run: bool = False, latency_n: int = 30, poll_seconds: float = 30.0,
                  n_resamples: int = bootstrap.N_RESAMPLES, log=print) -> dict:  # fmt: skip
    dev = cv.load_dev()
    rows = dev[~dev["is_example"]].reset_index(drop=True)
    cache, budget = InferenceCache(), Budget()
    clfs = llm.classifiers(llm.ROTATIONS, cache=cache, budget=budget)
    calls = llm.plan_calls(rows)
    lat_calls = latency_calls(latency_rows(rows, latency_n))
    estimate = estimate_llm(calls, lat_calls, clfs, budget)
    work_left = (estimate["batch_calls_to_make"] or estimate["latency_calls_to_make"]
                 or estimate["pending_batches"])  # fmt: skip
    if work_left and not run:
        return {"dry_run": estimate}

    if run and work_left:
        submitted = {}
        for rotation, items in llm.batch_items(calls).items():
            if (batch_id := clfs[rotation].submit_batch(items)) is not None:
                submitted[batch_id] = rotation
                log(f"rotation {rotation}: submitted batch {batch_id}")
        _measure_latency(lat_calls, clfs[0], log)
        for rotation, clf in clfs.items():
            for state in clf.pending_batches():
                outcome = clf.collect_batch(state["batch_id"], poll_seconds, log)
                log(f"rotation {rotation}: batch {state['batch_id']} -> {outcome}")

    answers = _collect(calls, clfs)
    latency = pd.DataFrame([llm.to_row(c, clfs[0], clfs[0].cached(c.text, c.sample_index))
                            for c in lat_calls])  # fmt: skip
    out_dir = cv.runs_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    answers.to_csv(out_dir / "haiku_answers.csv", index=False)
    latency.to_csv(out_dir / "haiku_latency.csv", index=False)

    probs = llm.frequency_probs(answers, 0, tuple(range(llm.K)))
    single = llm.frequency_probs(answers, 0, (0,))
    systems = {MAIN: llm.to_oof(probs, rows), SINGLE: llm.to_oof(single, rows, with_probs=False)}
    summaries = {}
    for name, oof in systems.items():
        summaries[name] = cv.evaluate(oof, n_resamples=n_resamples)
        cv.save(name, oof, [], summaries[name])

    references = {ref: pd.read_csv(out_dir / f"oof_{ref}.csv") for ref in REFERENCES
                  if (out_dir / f"oof_{ref}.csv").exists()}  # fmt: skip
    comparisons = {name: {ref: cv.compare(oof, r, n_resamples) for ref, r in references.items()}
                   for name, oof in systems.items()}  # fmt: skip
    adjusted = bootstrap.holm({ref: comparisons[MAIN][ref]["macro_f1"]["p_two_sided"]
                               for ref in references})  # fmt: skip
    for ref, p in adjusted.items():
        comparisons[MAIN][ref]["macro_f1"]["p_holm"] = p

    report = {
        "estimate": estimate,
        "usage": _usage(answers, latency, llm.K),
        "n_valid_samples_hist": probs["n_valid"].value_counts().sort_index().to_dict(),
        "stability": llm.stability(answers),
        "order": llm.order_sensitivity(answers),
        "comparisons": comparisons,
        "summaries": {n: {k: s[k] for k in ("n", "ci", "per_fold_macro_f1")}
                      for n, s in summaries.items()},  # fmt: skip
        "confusion_B5": metrics.confusion(systems[MAIN]["gold"], systems[MAIN]["pred"],
                                          LABELS).tolist(),  # fmt: skip
        "labels": list(LABELS),
    }
    (out_dir / "haiku_report.json").write_text(
        json.dumps(report, indent=2, default=_json_default) + "\n")  # fmt: skip
    return report


def _json_default(value):
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    raise TypeError(type(value))
