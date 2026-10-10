"""Phase-0 smoke test: real token counts, latency and pinned-version checks on a few dev items.

Without `run=True` it only prints a worst-case cost estimate and makes no API call.
"""

import json
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from reqcore.budget import Budget
from reqcore.cache import InferenceCache
from reqcore.clients.claude import HAIKU, OPUS, ClaudeClassifier
from reqcore.clients.jev import JevClient
from reqcore.paths import repo_root
from reqcore.questions import render_jev_questions, render_llm_prompt

from reqbench.data.build import load_processed
from reqbench.data.examples import EXAMPLES_VERSION, example_ids
from reqbench.data.splits import SEED, load_splits


def smoke_sample(n: int) -> pd.DataFrame:
    data = load_processed().merge(load_splits()[["req_id", "split"]], on="req_id")
    dev = data[(data["split"] == "dev") & ~data["req_id"].isin(example_ids())]
    rng = np.random.default_rng(SEED + 1)
    return dev.iloc[rng.choice(len(dev), size=n, replace=False)].reset_index(drop=True)


def _summary(name: str, rows: list[dict]) -> dict:
    fresh = [r for r in rows if not r["cached"]]
    latencies = [r["latency_ms"] for r in fresh]
    return {
        "system": name,
        "calls": len(rows),
        "fresh_calls": len(fresh),
        "mean_input_tokens": float(np.mean([r["input_tokens"] for r in rows])),
        "mean_output_tokens": float(np.mean([r.get("output_tokens") or 0 for r in rows])),
        "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "cost_usd": float(sum(r["cost_usd"] or 0 for r in fresh)),
        "accuracy_vs_gold_informative": float(np.mean([r["correct"] for r in rows])),
    }


def run_smoke(n_jev: int = 20, n_haiku: int = 20, n_opus: int = 3, run: bool = False) -> dict:
    sample = smoke_sample(max(n_jev, n_haiku, n_opus))
    questions = render_jev_questions(rotation=0, examples=EXAMPLES_VERSION)
    prompt = render_llm_prompt(examples=EXAMPLES_VERSION)
    cache, budget = InferenceCache(), Budget()
    jev = JevClient(cache=cache, budget=budget)
    haiku = ClaudeClassifier(HAIKU, prompt, cache=cache, budget=budget)
    opus = ClaudeClassifier(OPUS, prompt, cache=cache, budget=budget)

    estimate = {
        "jev-1.13.0": sum(
            jev.estimate_cost({"requirement": t}, questions) for t in sample["text"][:n_jev]
        ),
        "claude-haiku-5-5": sum(haiku.estimate_cost(t) for t in sample["text"][:n_haiku]),
        "claude-opus-5-5": sum(opus.estimate_cost(t) for t in sample["text"][:n_opus]),
    }
    report: dict = {
        "question_set": questions.version,
        "question_set_hash": questions.hash,
        "llm_prompt": prompt.version,
        "worst_case_estimate_usd": estimate,
        "budget_remaining_usd": budget.remaining(),
    }
    if not run:
        return report

    jev_rows, haiku_rows, opus_rows = [], [], []
    for _, row in sample.head(n_jev).iterrows():
        result = jev.ask({"requirement": row["text"]}, questions)
        answer = result.answers["req_type"]
        jev_rows.append(
            {**asdict(result), "answers": None, "correct": answer["choice"] == row["leaf"]}
        )
    for client, n, rows in ((haiku, n_haiku, haiku_rows), (opus, n_opus, opus_rows)):
        for _, row in sample.head(n).iterrows():
            result = client.classify(row["text"])
            rows.append({**asdict(result), "correct": result.label == row["leaf"]})

    report["models_reported"] = sorted({r["model_reported"] for r in jev_rows})
    report["summaries"] = [
        _summary("jev-1.13.0 (req_type)", jev_rows),
        _summary("claude-haiku-5-5", haiku_rows),
        _summary("claude-opus-5-5", opus_rows),
    ]
    report["stop_reasons"] = {
        "haiku": sorted({r["stop_reason"] for r in haiku_rows}),
        "opus": sorted({r["stop_reason"] for r in opus_rows}),
    }
    report["budget_remaining_usd"] = budget.remaining()
    out = repo_root() / "runs" / "smoke" / f"{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    report["written_to"] = str(out)
    return report
