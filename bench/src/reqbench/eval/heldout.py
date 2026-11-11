"""Phase 1, step 6d: the one-time evaluation on the locked test (and, later, the fresh set).

Three modes (`reqbench heldout <set>`):

- default, dry run: counts the paid calls missing from the cache and estimates their cost per
  system. It reads requirement texts only, never labels, and calls no API.
- `--run`: checks the freeze (frozen/phase1/FREEZE.json must match the files on disk and be
  committed), records the evaluation in docs/reports/phase1/EVAL_LOG.jsonl *before* any call
  or label is read, makes the paid calls, then evaluates.
- `--from-cache`: evaluates from the cache only (reproduction); fails if a call is missing.

A set already logged with one freeze hash can never be evaluated with another one: that would
be a second look at the test. Re-running with the same freeze only re-reads the cache.

Paid calls, per requirement of the set (the same plan as dev):
- Jev `jev_type_v1`: rotations 0, 1, 2 (sample 0) + samples 1-4 of rotation 0 = 7 calls. The
  rotation-0 sample-0 calls go first, one at a time (serial latency); the rest at concurrency 8.
- B5 Haiku 5.5: samples 0-4 of rotation 0 + rotations 1, 2 = 7 calls, Message Batches API;
  latency: 30 random requirements × 5 standard calls (K concurrent, as on dev).
- B6 Opus 5.5: 1 call (rotation 0, effort low), Message Batches API; latency: 30 standard calls.
- Test only: adversarial_v1 (60 variants of dev requirements) for J1/J4 (1 Jev call each) and
  B5 (5 batch calls each).

Every system is evaluated raw; systems with a frozen calibrator also as `<name>_cal`, with the
frozen review threshold. `decide` applies the preregistered GO/NO-GO rule (see its docstring).
"""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from reqcore.budget import Budget
from reqcore.cache import InferenceCache
from reqcore.clients.claude import HAIKU, OPUS, ClaudeClassifier
from reqcore.clients.jev import JevClient
from reqcore.hashing import content_hash
from reqcore.paths import data_dir, repo_root
from reqcore.pricing import AS_OF, cost_usd
from reqcore.questions import render_llm_prompt
from scipy.stats import binomtest

from reqbench.data.build import DATASET_VERSION
from reqbench.data.examples import EXAMPLES_VERSION
from reqbench.data.splits import SPLIT_VERSION
from reqbench.eval import (
    adversarial,
    bootstrap,
    calibrate_dev,
    cv,
    final_models,
    haiku_dev,
    metrics,
)
from reqbench.models import jev, llm
from reqbench.models.baselines import LABELS

SETS = ("test", "fresh")
DELTA = 0.03
ECE_MAX, FLIP_MAX, ORDER_MAX, P95_MAX_MS, USD_PER_1000_MAX = 0.05, 0.03, 0.05, 1000.0, 1.0
LATENCY_N = 30
OPUS_LATENCY_SAMPLE = 100
B4_SEEDS = calibrate_dev.B4_SEEDS
JEV_SYSTEMS = ("J1_jev_choice", "J2_jev_choice_rot_avg", "J3_jev_lr", "J4_jev_hierarchical")
# Candidates for B* (best non-Jev system). B6 Opus is a quality ceiling, not a production
# candidate (plan §6.2); exploratory variants (B5x, B4 seed ensemble) are not candidates.
BSTAR_CANDIDATES = ("B0_majority", "B1_keywords", "B2_tfidf_lr", "B3_bge_small_lr",
                    "B4_deberta_v3_small", "B5_haiku_k5")  # fmt: skip
# Systems whose cost per requirement is no higher than Jev's (local models), for GO 2(a).
NO_MORE_EXPENSIVE = ("B0_majority", "B1_keywords", "B2_tfidf_lr", "B3_bge_small_lr", *B4_SEEDS)
# Files whose content determines the results; FREEZE.json stores their hashes.
FROZEN_GLOBS = (
    "packages/reqcore/src/reqcore/**/*.py",
    "bench/src/reqbench/models/*.py", "bench/src/reqbench/data/*.py",
    "bench/src/reqbench/eval/metrics.py", "bench/src/reqbench/eval/bootstrap.py",
    "bench/src/reqbench/eval/cv.py", "bench/src/reqbench/eval/calibration.py",
    "bench/src/reqbench/eval/calibrate_dev.py", "bench/src/reqbench/eval/final_models.py",
    "bench/src/reqbench/eval/adversarial.py", "bench/src/reqbench/eval/haiku_dev.py",
    "bench/src/reqbench/eval/heldout.py",
    "questions/*.yaml", "rules/*.yaml", "uv.lock", "docs/PREREGISTRATION.md",
    "frozen/phase1/calibrators.json", "frozen/phase1/thresholds.json",
    "frozen/phase1/final_models.json",
    f"data/splits/{SPLIT_VERSION}.csv", f"data/manifests/{DATASET_VERSION}.json",
    f"data/manifests/{SPLIT_VERSION}.json",
)  # fmt: skip


def out_dir(set_name: str) -> Path:
    return repo_root() / "runs" / "phase1" / set_name


def log_path() -> Path:
    return repo_root() / "docs" / "reports" / "phase1" / "EVAL_LOG.jsonl"


def freeze_path() -> Path:
    return calibrate_dev.frozen_dir() / "FREEZE.json"


# ----- data ------------------------------------------------------------------------------


def load_texts(set_name: str) -> pd.DataFrame:
    """req_id, project_id, text of the evaluation set. Never reads labels."""
    if set_name == "test":
        splits = pd.read_csv(data_dir() / "splits" / f"{SPLIT_VERSION}.csv")
        processed = pd.read_csv(data_dir() / "processed" / f"{DATASET_VERSION}.csv",
                                usecols=["req_id", "text"])  # fmt: skip
        rows = splits[splits["split"] == "test"].merge(processed, on="req_id", how="left")
        return rows[["req_id", "project_id", "text"]].sort_values("req_id").reset_index(drop=True)
    if set_name == "fresh":
        fresh = _fresh_frame()
        rows = fresh[fresh["adjudicated"] != "discard"].rename(columns={"doc_id": "project_id"})
        return rows[["req_id", "project_id", "text"]].sort_values("req_id").reset_index(drop=True)
    raise ValueError(set_name)


def _fresh_frame() -> pd.DataFrame:
    df = pd.read_csv(data_dir() / "fresh" / "requirements.csv", dtype=str, keep_default_na=False)
    if (df["adjudicated"] == "").any():
        raise RuntimeError("fresh set not frozen: some rows have no adjudicated label "
                           "(adjudicate, commit, then evaluate)")  # fmt: skip
    return df


def load_gold(set_name: str, rows: pd.DataFrame) -> np.ndarray:
    if set_name == "test":
        processed = pd.read_csv(data_dir() / "processed" / f"{DATASET_VERSION}.csv",
                                usecols=["req_id", "leaf"]).set_index("req_id")  # fmt: skip
        return processed.loc[rows["req_id"], "leaf"].to_numpy()
    fresh = _fresh_frame().set_index("req_id")
    return fresh.loc[rows["req_id"], "adjudicated"].to_numpy()


# ----- freeze ----------------------------------------------------------------------------


def frozen_files() -> dict[str, str]:
    root = repo_root()
    files = sorted({p for g in FROZEN_GLOBS for p in root.glob(g) if p.is_file()})
    return {str(p.relative_to(root)): final_models.file_hash(p) for p in files}


def freeze_extra() -> dict:
    """Versions, prompt/question hashes, seeds and decision constants recorded in the freeze."""
    from reqbench.models.embeddings import BGE_SMALL, BGE_SMALL_REVISION
    from reqbench.models.finetune import DEBERTA, DEBERTA_REVISION, SEEDS

    haiku = llm.classifiers(llm.ROTATIONS)
    opus = ClaudeClassifier(OPUS, render_llm_prompt(llm.PROMPT, EXAMPLES_VERSION, rotation=0))
    dev_files = sorted(cv.runs_dir().glob("oof_*.csv")) + [cv.runs_dir() / "jev_answers.csv"]
    return {
        "models": {"jev": jev.PINNED_MODEL, "haiku": HAIKU.model, "opus": OPUS.model,
                   "bge_small": f"{BGE_SMALL}@{BGE_SMALL_REVISION}",
                   "deberta": f"{DEBERTA}@{DEBERTA_REVISION}"},
        "claude_configs": {"haiku": {**HAIKU.request_params(), "effort": HAIKU.effort},
                           "opus": {**OPUS.request_params(), "effort": OPUS.effort}},
        "jev_question_set": jev.QUESTION_SET,
        "jev_question_set_hashes": {str(r): q.hash for r, q in jev.question_sets().items()},
        "llm_prompt": llm.PROMPT,
        "haiku_question_set_hashes": {str(r): c.question_set_hash for r, c in haiku.items()},
        "opus_question_set_hash": opus.question_set_hash,
        "examples": EXAMPLES_VERSION, "adversarial": adversarial.VERSION,
        "seeds": {"split": 20261010, "bootstrap": bootstrap.SEED, "b4": list(SEEDS),
                  "adversarial": adversarial.spec()["seed"]},
        "k_haiku": llm.K, "n_bootstrap": bootstrap.N_RESAMPLES,
        "decision": {"delta": DELTA, "ece_max": ECE_MAX, "flip_max": FLIP_MAX,
                     "order_max": ORDER_MAX, "p95_max_ms": P95_MAX_MS,
                     "usd_per_1000_max": USD_PER_1000_MAX,
                     "target_precision": metrics.TARGET_PRECISION,
                     "bstar_candidates": list(BSTAR_CANDIDATES)},
        "dev_inputs_sha256": {p.name: final_models.file_hash(p) for p in dev_files},
    }  # fmt: skip


def write_freeze(extra: dict | None = None) -> dict:
    extra = freeze_extra() if extra is None else extra
    files = frozen_files()
    freeze = {"files": files, **extra}
    freeze["freeze_hash"] = content_hash({"files": files, **extra})
    freeze_path().write_text(json.dumps(freeze, indent=1, sort_keys=True) + "\n")
    return freeze


def check_freeze() -> dict:
    """The committed freeze, if every frozen file still matches it and is committed."""
    freeze = json.loads(freeze_path().read_text())
    now = frozen_files()
    changed = sorted(k for k in set(now) | set(freeze["files"])
                     if now.get(k) != freeze["files"].get(k))  # fmt: skip
    if changed:
        raise RuntimeError(f"frozen files changed since the freeze: {changed[:10]}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *freeze["files"], str(freeze_path())],
        capture_output=True, text=True, cwd=repo_root(), check=True).stdout.strip()  # fmt: skip
    if dirty:
        raise RuntimeError(f"frozen files are not committed:\n{dirty}")
    return freeze


def _log_entries(set_name: str) -> list[dict]:
    if not log_path().exists():
        return []
    entries = [json.loads(line) for line in log_path().read_text().splitlines() if line.strip()]
    return [e for e in entries if e["set"] == set_name]


def _append(entry: dict) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=repo_root(), check=True).stdout.strip()  # fmt: skip
    entry = {**entry, "at": datetime.now(UTC).isoformat(), "git_commit": commit}
    log_path().parent.mkdir(parents=True, exist_ok=True)
    with log_path().open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def register(set_name: str, freeze: dict, restart_reason: str | None = None) -> dict:
    """Log the evaluation before any label is read.

    - Once a set has a `completed` event, only its freeze may ever be used again (a cache
      re-read reproduces the same numbers); any other freeze is refused.
    - If an earlier attempt `started` but never completed (a crash before any result was
      written), a new freeze is accepted only with an explicit `restart_reason`, logged.
    """
    entries = _log_entries(set_name)
    done = [e for e in entries if e["event"] == "completed"]
    if done:
        if done[0]["freeze_hash"] != freeze["freeze_hash"]:
            raise RuntimeError(
                f"{set_name} was already evaluated with freeze {done[0]['freeze_hash'][:12]}…; "
                "another freeze would be a second evaluation.")  # fmt: skip
        return done[0]
    started = [e for e in entries if e["event"] in ("started", "restarted")]
    if started and started[-1]["freeze_hash"] == freeze["freeze_hash"]:
        return started[-1]
    entry = {"set": set_name, "event": "started", "freeze_hash": freeze["freeze_hash"]}
    if started:
        if not restart_reason:
            raise RuntimeError(f"an attempt on {set_name} with another freeze never completed; "
                               "pass a restart reason to log a restart")  # fmt: skip
        entry |= {"event": "restarted", "reason": restart_reason,
                  "previous_freeze_hash": started[-1]["freeze_hash"]}  # fmt: skip
    if set_name == "fresh":
        entry["fresh_file_sha256"] = final_models.file_hash(
            data_dir() / "fresh" / "requirements.csv")
    return _append(entry)


def complete(set_name: str, freeze: dict, results_file: Path) -> dict:
    if any(e["event"] == "completed" for e in _log_entries(set_name)):
        return {}
    return _append({"set": set_name, "event": "completed", "freeze_hash": freeze["freeze_hash"],
                    "results_sha256": final_models.file_hash(results_file)})  # fmt: skip


# ----- paid calls ------------------------------------------------------------------------


def _clients():
    cache, budget = InferenceCache(), Budget()
    haiku = llm.classifiers(llm.ROTATIONS, cache=cache, budget=budget)
    opus = ClaudeClassifier(OPUS, render_llm_prompt(llm.PROMPT, EXAMPLES_VERSION, rotation=0),
                            cache=cache, budget=budget)  # fmt: skip
    return JevClient(cache=cache, budget=budget), jev.question_sets(), haiku, opus, budget


def plan(set_name: str, rows: pd.DataFrame) -> dict:
    adv = adversarial.variants() if set_name == "test" else None
    lat_rows = haiku_dev.latency_rows(rows, LATENCY_N)
    p = {
        "jev": jev.plan_calls(rows),
        "haiku_batch": llm.plan_calls(rows),
        "haiku_latency": haiku_dev.latency_calls(lat_rows),
        "opus_batch": [llm.Call(r.req_id, r.text, 0, 0) for r in rows.itertuples()],
        "opus_latency": [llm.Call(r.req_id, r.text, 0, OPUS_LATENCY_SAMPLE)
                         for r in lat_rows.itertuples()],  # fmt: skip
        "adversarial": adv,
    }
    if adv is not None:
        p["jev_adv"] = [jev.Call(v.variant_id, v.text, 0, 0) for v in adv.itertuples()]
        p["haiku_adv"] = [llm.Call(v.variant_id, v.text, 0, s) for v in adv.itertuples()
                          for s in range(llm.K)]  # fmt: skip
    else:
        p["jev_adv"], p["haiku_adv"] = [], []
    return p


def _dev_tokens(name: str, default: tuple[float, float]) -> tuple[float, float]:
    path = cv.runs_dir() / name
    if not path.exists():
        return default
    usage = json.loads(path.read_text())["usage"]
    return float(usage["mean_input_tokens"]), float(usage.get("mean_output_tokens", 0.0))


def estimate(set_name: str) -> dict:
    rows = load_texts(set_name)
    p = plan(set_name, rows)
    client, qsets, haiku, opus, budget = _clients()
    jev_tokens = _dev_tokens("jev_report.json", (2132.0, 0.0))[0]
    jev_est = jev.estimate(p["jev"] + p["jev_adv"], client, qsets, tokens_per_call=jev_tokens)
    haiku_est = haiku_dev.estimate_llm(p["haiku_batch"] + p["haiku_adv"], p["haiku_latency"],
                                       haiku, budget, HAIKU.model,
                                       _dev_tokens("haiku_report.json", haiku_dev.SMOKE_TOKENS))
    opus_est = haiku_dev.estimate_llm(p["opus_batch"], p["opus_latency"], {0: opus}, budget,
                                      OPUS.model, (1214.0, 10.0))  # smoke test means  # fmt: skip
    # Opus may think before answering (it cannot be disabled): a scenario with 300 output tokens.
    n_b, n_l = opus_est["batch_calls_to_make"], opus_est["latency_calls_to_make"]
    tin = opus_est["tokens_per_call"]["input"]
    opus_est["if_300_output_tokens_usd"] = (n_b * cost_usd(OPUS.model, round(tin), 300, True)
                                            + n_l * cost_usd(OPUS.model, round(tin), 300))
    out = {
        "set": set_name, "requirements": len(rows),
        "adversarial_variants": 0 if p["adversarial"] is None else len(p["adversarial"]),
        "jev": jev_est, "haiku_B5": haiku_est, "opus_B6": opus_est,
        "expected_total_usd": (jev_est.get("expected_usd", 0.0)
                               + haiku_est["expected_usd"]["total"]
                               + opus_est["expected_usd"]["total"]),
        "budget_remaining_usd": budget.remaining(), "prices_as_of": AS_OF,
    }  # fmt: skip
    return out


def make_calls(set_name: str, rows: pd.DataFrame, log=print) -> None:
    p = plan(set_name, rows)
    client, qsets, haiku, opus, _ = _clients()
    submitted = []
    for rotation, items in llm.batch_items(p["haiku_batch"] + p["haiku_adv"]).items():
        if (batch_id := haiku[rotation].submit_batch(items)) is not None:
            submitted.append(batch_id)
            log(f"B5 rotation {rotation}: batch {batch_id}")
    if (batch_id := opus.submit_batch(llm.batch_items(p["opus_batch"])[0])) is not None:
        log(f"B6: batch {batch_id}")

    serial = [c for c in p["jev"] if c.rotation == 0 and c.sample_index == 0]
    log(f"Jev: {len(serial)} serial calls")
    jev.collect(serial, client, qsets, run=True, workers=1, progress=False)
    rest = [c for c in p["jev"] if c.rotation or c.sample_index] + p["jev_adv"]
    log(f"Jev: {len(rest)} calls at concurrency 8")
    jev.collect(rest, client, qsets, run=True, workers=8)

    log("B5 latency (standard calls)")
    haiku_dev._measure_latency(p["haiku_latency"], haiku[0], log)
    log("B6 latency (standard calls, serial)")
    for c in p["opus_latency"]:
        opus.classify(c.text, c.sample_index)
    for clf in [*haiku.values(), opus]:
        for state in clf.pending_batches():
            log(f"batch {state['batch_id']}: {clf.collect_batch(state['batch_id'])}")


# ----- predictions -----------------------------------------------------------------------


def _llm_answers(calls, clfs: dict, model: str) -> pd.DataFrame:
    rows, missing = [], 0
    for c in calls:
        result = clfs[c.rotation].cached(c.text, c.sample_index)
        if result is None:
            missing += 1
        else:
            rows.append(llm.to_row(c, clfs[c.rotation], result))
    if missing:
        raise RuntimeError(f"{model}: {missing} calls missing from the cache")
    frame = pd.DataFrame(rows)
    if set(frame["model"]) != {model}:
        raise RuntimeError(f"unexpected models: {sorted(set(frame['model']))}")
    return frame


def _jev_answers(calls, client, qsets) -> pd.DataFrame:
    frame = jev.collect(calls, client, qsets, run=False)
    if len(frame) != len(calls):
        raise RuntimeError(f"Jev: {len(calls) - len(frame)} calls missing from the cache")
    return frame


def _frame(rows: pd.DataFrame, gold, pred, probs: np.ndarray | None) -> pd.DataFrame:
    out = pd.DataFrame({"req_id": rows["req_id"].to_numpy(),
                        "project_id": rows["project_id"].to_numpy(), "fold": -1,
                        "gold": gold, "is_example": False, "pred": pred})  # fmt: skip
    if probs is not None:
        for k, label in enumerate(LABELS):
            out[f"p_{label}"] = probs[:, k]
    return out


def _argmax(probs: np.ndarray) -> np.ndarray:
    return np.array([LABELS[i] for i in probs.argmax(axis=1)])


def predict_all(set_name: str, rows: pd.DataFrame, gold: np.ndarray, log=print) -> dict:
    """name -> (frame, extras). Raw systems, `_cal` variants and the exploratory ones."""
    p = plan(set_name, rows)
    client, qsets, haiku, opus, _ = _clients()
    answers = _jev_answers(p["jev"], client, qsets)
    texts = rows["text"].tolist()
    systems: dict[str, pd.DataFrame] = {}
    raw: dict[str, dict] = {}

    for name in ("B0_majority", "B1_keywords"):
        systems[name] = _frame(rows, gold, final_models.load(name).predict(texts), None)
    for name in ("B2_tfidf_lr", "B3_bge_small_lr", *B4_SEEDS):
        log(f"predicting {name}")
        probs = final_models.load(name).predict_proba(texts)
        systems[name] = _frame(rows, gold, _argmax(probs), probs)
        raw[name] = {"system": probs}

    flat = {"J1_jev_choice": jev.flat_probs(answers, (0,)),
            "J2_jev_choice_rot_avg": jev.flat_probs(answers, jev.ROTATIONS),
            "J4_jev_hierarchical": jev.hierarchical_probs(answers, (0,)),
            "J4x_jev_hierarchical_rot_avg": jev.hierarchical_probs(answers, jev.ROTATIONS)}
    for name, table in flat.items():
        probs = table.loc[rows["req_id"], list(LABELS)].to_numpy()
        systems[name] = _frame(rows, gold, _argmax(probs), probs)
        raw[name] = {"system": probs}
    r0 = answers[(answers["rotation"] == 0) & (answers["sample_index"] == 0)]
    r0 = r0.set_index("req_id").loc[rows["req_id"]]
    raw["J4_jev_hierarchical"] = {
        "is_functional": r0["nf_is_functional"].to_numpy(),
        "nfr_subtype": jev._renorm(r0[[f"ns_{k}" for k in jev.NFR]].to_numpy())}
    j3 = final_models.load("J3_jev_lr")
    table = jev.features(answers, jev.ROTATIONS)
    table.index = rows.set_index("req_id").loc[table.index, "text"]
    j3.table = table
    probs = j3.predict_proba(texts)
    systems["J3_jev_lr"] = _frame(rows, gold, _argmax(probs), probs)
    raw["J3_jev_lr"] = {"system": probs}

    haiku_answers = _llm_answers(p["haiku_batch"], haiku, HAIKU.model)
    freq = llm.frequency_probs(haiku_answers, 0, tuple(range(llm.K))).loc[rows["req_id"]]
    probs = freq[list(LABELS)].to_numpy(dtype=float)
    systems["B5_haiku_k5"] = _frame(rows, gold, freq["pred"].to_numpy(), probs)
    raw["B5_haiku_k5"] = {"system": probs}
    single = llm.frequency_probs(haiku_answers, 0, (0,)).loc[rows["req_id"]]
    systems["B5x_haiku_single"] = _frame(rows, gold, single["pred"].to_numpy(), None)
    opus_answers = _llm_answers(p["opus_batch"], {0: opus}, OPUS.model)
    opus_pred = opus_answers.set_index("req_id").loc[rows["req_id"], "label"]
    systems["B6_opus"] = _frame(rows, gold, opus_pred.fillna("invalid").to_numpy(), None)

    ens = np.mean([systems[s][[f"p_{k}" for k in LABELS]].to_numpy() for s in B4_SEEDS], axis=0)
    systems["B4x_deberta_seed_ensemble"] = _frame(rows, gold, _argmax(ens), ens)

    spec = json.loads((calibrate_dev.frozen_dir() / "calibrators.json").read_text())["systems"]
    for name in spec:
        probs = calibrate_dev.apply(spec, name, raw[name])
        systems[f"{name}_cal"] = _frame(rows, gold, _argmax(probs), probs)

    extras = {"jev_answers": answers, "haiku_answers": haiku_answers,
              "opus_answers": opus_answers}  # fmt: skip
    if p["adversarial"] is not None:
        extras["jev_adv"] = _jev_answers(p["jev_adv"], client, qsets)
        extras["haiku_adv"] = _llm_answers(p["haiku_adv"], haiku, HAIKU.model)
    extras["haiku_latency"] = _llm_answers(p["haiku_latency"], haiku, HAIKU.model)
    extras["opus_latency"] = _llm_answers(p["opus_latency"], {0: opus}, OPUS.model)
    return {"systems": systems, **extras}


# ----- measurements ----------------------------------------------------------------------


def local_latency(rows: pd.DataFrame) -> dict:
    """CPU, batch 1, one requirement at a time (plan §6.3); B3 embeds without its cache."""
    texts = rows["text"].tolist()
    out = {}
    for name in ("B2_tfidf_lr", "B3_bge_small_lr", B4_SEEDS[0]):
        model = final_models.load(name)
        if name == "B3_bge_small_lr":
            embedder = model.embed
            model.embed = lambda t, e=embedder: e._encode(list(t))
        model.predict_proba(texts[:3])  # warm-up
        times = []
        for text in texts:
            started = time.perf_counter()
            model.predict_proba([text])
            times.append((time.perf_counter() - started) * 1000)
        out[name] = {"ms_p50": float(np.percentile(times, 50)),
                     "ms_p95": float(np.percentile(times, 95)), "n": len(times),
                     "device": "cpu", "batch": 1}  # fmt: skip
    return out


def jev_latency(answers: pd.DataFrame, n_rows: int) -> dict:
    r0s0 = answers[(answers["rotation"] == 0) & (answers["sample_index"] == 0)]
    conc = answers[(answers["rotation"] == 0) & (answers["sample_index"] > 0)]
    s0 = answers[answers["sample_index"] == 0]
    per_req_3 = s0.groupby("req_id")["latency_ms"].max()  # J2/J3: 3 rotations, concurrent
    return {
        "serial_ms_p50": float(r0s0["latency_ms"].quantile(0.5)),
        "serial_ms_p95": float(r0s0["latency_ms"].quantile(0.95)), "serial_n": len(r0s0),
        "concurrency8_ms_p50": float(conc["latency_ms"].quantile(0.5)),
        "concurrency8_ms_p95": float(conc["latency_ms"].quantile(0.95)),
        "concurrency8_n": len(conc),
        "three_rotations_per_requirement_ms_p95": float(per_req_3.quantile(0.95)),
        "note": "stored latency of the original (non-cached) calls; the serial pass ran first",
    }  # fmt: skip


def llm_usage(answers: pd.DataFrame, latency: pd.DataFrame, model: str, k: int) -> dict:
    tin, tout = float(answers["input_tokens"].mean()), float(answers["output_tokens"].mean())
    per_req = latency.groupby("req_id")["latency_ms"].max()
    return {
        "calls": len(answers), "mean_input_tokens": tin, "mean_output_tokens": tout,
        "invalid_labels": int(answers["label"].isna().sum()),
        "stop_reasons": answers["stop_reason"].value_counts().to_dict(),
        "spent_usd_recorded": float(answers["cost_usd"].sum() + latency["cost_usd"].sum()),
        "usd_per_1000_standard": 1000 * k * cost_usd(model, round(tin), round(tout), False),
        "usd_per_1000_batch": 1000 * k * cost_usd(model, round(tin), round(tout), True),
        "latency_per_call_ms_p50": float(latency["latency_ms"].quantile(0.5)),
        "latency_per_call_ms_p95": float(latency["latency_ms"].quantile(0.95)),
        "latency_per_requirement_ms_p95": float(per_req.quantile(0.95)),
        "prices_as_of": AS_OF,
    }  # fmt: skip


def adversarial_results(adv_variants, extras) -> dict:
    dev_j1 = pd.read_csv(cv.runs_dir() / "oof_J1_jev_choice.csv").set_index("req_id")["pred"]
    dev_j4 = pd.read_csv(cv.runs_dir() / "oof_J4_jev_hierarchical.csv").set_index("req_id")["pred"]
    dev_b5 = pd.read_csv(cv.runs_dir() / "oof_B5_haiku_k5.csv").set_index("req_id")["pred"]
    ja = extras["jev_adv"]
    j1 = pd.Series(_argmax(jev.flat_probs(ja, (0,)).to_numpy()),
                   index=jev.flat_probs(ja, (0,)).index)  # fmt: skip
    h = jev.hierarchical_probs(ja, (0,))
    j4 = pd.Series(_argmax(h.to_numpy()), index=h.index)
    b5 = llm.frequency_probs(extras["haiku_adv"], 0, tuple(range(llm.K)))["pred"]
    return {"J1_jev_choice": adversarial.flip_rates(adv_variants, dev_j1, j1),
            "J4_jev_hierarchical": adversarial.flip_rates(adv_variants, dev_j4, j4),
            "B5_haiku_k5": adversarial.flip_rates(adv_variants, dev_b5, b5)}  # fmt: skip


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    ra, rb = a["gold"] == a["pred"], b["gold"] == b["pred"]
    n01, n10 = int((ra & ~rb).sum()), int((~ra & rb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_exact": float(p)}


def project_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_resamples: int) -> dict:
    """Sensitivity: macro-F1(a) - macro-F1(b) resampling whole projects."""
    projects = a["project_id"].unique()
    rng = np.random.default_rng(bootstrap.SEED)
    by = {p: np.flatnonzero(a["project_id"].to_numpy() == p) for p in projects}
    g, pa, pb = a["gold"].to_numpy(), a["pred"].to_numpy(), b["pred"].to_numpy()
    diffs = []
    for _ in range(n_resamples):
        idx = np.concatenate([by[p] for p in rng.choice(projects, len(projects))])
        diffs.append(metrics.macro_f1(g[idx], pa[idx], LABELS)
                     - metrics.macro_f1(g[idx], pb[idx], LABELS))  # fmt: skip
    return {"n_projects": len(projects), "value": metrics.macro_f1(g, pa, LABELS)
            - metrics.macro_f1(g, pb, LABELS),
            "ci95": [float(x) for x in np.percentile(diffs, [2.5, 97.5])]}  # fmt: skip


# ----- decision --------------------------------------------------------------------------


def decide(r: dict) -> dict:
    """Preregistered GO / NO-GO / inconclusive rule (docs/PREREGISTRATION.md), as frozen:

    - best Jev = highest test macro-F1 among J1-J4 (raw, the preregistered systems);
    - B* = highest test macro-F1 among BSTAR_CANDIDATES; B4 counts with the mean of its 5
      seeds, and its paired comparisons use the seed with the highest test macro-F1 (the
      least favourable to Jev);
    - 1: lower CI bound of macro-F1(best Jev) - macro-F1(B*) > -δ;
    - 2a: against every system costing no more (local B0-B4, every seed): difference ≥ δ and
      lower CI bound > 0; 2b: AURC difference (raw probabilities) with upper CI bound < 0;
      2c: needs the fresh set (or the external set): "pending" until it is evaluated;
    - 3: ECE of best Jev with its frozen calibrator ≤ 0.05 (point estimate); run-to-run flip
      rate ≤ 3 % (J1/J2/J3: `req_type` labels, J4: its own); option-order flip rate ≤ 5 %, or
      best Jev is J2 (which averages the orders); p95 latency ≤ 1 s (J1/J4 serial per call,
      J2/J3 slowest of 3 calls); cost ≤ $1 per 1,000; adversarial flip rate ≤ B5's;
    - NO-GO if: the whole CI of the difference vs B* is below -δ; or best Jev < B2 with
      neither 2b nor 2c; or criterion 3's stability/order thresholds fail;
    - otherwise, if GO is not met: inconclusive, and the simpler, cheaper system wins.
    """
    s, comp = r["summaries"], r["comparisons"]
    best = max(JEV_SYSTEMS, key=lambda n: s[n]["macro_f1"])

    def f1(name: str) -> float:
        if name == "B4_deberta_v3_small":
            return float(np.mean([s[k]["macro_f1"] for k in B4_SEEDS]))
        return s[name]["macro_f1"]

    bstar = max(BSTAR_CANDIDATES, key=f1)
    bstar_ref = max(B4_SEEDS, key=lambda k: s[k]["macro_f1"]) if bstar.startswith("B4") else bstar
    d = comp[best][bstar_ref]
    c1 = d["macro_f1"]["ci95"][0] > -DELTA
    c2a = all(comp[best][o]["macro_f1"]["value"] >= DELTA
              and comp[best][o]["macro_f1"]["ci95"][0] > 0 for o in NO_MORE_EXPENSIVE)
    c2b = "aurc" in d and d["aurc"]["ci95"][1] < 0
    c2c = r.get("fresh_criterion_2c", "pending")
    rob = r["robustness"]
    flip = rob["jev_stability"]["J4" if best.startswith("J4") else "J1"]["flip_rate_vs_sample0"]
    order = rob["jev_order"]["J4" if best.startswith("J4") else "req_type"]["flip_rate_pairwise"]
    lat = r["latency"]["jev"]
    p95 = lat["three_rotations_per_requirement_ms_p95"] if best in (
        "J2_jev_choice_rot_avg", "J3_jev_lr") else lat["serial_ms_p95"]  # fmt: skip
    usd = r["cost"]["jev_usd_per_1000"]["J2" if best in ("J2_jev_choice_rot_avg", "J3_jev_lr")
                                        else "J1"]  # fmt: skip
    adv = r["adversarial"]
    adv_name = "J4_jev_hierarchical" if best.startswith("J4") else "J1_jev_choice"
    c3 = {
        "ece_calibrated": s[f"{best}_cal"]["ece"] <= ECE_MAX,
        "flip_rate": flip <= FLIP_MAX,
        "order": order <= ORDER_MAX or best == "J2_jev_choice_rot_avg",
        "p95_latency": p95 <= P95_MAX_MS,
        "cost": usd <= USD_PER_1000_MAX,
        "adversarial": (adv[adv_name]["flip_rate"] <= adv["B5_haiku_k5"]["flip_rate"]
                        if adv else "not measured"),
    }  # fmt: skip
    c3_ok = all(v is True for v in c3.values())
    c2 = c2a or c2b or c2c is True
    nogo = {
        "below_bstar": d["macro_f1"]["ci95"][1] < -DELTA,
        "below_b2_without_b_or_c": (s[best]["macro_f1"] < s["B2_tfidf_lr"]["macro_f1"]
                                    and not c2b and c2c is not True),
        "instability_or_order": not (c3["flip_rate"] and c3["order"]),
    }  # fmt: skip
    if any(nogo.values()):
        verdict = "NO-GO"
    elif c1 and c2 and c3_ok:
        verdict = "GO"
    elif c1 and c3_ok and not c2 and c2c == "pending":
        verdict = "INCONCLUSIVE (pending 2c: fresh set)"
    else:
        verdict = "INCONCLUSIVE"
    half = (d["macro_f1"]["ci95"][1] - d["macro_f1"]["ci95"][0]) / 2
    margin = d["macro_f1"]["value"] + DELTA
    needed = (int(np.ceil(r["n"] * (half / margin) ** 2)) if margin > 0 and not c1 else None)
    return {
        "best_jev": best, "bstar": bstar, "bstar_comparison_reference": bstar_ref,
        "diff_macro_f1": d["macro_f1"], "criterion_1": c1,
        "criterion_2": {"a": c2a, "b": c2b, "c": c2c}, "criterion_3": c3,
        "nogo_conditions": nogo, "verdict": verdict,
        "n_needed_for_non_inferiority_approx": needed,
        "delta": DELTA,
    }  # fmt: skip


# ----- evaluation ------------------------------------------------------------------------


def evaluate(set_name: str, rows: pd.DataFrame, n_resamples: int = bootstrap.N_RESAMPLES,
             log=print) -> dict:  # fmt: skip
    gold = load_gold(set_name, rows)
    pred = predict_all(set_name, rows, gold, log)
    systems = pred["systems"]
    out = out_dir(set_name)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = json.loads((calibrate_dev.frozen_dir() / "thresholds.json").read_text())

    summaries = {}
    for name, frame in systems.items():
        log(f"evaluating {name}")
        frame.to_csv(out / f"pred_{name}.csv", index=False)
        summaries[name] = cv.evaluate(frame, n_resamples=n_resamples)
        if name.endswith("_cal"):
            probs = cv.probabilities(frame)
            summaries[name]["review"] = metrics.selective(
                frame["gold"].to_numpy(), frame["pred"].to_numpy(), probs,
                thresholds[name[:-4]]["threshold"])  # fmt: skip
    # Paired comparisons of every Jev variant against every other system.
    others = [n for n in systems if not n.startswith("J")]
    comparisons = {j: {o: cv.compare(systems[j], systems[o], n_resamples) for o in others}
                   for j in JEV_SYSTEMS}  # fmt: skip
    for j in JEV_SYSTEMS:
        adjusted = bootstrap.holm({o: comparisons[j][o]["macro_f1"]["p_two_sided"]
                                   for o in others})  # fmt: skip
        for o, p in adjusted.items():
            comparisons[j][o]["macro_f1"]["p_holm"] = p
            comparisons[j][o]["mcnemar"] = mcnemar(systems[j], systems[o])

    answers = pred["jev_answers"]
    jev_tokens = float(answers["input_tokens"].mean())
    per_call = cost_usd(jev.PINNED_MODEL, round(jev_tokens), 0)
    results = {
        "set": set_name, "n": len(rows), "labels": list(LABELS),
        "summaries": summaries, "comparisons": comparisons,
        "confusion": {n: metrics.confusion(f["gold"], f["pred"], LABELS).tolist()
                      for n, f in systems.items()},
        "robustness": {"jev_stability": jev.stability(answers),
                       "jev_order": jev.order_sensitivity(answers),
                       "haiku_stability": llm.stability(pred["haiku_answers"]),
                       "haiku_order": llm.order_sensitivity(pred["haiku_answers"])},
        "latency": {"jev": jev_latency(answers, len(rows)), "local": local_latency(rows)},
        "cost": {"jev_mean_input_tokens": jev_tokens,
                 "jev_usd_per_1000": {"J1": 1000 * per_call, "J4": 1000 * per_call,
                                      "J2": 3000 * per_call, "J3": 3000 * per_call},
                 "B5": llm_usage(pred["haiku_answers"][pred["haiku_answers"]["rotation"] == 0],
                                 pred["haiku_latency"], HAIKU.model, llm.K),
                 "B6": llm_usage(pred["opus_answers"], pred["opus_latency"], OPUS.model, 1),
                 "prices_as_of": AS_OF},
        "models_reported": {"jev": sorted(answers["model"].unique()),
                            "haiku": sorted(pred["haiku_answers"]["model"].unique()),
                            "opus": sorted(pred["opus_answers"]["model"].unique())},
        "adversarial": (adversarial_results(adversarial.variants(), pred)
                        if "jev_adv" in pred else None),
    }  # fmt: skip
    results["latency"]["B5"] = {k: v for k, v in results["cost"]["B5"].items() if "latency" in k}
    results["latency"]["B6"] = {k: v for k, v in results["cost"]["B6"].items() if "latency" in k}
    best = max(JEV_SYSTEMS, key=lambda n: summaries[n]["macro_f1"])
    bstar = max(BSTAR_CANDIDATES, key=lambda n: summaries.get(n, {"macro_f1": np.mean(
        [summaries[k]["macro_f1"] for k in B4_SEEDS])})["macro_f1"])  # fmt: skip
    ref = (max(B4_SEEDS, key=lambda k: summaries[k]["macro_f1"]) if bstar.startswith("B4")
           else bstar)
    results["project_bootstrap_best_jev_vs_bstar"] = project_bootstrap(
        systems[best], systems[ref], n_resamples)
    if set_name == "test":
        results["decision"] = decide(results)
    pred["jev_answers"].to_csv(out / "jev_answers.csv", index=False)
    pred["haiku_answers"].to_csv(out / "haiku_answers.csv", index=False)
    pred["opus_answers"].to_csv(out / "opus_answers.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=_json_default) + "\n")
    return results


def run(set_name: str, mode: str = "dry", n_resamples: int = bootstrap.N_RESAMPLES,
        restart_reason: str | None = None, log=print) -> dict:  # fmt: skip
    if mode == "dry":
        return {"dry_run": estimate(set_name)}
    freeze = check_freeze()
    if mode == "cache" and not _log_entries(set_name):
        raise RuntimeError(f"{set_name} has never been evaluated: use --run (after the freeze)")
    entry = register(set_name, freeze, restart_reason)
    rows = load_texts(set_name)
    if mode == "run":
        make_calls(set_name, rows, log)
    results = evaluate(set_name, rows, n_resamples, log)
    complete(set_name, freeze, out_dir(set_name) / "results.json")
    results["eval_log_entry"] = entry
    return results


def _json_default(value):
    if isinstance(value, np.integer | np.floating | np.bool_):
        return value.item()
    raise TypeError(type(value))
