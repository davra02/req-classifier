"""Zero-shot Claude classifier with structured output, inference cache and budget guard.

Two ways to fill the cache, with the same request body and the same cache key:

- `classify`: one standard Messages call; its latency is measured and stored.
- `submit_batch` / `collect_batch`: the Message Batches API (50 % cheaper, no latency). The
  batch id and its items are written to `cache/batches/` before polling, so an interrupted run
  resumes the same batch instead of paying for it twice.

The requirement goes only in the user turn (`LlmPrompt.user`), never in the system prompt, and
the answer is constrained to the label enum by `output_config.format`.
"""

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic

from reqcore.budget import Budget
from reqcore.cache import CachedInference, InferenceCache, cache_key
from reqcore.hashing import content_hash
from reqcore.paths import repo_root
from reqcore.pricing import cost_usd
from reqcore.questions import LlmPrompt

PROVIDER = "anthropic"
_CHARS_PER_TOKEN = 3.0


class ModelMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaudeConfig:
    model: str
    effort: str = "low"
    # Haiku 5.5 accepts disabled thinking at effort <= high; Opus 5.5 rejects it.
    disable_thinking: bool = False
    max_tokens: int = 1024

    def request_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens}
        if self.disable_thinking:
            params["thinking"] = {"type": "disabled"}
        return params


HAIKU = ClaudeConfig("claude-haiku-5-5", effort="low", disable_thinking=True, max_tokens=256)
# Opus 5.5 cannot disable thinking; thinking tokens count against max_tokens. At effort low the
# smoke test used ~10 output tokens; 1024 keeps the budget guard's worst case (full max_tokens
# on every call) for the 171-call test batch under the $3 Opus bucket. A truncated answer is
# recorded as an invalid label (stop_reason max_tokens), never retried silently.
OPUS = ClaudeConfig("claude-opus-5-5", effort="low", disable_thinking=False, max_tokens=1024)


@dataclass(frozen=True)
class ClaudeResult:
    key: str
    label: str | None
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    latency_ms: float | None
    cached: bool
    model_reported: str | None = None
    batch: bool = False


@dataclass(frozen=True)
class BatchItem:
    requirement: str
    sample_index: int


def batches_dir() -> Path:
    return repo_root() / "cache" / "batches"


class ClaudeClassifier:
    def __init__(
        self,
        config: ClaudeConfig,
        prompt: LlmPrompt,
        cache: InferenceCache | None = None,
        budget: Budget | None = None,
        client: anthropic.Anthropic | None = None,
        state_dir: Path | None = None,
    ):
        self.config = config
        self.prompt = prompt
        self.cache = cache or InferenceCache()
        self.budget = budget or Budget()
        self._client = client
        self.state_dir = state_dir or batches_dir()
        self.question_set_hash = content_hash(
            {"prompt": prompt.hash, "effort": config.effort, **config.request_params()}
        )

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:  # created lazily: dry runs need no API key
            self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        return self._client

    # ----- requests and cache -------------------------------------------------------------

    def request(self, requirement: str) -> dict[str, Any]:
        return {
            **self.config.request_params(),
            "system": self.prompt.system,
            "messages": [{"role": "user", "content": self.prompt.user(requirement)}],
            "output_config": {
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": self.prompt.output_schema()},
            },
        }

    def key(self, requirement: str, sample_index: int = 0) -> str:
        state = {"requirement": requirement}
        return cache_key(PROVIDER, self.config.model, self.question_set_hash, state, sample_index)

    def estimate_cost(self, requirement: str, batch: bool = False) -> float:
        """Worst case: estimated input plus the full `max_tokens` of output."""
        chars = len(self.prompt.system) + len(self.prompt.user(requirement))
        return cost_usd(
            self.config.model, int(chars / _CHARS_PER_TOKEN), self.config.max_tokens, batch
        )

    def cached(self, requirement: str, sample_index: int = 0) -> ClaudeResult | None:
        hit = self.cache.get(self.key(requirement, sample_index))
        return None if hit is None else self._from_cache(hit)

    def _from_cache(self, hit: CachedInference) -> ClaudeResult:
        return ClaudeResult(
            hit.key, hit.response.get("_label"), hit.response.get("stop_reason"),
            hit.input_tokens, hit.output_tokens, hit.cost_usd, hit.latency_ms, cached=True,
            model_reported=hit.model_reported, batch=bool(hit.response.get("_batch", False)),
        )  # fmt: skip

    def parse_label(self, message: Any) -> str | None:
        """The enum label, or None (refusal, truncation, or a value outside the enum).

        The structured-outputs docs warn that enum values may come back with different
        capitalization, so the comparison is case-insensitive.
        """
        if message.stop_reason != "end_turn":
            return None
        text = next((b.text for b in message.content if b.type == "text"), None)
        if text is None:
            return None
        try:
            value = json.loads(text).get("label")
        except (json.JSONDecodeError, AttributeError):
            return None
        if not isinstance(value, str):
            return None
        return next((lab for lab in self.prompt.labels if lab.lower() == value.lower()), None)

    def _store(self, key: str, sample_index: int, request: dict[str, Any], message: Any,
               latency_ms: float | None, batch: bool) -> ClaudeResult:  # fmt: skip
        usage = message.usage
        cost = cost_usd(self.config.model, usage.input_tokens, usage.output_tokens, batch)
        self.budget.record(self.config.model, cost, key)
        if message.model != self.config.model:  # billed, but never cached or used
            raise ModelMismatchError(f"requested {self.config.model}, got {message.model!r}")
        label = self.parse_label(message)
        body = message.model_dump(mode="json")
        body["_label"], body["_batch"] = label, batch
        self.cache.put(
            key=key, provider=PROVIDER, model_requested=self.config.model,
            model_reported=message.model, question_set_hash=self.question_set_hash,
            sample_index=sample_index, request=request, response=body,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cost_usd=cost, latency_ms=latency_ms, batch=batch,
        )  # fmt: skip
        return ClaudeResult(
            key, label, message.stop_reason, usage.input_tokens, usage.output_tokens, cost,
            latency_ms, cached=False, model_reported=message.model, batch=batch,
        )  # fmt: skip

    # ----- standard calls -----------------------------------------------------------------

    def classify(self, requirement: str, sample_index: int = 0) -> ClaudeResult:
        if (hit := self.cached(requirement, sample_index)) is not None:
            return hit
        self.budget.ensure_affordable(self.config.model, self.estimate_cost(requirement))
        request = self.request(requirement)
        started = time.perf_counter()
        message = self.client.messages.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._store(
            self.key(requirement, sample_index), sample_index, request, message, latency_ms,
            batch=False,
        )  # fmt: skip

    # ----- Message Batches API ------------------------------------------------------------

    def pending_batches(self) -> list[dict[str, Any]]:
        """Batches submitted for this prompt+config whose results are not ingested yet."""
        if not self.state_dir.exists():
            return []
        states = (json.loads(p.read_text()) for p in sorted(self.state_dir.glob("*.json")))
        mine = (s for s in states if s["question_set_hash"] == self.question_set_hash)
        return [s for s in mine if not s["ingested"]]

    def submit_batch(self, items: Iterable[BatchItem]) -> str | None:
        """Submit the items that are neither cached nor in a pending batch; return its id."""
        pending = {cid for s in self.pending_batches() for cid in s["items"]}
        todo: dict[str, BatchItem] = {}
        for item in items:
            key = self.key(item.requirement, item.sample_index)
            if key not in pending and key not in todo and self.cache.get(key) is None:
                todo[key] = item  # sha256 hex: a valid custom_id (^[a-zA-Z0-9_-]{1,64}$)
        if not todo:
            return None
        worst = sum(self.estimate_cost(i.requirement, batch=True) for i in todo.values())
        self.budget.ensure_affordable(self.config.model, worst)
        batch = self.client.messages.batches.create(requests=[
            {"custom_id": key, "params": self.request(item.requirement)}
            for key, item in todo.items()
        ])  # fmt: skip
        self.state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "batch_id": batch.id, "model": self.config.model, "prompt": self.prompt.version,
            "question_set_hash": self.question_set_hash, "worst_case_usd": worst,
            "ingested": False,
            "items": {k: {"requirement": i.requirement, "sample_index": i.sample_index}
                      for k, i in todo.items()},
        }  # fmt: skip
        (self.state_dir / f"{batch.id}.json").write_text(json.dumps(state, indent=1) + "\n")
        return batch.id

    def collect_batch(self, batch_id: str, poll_seconds: float = 30.0,
                      log=print) -> dict[str, int]:  # fmt: skip
        """Wait for a batch to end, then cache every succeeded result.

        Errored, canceled and expired requests are not cached, so a re-run resubmits them.
        """
        path = self.state_dir / f"{batch_id}.json"
        state = json.loads(path.read_text())
        while True:
            batch = self.client.messages.batches.retrieve(batch_id)
            if batch.processing_status == "ended":
                break
            counts = batch.request_counts
            log(f"  batch {batch_id}: {batch.processing_status}, "
                f"{counts.processing} processing, {counts.succeeded} succeeded")  # fmt: skip
            time.sleep(poll_seconds)
        outcome = {"succeeded": 0, "errored": 0, "canceled": 0, "expired": 0, "unknown_id": 0}
        for entry in self.client.messages.batches.results(batch_id):
            item = state["items"].get(entry.custom_id)
            if item is None:
                outcome["unknown_id"] += 1
                continue
            outcome[entry.result.type] += 1
            if entry.result.type == "succeeded":
                self._store(
                    entry.custom_id, item["sample_index"], self.request(item["requirement"]),
                    entry.result.message, latency_ms=None, batch=True,
                )  # fmt: skip
        state["ingested"], state["outcome"] = True, outcome
        path.write_text(json.dumps(state, indent=1) + "\n")
        return outcome
