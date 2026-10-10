"""Zero-shot Claude classifier with structured output, inference cache and budget guard."""

import json
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from reqcore.budget import Budget
from reqcore.cache import InferenceCache, cache_key
from reqcore.hashing import content_hash
from reqcore.pricing import cost_usd
from reqcore.questions import LlmPrompt

PROVIDER = "anthropic"
_CHARS_PER_TOKEN = 3.0


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
OPUS = ClaudeConfig("claude-opus-5-5", effort="low", disable_thinking=False, max_tokens=4096)


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


class ClaudeClassifier:
    def __init__(
        self,
        config: ClaudeConfig,
        prompt: LlmPrompt,
        cache: InferenceCache | None = None,
        budget: Budget | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        self.config = config
        self.prompt = prompt
        self.cache = cache or InferenceCache()
        self.budget = budget or Budget()
        # Anthropic() reads ANTHROPIC_API_KEY from the environment.
        self._client = client or anthropic.Anthropic()
        self.question_set_hash = content_hash(
            {"prompt": prompt.hash, "effort": config.effort, **config.request_params()}
        )

    def _request(self, requirement: str) -> dict[str, Any]:
        return {
            **self.config.request_params(),
            "system": self.prompt.system,
            "messages": [{"role": "user", "content": self.prompt.user(requirement)}],
            "output_config": {
                "effort": self.config.effort,
                "format": {"type": "json_schema", "schema": self.prompt.output_schema()},
            },
        }

    def estimate_cost(self, requirement: str) -> float:
        """Worst case: estimated input plus the full `max_tokens` of output."""
        chars = len(self.prompt.system) + len(self.prompt.user(requirement))
        return cost_usd(self.config.model, int(chars / _CHARS_PER_TOKEN), self.config.max_tokens)

    def classify(self, requirement: str, sample_index: int = 0) -> ClaudeResult:
        state = {"requirement": requirement}
        key = cache_key(
            PROVIDER, self.config.model, self.question_set_hash, state, sample_index
        )
        if (hit := self.cache.get(key)) is not None:
            return ClaudeResult(
                key, hit.response.get("_label"), hit.response.get("stop_reason"),
                hit.input_tokens, hit.output_tokens, hit.cost_usd, hit.latency_ms, cached=True,
            )

        self.budget.ensure_affordable(self.config.model, self.estimate_cost(requirement))
        request = self._request(requirement)
        started = time.perf_counter()
        response = self._client.messages.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000

        usage = response.usage
        cost = cost_usd(self.config.model, usage.input_tokens, usage.output_tokens)
        self.budget.record(self.config.model, cost, key)

        label = None
        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if b.type == "text"), None)
            if text is not None:
                label = json.loads(text).get("label")
        body = response.model_dump(mode="json")
        body["_label"] = label

        self.cache.put(
            key=key,
            provider=PROVIDER,
            model_requested=self.config.model,
            model_reported=response.model,
            question_set_hash=self.question_set_hash,
            sample_index=sample_index,
            request=request,
            response=body,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        return ClaudeResult(
            key, label, response.stop_reason, usage.input_tokens, usage.output_tokens,
            cost, latency_ms, cached=False,
        )
