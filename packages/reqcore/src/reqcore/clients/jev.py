"""Jev client with version pinning, inference cache and budget guard."""

import time
from dataclasses import dataclass
from typing import Any

from typesafe_sdk import TypeSafeClient

from reqcore.budget import Budget
from reqcore.cache import InferenceCache, cache_key
from reqcore.hashing import canonical_json
from reqcore.pricing import cost_usd
from reqcore.questions import JevQuestionSet

PROVIDER = "typesafe"
PINNED_MODEL = "jev-1.13.0"
# Conservative characters-per-token ratio for the pre-call cost estimate.
_CHARS_PER_TOKEN = 3.0


class ModelVersionMismatchError(RuntimeError):
    pass


@dataclass(frozen=True)
class JevResult:
    key: str
    model_reported: str | None
    answers: dict[str, Any]
    input_tokens: int | None
    cost_usd: float | None
    latency_ms: float | None
    cached: bool


class JevClient:
    def __init__(
        self,
        model: str = PINNED_MODEL,
        cache: InferenceCache | None = None,
        budget: Budget | None = None,
        client: TypeSafeClient | None = None,
    ):
        self.model = model
        self.cache = cache or InferenceCache()
        self.budget = budget or Budget()
        # TypeSafeClient reads TYPESAFE_API_KEY from the environment.
        self._client = client or TypeSafeClient(model=model)

    def estimate_cost(self, state: dict[str, Any], question_set: JevQuestionSet) -> float:
        chars = len(canonical_json({"state": state, "questions": question_set.questions}))
        return cost_usd(self.model, int(chars / _CHARS_PER_TOKEN), 0)

    def ask(
        self,
        state: dict[str, Any],
        question_set: JevQuestionSet,
        sample_index: int = 0,
        nonce: str | None = None,
    ) -> JevResult:
        """Evaluate every question of `question_set` against `state` in a single request.

        `nonce` adds a throwaway `uid` field to the state so that deliberate repeats are
        independent draws (the pattern TypeSafe uses in its self-consistency cookbooks).
        """
        if nonce is not None:
            state = {**state, "uid": nonce}
        key = cache_key(PROVIDER, self.model, question_set.hash, state, sample_index)
        if (hit := self.cache.get(key)) is not None:
            return JevResult(
                key, hit.model_reported, hit.response["answers"], hit.input_tokens,
                hit.cost_usd, hit.latency_ms, cached=True,
            )

        self.budget.ensure_affordable(self.model, self.estimate_cost(state, question_set))
        started = time.perf_counter()
        response = self._client.system_one(state, question_set.questions, model=self.model)
        latency_ms = (time.perf_counter() - started) * 1000

        body = response.model_dump(mode="json")
        input_tokens = response.usage.input_tokens
        cost = cost_usd(self.model, input_tokens, 0)
        self.budget.record(self.model, cost, key)
        if response.model != self.model:
            raise ModelVersionMismatchError(
                f"Requested {self.model} but {response.model} answered; recalibration needed."
            )

        self.cache.put(
            key=key,
            provider=PROVIDER,
            model_requested=self.model,
            model_reported=response.model,
            question_set_hash=question_set.hash,
            sample_index=sample_index,
            request={"state": state, "question_set": question_set.version},
            response=body,
            input_tokens=input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )
        return JevResult(
            key, response.model, body["answers"], input_tokens, cost, latency_ms, cached=False
        )
