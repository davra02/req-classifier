"""Jev variants and robustness measures, with a fake TypeSafe client (no API calls)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from reqbench.eval import bootstrap
from reqbench.models import jev
from reqbench.models.baselines import LABELS
from reqcore.budget import Budget, BudgetConfig
from reqcore.cache import InferenceCache
from reqcore.clients.jev import PINNED_MODEL, JevClient, ModelVersionMismatchError

NFR = jev.NFR


class FakeTypeSafe:
    """Puts extra mass on the first option of each Choice (a first-position bias)."""

    def __init__(self, model: str = PINNED_MODEL):
        self.model, self.calls = model, 0

    def system_one(self, state, questions, model):
        self.calls += 1

        def choice(options):
            probs = {o: 0.1 for o in options}
            probs[options[0]] += 0.3
            probs["security" if "security" in options else options[-1]] += 0.2
            total = sum(probs.values())
            probs = {k: round(v / total, 2) for k, v in probs.items()}
            return {"type": "choice", "choice": max(probs, key=probs.get),
                    "confidence": 0.5, "probabilities": probs}  # fmt: skip

        answers = {qid: choice(list(q["criteria"])) for qid, q in questions.items()
                   if q["type"] == "choice"}  # fmt: skip
        answers |= {qid: {"type": "noul", "noul": 0.25} for qid, q in questions.items()
                    if q["type"] == "noul"}  # fmt: skip
        body = {"model": self.model, "usage": {"input_tokens": 2000, "output_tokens": 10},
                "answers": answers}  # fmt: skip
        return SimpleNamespace(
            model=self.model, usage=SimpleNamespace(input_tokens=2000, output_tokens=10),
            model_dump=lambda mode="json": body,
        )


@pytest.fixture
def client(tmp_path):
    def make(model=PINNED_MODEL):
        budget = Budget(BudgetConfig(10.0, {"jev": 2.0, "haiku": 2.0, "opus": 3.0}),
                        ledger_path=tmp_path / "ledger.jsonl")  # fmt: skip
        return JevClient(cache=InferenceCache(tmp_path / "cache.sqlite"), budget=budget,
                         client=FakeTypeSafe(model))  # fmt: skip
    return make


ROWS = pd.DataFrame({"req_id": ["r1", "r2"], "text": ["The system shall log in users.",
                                                      "Pages load in 2 s."]})  # fmt: skip


def test_plan_calls_rotations_and_repeats():
    calls = jev.plan_calls(ROWS, rotations=(0, 1, 2), n_samples=5)
    assert len(calls) == 2 * (3 + 4)
    assert {c.nonce for c in calls if c.sample_index == 0} == {None}
    assert {c.rotation for c in calls if c.sample_index > 0} == {0}
    assert len({(c.req_id, c.rotation, c.sample_index) for c in calls}) == len(calls)


def test_dry_run_makes_no_call_then_run_fills_cache(client):
    c = client()
    qsets = jev.question_sets()
    calls = jev.plan_calls(ROWS)
    est = jev.estimate(calls, c, qsets)
    assert est["calls_to_make"] == len(calls) and est["calls_cached"] == 0
    assert jev.collect(calls, c, qsets, run=False).empty
    assert c._client.calls == 0
    answers = jev.collect(calls, c, qsets, run=True, workers=2, progress=False)
    assert len(answers) == len(calls) and c._client.calls == len(calls)
    assert jev.estimate(calls, c, qsets)["calls_to_make"] == 0
    again = jev.collect(calls, c, qsets, run=False)
    assert again["cached"].all() and c._client.calls == len(calls)


def test_wrong_model_aborts(client):
    c = client(model="jev-1.13.0")
    with pytest.raises(ModelVersionMismatchError):
        jev.collect(jev.plan_calls(ROWS[:1], (0,), 1), c, jev.question_sets((0,)), run=True,
                    workers=1, progress=False)  # fmt: skip


def _answers(rows: list[dict]) -> pd.DataFrame:
    base = {f"rt_{k}": 0.0 for k in LABELS} | {f"ns_{k}": 0.0 for k in NFR}
    base |= {"nf_is_functional": 0.5, "nf_is_requirement": 1.0, "nf_addresses_ai": 0.0}
    return pd.DataFrame([base | row for row in rows])


def test_flat_and_hierarchical_composition():
    answers = _answers([
        {"req_id": "a", "rotation": 0, "sample_index": 0, "rt_security": 1.0,
         "ns_security": 0.6, "ns_performance": 0.4, "nf_is_functional": 0.2},
        {"req_id": "a", "rotation": 1, "sample_index": 0, "rt_functional": 1.0,
         "ns_security": 0.6, "ns_performance": 0.4, "nf_is_functional": 0.2},
    ])  # fmt: skip
    j1 = jev.flat_probs(answers, (0,)).loc["a"]
    assert j1["security"] == 1.0
    j2 = jev.flat_probs(answers, (0, 1)).loc["a"]
    assert j2["security"] == pytest.approx(0.5) and j2["functional"] == pytest.approx(0.5)
    j4 = jev.hierarchical_probs(answers, (0,)).loc["a"]
    assert j4["functional"] == pytest.approx(0.2)
    assert j4["security"] == pytest.approx(0.8 * 0.6)
    assert j4["performance"] == pytest.approx(0.8 * 0.4)
    assert j4.sum() == pytest.approx(1.0)


def test_stability_and_order_on_fake_bias(client):
    c = client()
    qsets = jev.question_sets()
    answers = jev.collect(jev.plan_calls(ROWS), c, qsets, run=True, workers=2, progress=False)
    stab = jev.stability(answers)
    assert stab["J1"]["flip_rate_vs_sample0"] == 0.0  # the fake is deterministic
    order = jev.order_sensitivity(answers)
    lift = order["first_position"]["req_type"]
    # The fake adds mass to whichever option is listed first.
    assert lift["functional"]["lift"] > 0 and lift["performance"]["lift"] > 0
    assert order["req_type"]["share_rows_any_change"] == 1.0
    usage = jev.usage_summary(answers)
    assert usage["usd_per_1000_requirements"]["J2"] == pytest.approx(3 * 1000 * 2000 * 0.042e-6)


def test_paired_difference_p_value_and_holm():
    gold = np.array([1] * 50 + [0] * 50)
    better = gold.copy()
    worse = np.where(np.arange(100) % 3 == 0, 1 - gold, gold)
    diff = bootstrap.paired_difference(
        lambda i: float(np.mean(better[i] == gold[i])),
        lambda i: float(np.mean(worse[i] == gold[i])), 100, n_resamples=500)
    assert diff["value"] > 0 and diff["ci95"][0] > 0 and diff["p_two_sided"] < 0.01
    adjusted = bootstrap.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == {"a": pytest.approx(0.03), "c": pytest.approx(0.06),
                        "b": pytest.approx(0.06)}  # fmt: skip
