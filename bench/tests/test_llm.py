"""B5 (Claude Haiku) helpers and the Batch API path, with a fake Anthropic client."""

from types import SimpleNamespace

import pandas as pd
import pytest
from reqbench.models import llm
from reqbench.models.baselines import LABELS
from reqcore.budget import Budget, BudgetConfig
from reqcore.cache import InferenceCache
from reqcore.clients.claude import HAIKU, BatchItem, ClaudeClassifier, ModelMismatchError
from reqcore.questions import render_llm_prompt


def _message(model, label="security", stop="end_turn"):
    body = {"model": model, "stop_reason": stop}
    return SimpleNamespace(
        model=model, stop_reason=stop,
        content=[SimpleNamespace(type="text", text=f'{{"label": "{label}"}}')],
        usage=SimpleNamespace(input_tokens=1200, output_tokens=10),
        model_dump=lambda mode: dict(body),
    )  # fmt: skip


class FakeBatches:
    def __init__(self, parent):
        self.parent, self.created, self.retrieved = parent, [], 0

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}")

    def retrieve(self, batch_id):
        self.retrieved += 1
        status = "ended" if self.retrieved > 1 else "in_progress"
        counts = SimpleNamespace(processing=0, succeeded=0)
        return SimpleNamespace(processing_status=status, request_counts=counts)

    def results(self, batch_id):
        requests = self.created[int(batch_id.split("_")[1]) - 1]
        out = []
        for i, r in enumerate(requests):
            if i == 0:  # one request fails: it must not be cached
                out.append(SimpleNamespace(custom_id=r["custom_id"],
                                           result=SimpleNamespace(type="errored")))  # fmt: skip
            else:
                msg = _message(self.parent.model, label="Security")  # capitalization drift
                out.append(SimpleNamespace(
                    custom_id=r["custom_id"],
                    result=SimpleNamespace(type="succeeded", message=msg)))  # fmt: skip
        return reversed(out)  # results come back in any order


class FakeAnthropic:
    def __init__(self, model=HAIKU.model):
        self.model, self.requests = model, []
        self.messages = SimpleNamespace(create=self._create, batches=FakeBatches(self))

    def _create(self, **request):
        self.requests.append(request)
        return _message(self.model)


@pytest.fixture
def make(tmp_path):
    def build(model=HAIKU.model, rotation=0):
        budget = Budget(BudgetConfig(10.0, {"jev": 2.0, "haiku": 2.0, "opus": 3.0}),
                        ledger_path=tmp_path / "ledger.jsonl")  # fmt: skip
        fake = FakeAnthropic(model)
        clf = ClaudeClassifier(HAIKU, render_llm_prompt(rotation=rotation),
                               cache=InferenceCache(tmp_path / "c.sqlite"), budget=budget,
                               client=fake, state_dir=tmp_path / "batches")  # fmt: skip
        return clf, fake, budget
    return build


def test_llm_prompt_rotation_rotates_definitions_and_enum():
    r0, r1 = render_llm_prompt(), render_llm_prompt(rotation=1)
    assert r0.labels == LABELS and r1.labels == LABELS[1:] + LABELS[:1]
    assert r1.output_schema()["properties"]["label"]["enum"] == list(r1.labels)
    assert r1.system.index(f"- {LABELS[1]}:") < r1.system.index(f"- {LABELS[0]}:")
    assert r0.hash != r1.hash and r0.hash == render_llm_prompt(rotation=0).hash


def test_batch_submit_resume_and_ingest(make):
    clf, fake, budget = make()
    items = [BatchItem(f"Req {i} shall be secure.", s) for i in range(3) for s in range(2)]
    batch_id = clf.submit_batch(items + items[:2])  # duplicates are sent once
    assert len(fake.messages.batches.created[0]) == 6
    request = fake.messages.batches.created[0][0]
    assert request["params"]["thinking"] == {"type": "disabled"}
    assert "Req 0" not in request["params"]["system"]
    assert clf.submit_batch(items) is None  # pending: never resubmitted
    outcome = clf.collect_batch(batch_id, poll_seconds=0, log=lambda *_: None)
    assert outcome["succeeded"] == 5 and outcome["errored"] == 1
    assert clf.pending_batches() == []
    cached = [clf.cached(i.requirement, i.sample_index) for i in items]
    assert sum(r is not None for r in cached) == 5
    hit = next(r for r in cached if r is not None)
    assert hit.label == "security" and hit.batch and hit.latency_ms is None
    assert hit.cost_usd == pytest.approx((1200 * 0.10 + 10 * 0.50) / 1e6 / 2)
    assert budget.spent_by_bucket()["haiku"] == pytest.approx(5 * hit.cost_usd)
    # the errored request is resubmitted alone
    clf.submit_batch(items)
    assert len(fake.messages.batches.created[1]) == 1


def test_classify_rejects_other_model(make):
    clf, _, budget = make(model="claude-haiku-9")
    with pytest.raises(ModelMismatchError):
        clf.classify("The system shall encrypt data.")
    assert budget.spent_by_bucket()["haiku"] > 0
    assert clf.cached("The system shall encrypt data.") is None


def _answers(labels_by_req: dict[str, list[str | None]], rotation=0) -> pd.DataFrame:
    return pd.DataFrame([
        {"req_id": rid, "rotation": rotation, "sample_index": s, "label": lab}
        for rid, labels in labels_by_req.items() for s, lab in enumerate(labels)
    ])  # fmt: skip


def test_frequency_probs_ties_and_invalid_samples():
    answers = _answers({
        "a": ["security", "security", "usability", "security", "usability"],
        "b": ["usability", "security", "security", "usability", None],  # 2-2 tie
        "c": [None] * 5,
    })  # fmt: skip
    probs = llm.frequency_probs(answers)
    assert probs.loc["a", "security"] == pytest.approx(0.6) and probs.loc["a", "pred"] == "security"
    assert probs.loc["b", "pred"] == "usability"  # drawn first among the tied
    assert probs.loc["b", "n_valid"] == 4 and probs.loc["b", "security"] == pytest.approx(0.5)
    assert probs.loc["c", list(LABELS)].sum() == pytest.approx(1.0)
    assert probs.loc["c", "n_valid"] == 0


def test_stability_and_order():
    labels = {"a": ["security"] * 5, "b": ["security", "usability", "security", "security",
                                          "security"]}  # fmt: skip
    base = _answers(labels)
    rot = pd.concat([
        _answers({"a": ["security"], "b": ["performance"]}, rotation=1),
        _answers({"a": ["security"], "b": ["security"]}, rotation=2),
    ])  # fmt: skip
    stab = llm.stability(base)
    assert stab["flip_rate_vs_sample0"] == pytest.approx(1 / 8)
    assert stab["share_rows_any_change"] == pytest.approx(0.5)
    assert stab["share_unanimous"] == pytest.approx(0.5)
    order = llm.order_sensitivity(pd.concat([base, rot]))
    assert order["first_label"] == {0: LABELS[0], 1: LABELS[1], 2: LABELS[2]}
    assert order["flip_rate_pairwise"] == pytest.approx(2 / 6)
    assert order["noise_floor_same_order"] == pytest.approx(0.5)


def test_plan_calls_mirrors_jev_count():
    rows = pd.DataFrame({"req_id": ["r1", "r2"], "text": ["x", "y"]})
    calls = llm.plan_calls(rows)
    assert len(calls) == 2 * (llm.K + 2)
    assert {(c.rotation, c.sample_index) for c in calls if c.req_id == "r1"} == (
        {(0, s) for s in range(llm.K)} | {(1, 0), (2, 0)})  # fmt: skip
