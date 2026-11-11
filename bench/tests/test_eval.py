import numpy as np
import pytest
from reqbench.eval import bootstrap, metrics
from reqbench.models.baselines import LABELS, Keywords, Majority, TfidfLR

NFR = ("performance", "security", "usability", "operational_constraint", "other")


def test_classification_metrics_on_known_values():
    gold = ["functional", "functional", "security", "performance"]
    pred = ["functional", "security", "security", "functional"]
    pc = metrics.per_class(gold, pred, LABELS)
    assert pc["functional"]["precision"] == 0.5 and pc["functional"]["recall"] == 0.5
    assert pc["security"]["f1"] == pytest.approx(2 / 3)
    assert pc["usability"]["f1"] == 0.0 and pc["usability"]["support"] == 0
    assert metrics.accuracy(gold, pred) == 0.5
    assert metrics.macro_f1(gold, gold, LABELS[:3]) == 1.0
    # Level 1: F/NFR; level 2 only on gold-NFR rows (security right, performance missed).
    assert metrics.level1_f1(gold, pred) == pytest.approx(0.5)  # F: 1 TP, 1 FP, 1 FN; NFR same
    assert metrics.level2_macro_f1(gold, pred, NFR) == pytest.approx(1 / 5)
    cm = metrics.confusion(gold, pred, LABELS)
    assert cm.sum() == 4 and cm[LABELS.index("functional"), LABELS.index("security")] == 1


def _probs(rows):
    return np.array([[p if label == c else (1 - p) / 5 for label in LABELS] for c, p in rows])


def test_probability_metrics_on_known_values():
    gold = ["functional", "security"]
    probs = _probs([("functional", 1.0), ("security", 1.0)])
    assert metrics.log_loss(gold, probs, LABELS) == pytest.approx(0.0, abs=1e-12)
    assert metrics.brier(gold, probs, LABELS) == pytest.approx(0.0)
    assert metrics.ece_top_label(gold, probs, LABELS) == pytest.approx(0.0)
    # Fully confident and right half of the time: top-label ECE 0.5.
    gold = ["functional", "security"] * 10
    probs = _probs([("functional", 1.0), ("functional", 1.0)] * 10)
    assert metrics.ece_top_label(gold, probs, LABELS) == pytest.approx(0.5)
    assert metrics.log_loss(["functional"], _probs([("functional", 0.5)]), LABELS) == (
        pytest.approx(np.log(2)))


def test_selective_metrics_rank_by_confidence():
    gold = ["functional", "security", "usability", "other"]
    probs = _probs([("functional", 0.9), ("functional", 0.8), ("usability", 0.7),
                    ("other", 0.6)])  # fmt: skip
    risk = metrics.risk_coverage(gold, probs, LABELS)
    assert risk.tolist() == pytest.approx([0.0, 0.5, 1 / 3, 0.25])
    assert metrics.aurc(gold, probs, LABELS) == pytest.approx(np.mean([0, 0.5, 1 / 3, 0.25]))
    assert metrics.coverage_at_precision(gold, probs, LABELS) == 0.25
    assert metrics.coverage_at_precision(gold, probs, LABELS, target=0.75) == 1.0


def test_selective_metrics_do_not_depend_on_order_of_ties():
    # three rows tied at 0.9 (one wrong), then one at 0.5 (wrong)
    gold = ["functional", "security", "functional", "other"]
    probs = _probs([("functional", 0.9), ("functional", 0.9), ("functional", 0.9),
                    ("functional", 0.5)])  # fmt: skip
    risk = metrics.risk_coverage(gold, probs, LABELS)
    assert risk.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.5])
    for perm in ([1, 0, 2, 3], [2, 1, 0, 3], [3, 1, 2, 0]):
        g, p = [gold[i] for i in perm], probs[perm]
        assert metrics.aurc(g, p, LABELS) == pytest.approx(metrics.aurc(gold, probs, LABELS))
        # a threshold can keep all three tied rows (precision 2/3) or none
        assert metrics.coverage_at_precision(g, p, LABELS, target=0.6) == 0.75
        assert metrics.coverage_at_precision(g, p, LABELS, target=0.7) == 0.0


def test_bootstrap_is_deterministic_and_paired():
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    first = bootstrap.ci(lambda i: values[i].mean(), len(values), n_resamples=500)
    assert first == bootstrap.ci(lambda i: values[i].mean(), len(values), n_resamples=500)
    assert first["ci95"][0] <= first["value"] <= first["ci95"][1]
    same = bootstrap.paired_difference(lambda i: values[i].mean(), lambda i: values[i].mean(),
                                       len(values), n_resamples=200)  # fmt: skip
    assert same == {"value": 0.0, "ci95": [0.0, 0.0], "p_two_sided": 1.0}


def test_keywords_use_stems_phrases_priority_and_default():
    model = Keywords()
    assert model.classify("The system shall encrypt all stored passwords.")[0] == "security"
    assert model.classify("Pages shall load within 2 seconds.")[0] == "performance"
    # security outranks usability when both match
    assert model.classify("Login shall be easy for new users.")[0] == "security"
    assert model.classify("The product shall run on Windows and Linux.")[0] == (
        "operational_constraint")
    # A keyword never matches inside a word; "*" only extends the ending.
    assert model.classify("Users shall be able to export invoices.")[0] == "functional"
    assert model.classify("The pipeline shall accept nonstandard input.")[0] == "functional"
    assert model.classify("The pipeline shall standardize dates.")[0] == "operational_constraint"
    assert model.classify("Data shall follow the ISO standards.")[0] == "operational_constraint"


def test_majority_and_tfidf_follow_label_order():
    texts = [f"the system shall {w} item {i}" for i in range(6)
             for w in ("display", "encrypt", "load quickly")]  # fmt: skip
    labels = [lab for _ in range(6) for lab in ("functional", "security", "performance")]
    groups = [i % 3 for i in range(len(texts))]
    assert Majority().fit(texts, labels).predict(["x"]) == ["functional"]  # tie: taxonomy order
    assert Majority().fit(texts, labels[:-1]).predict(["x"]) == ["functional"]
    model = TfidfLR().fit(texts, labels, groups)
    proba = model.predict_proba(["the system shall encrypt item 9"])
    assert proba.shape == (1, len(LABELS)) and proba.sum() == pytest.approx(1.0)
    assert proba[0, LABELS.index("usability")] == 0.0  # unseen class
    assert model.predict(["the system shall encrypt item 9"]) == ["security"]


def test_dev_loader_never_returns_test_rows():
    from reqbench.eval.cv import load_dev
    from reqcore.paths import data_dir

    if not (data_dir() / "processed" / "promise_v1.csv").exists():
        pytest.skip("processed data not built (`reqbench data all`)")
    dev = load_dev()
    assert set(dev["split"]) == {"dev"} and len(dev) == 449
    assert dev["is_example"].sum() == 12
