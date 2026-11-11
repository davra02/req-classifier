import numpy as np
import pytest
from reqbench.eval import calibration, metrics
from reqbench.models.baselines import LABELS


def _overconfident(n=600, k=6, seed=0):
    """Synthetic multiclass data: right 70 % of the time, but always claims ~0.97."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    pred = np.where(rng.random(n) < 0.7, y, (y + 1) % k)
    probs = np.full((n, k), 0.006)
    probs[np.arange(n), pred] = 0.97
    return probs / probs.sum(axis=1, keepdims=True), y, np.arange(n) % 5


def test_temperature_reduces_overconfidence_and_keeps_argmax():
    probs, y, _ = _overconfident()
    cal = calibration.Temperature().fit(probs, y)
    out = cal.transform(probs)
    assert cal.t_ > 1
    assert (out.argmax(axis=1) == probs.argmax(axis=1)).all()
    assert calibration.nll(out, y) < calibration.nll(probs, y)
    assert np.allclose(out.sum(axis=1), 1)


def test_vector_isotonic_and_platt_are_proper_distributions():
    probs, y, _ = _overconfident()
    for name in ("vector", "isotonic"):
        out = calibration.FACTORIES[name]().fit(probs, y).transform(probs)
        assert np.allclose(out.sum(axis=1), 1) and (out >= 0).all()
    p = np.clip(probs[:, 0], 0.01, 0.99)
    binary = np.column_stack([1 - p, p])
    yb = (y == 0).astype(int)
    out = calibration.Platt().fit(binary, yb).transform(binary)
    assert np.allclose(out.sum(axis=1), 1)


def test_identity_wins_when_already_calibrated():
    rng = np.random.default_rng(1)
    n, k = 2000, 3
    probs = rng.dirichlet(np.ones(k), n)
    y = np.array([rng.choice(k, p=row) for row in probs])
    best, scores = calibration.select(probs, y, np.arange(n) % 5,
                                      ("identity", "temperature"))  # fmt: skip
    assert scores["temperature"] >= scores["identity"] - 0.005  # no real gain from scaling


def test_nested_oof_and_round_trip():
    probs, y, folds = _overconfident()
    res = calibration.nested_oof(probs, y, folds)
    assert res.calibrated.shape == probs.shape and set(res.chosen_per_fold) == set(range(5))
    assert calibration.nll(res.calibrated, y) < calibration.nll(probs, y)
    for name in ("temperature", "vector", "isotonic"):
        cal = calibration.FACTORIES[name]().fit(probs, y)
        again = calibration.from_dict(calibration.to_dict(cal))
        assert np.allclose(again.transform(probs), cal.transform(probs), atol=1e-9)


def test_nested_oof_fit_mask_uses_only_masked_rows():
    probs, y, folds = _overconfident()
    mask = y != 0
    res = calibration.nested_oof(probs, y, folds, fit_mask=mask)
    assert res.calibrated.shape == probs.shape  # every row transformed


def test_threshold_and_selective():
    gold = ["functional"] * 10 + ["security"] * 2
    probs = np.zeros((12, len(LABELS)))
    probs[:, 0] = [0.99] * 10 + [0.6, 0.6]  # two wrong, least confident
    probs[:, 1] = 1 - probs[:, 0]
    t = metrics.threshold_at_precision(gold, probs, LABELS)
    assert t == pytest.approx(0.99)
    pred = np.array([LABELS[i] for i in probs.argmax(axis=1)])
    sel = metrics.selective(gold, pred, probs, t)
    assert sel["coverage"] == pytest.approx(10 / 12) and sel["precision"] == 1.0
    assert metrics.selective(gold, pred, probs, None)["coverage"] == 0.0
    # A wrong answer inside the most confident block: no threshold reaches 0.95.
    probs[0, 0], probs[0, 1] = 0.99, 0.01
    gold[0] = "security"
    assert metrics.threshold_at_precision(gold, probs, LABELS) is None


# ----- adversarial suite and decision rule ----------------------------------------------

def test_adversarial_allocation_and_flip_rates():
    import pandas as pd
    from reqbench.eval import adversarial

    counts = pd.Series({"a": 184, "b": 52, "c": 45, "d": 74, "e": 50, "f": 32})
    alloc = adversarial._allocation(counts, 20)
    assert sum(alloc.values()) == 20 and min(alloc.values()) >= 2
    adv = pd.DataFrame({"req_id": ["r1", "r1", "r2"], "variant_id": ["v1", "v2", "v3"],
                        "attack": ["x", "y", "x"], "target": ["other", "other", "security"]})
    orig = pd.Series({"r1": "functional", "r2": "usability"})
    new = pd.Series({"v1": "functional", "v2": "other", "v3": "security"})
    out = adversarial.flip_rates(adv, orig, new)
    assert out["flip_rate"] == pytest.approx(2 / 3)
    assert out["attack_success"] == pytest.approx(2 / 3)


def _results(j1=0.83, b5=0.86, lo=-0.02, hi=0.01, aurc_hi=-0.01, ece=0.04, adv_j=0.0, adv_b=0.1):
    from reqbench.eval import heldout

    names = [*heldout.JEV_SYSTEMS, *heldout.BSTAR_CANDIDATES[:-2], *heldout.B4_SEEDS,
             "B5_haiku_k5"]
    s = {n: {"macro_f1": 0.6} for n in names}
    s["J1_jev_choice"]["macro_f1"], s["B5_haiku_k5"]["macro_f1"] = j1, b5
    for j in heldout.JEV_SYSTEMS:
        s[f"{j}_cal"] = {"ece": ece}
    diff = {"macro_f1": {"value": j1 - b5, "ci95": [lo, hi]}, "aurc": {"ci95": [-0.1, aurc_hi]}}
    big = {"macro_f1": {"value": 0.2, "ci95": [0.15, 0.25]}}
    comp = {j: {o: big for o in names} | {"B5_haiku_k5": diff} for j in heldout.JEV_SYSTEMS}
    return {
        "n": 171, "summaries": s, "comparisons": comp,
        "robustness": {"jev_stability": {"J1": {"flip_rate_vs_sample0": 0.01},
                                         "J4": {"flip_rate_vs_sample0": 0.01}},
                       "jev_order": {"req_type": {"flip_rate_pairwise": 0.02},
                                     "J4": {"flip_rate_pairwise": 0.02}}},
        "latency": {"jev": {"serial_ms_p95": 300.0,
                            "three_rotations_per_requirement_ms_p95": 400.0}},
        "cost": {"jev_usd_per_1000": {"J1": 0.09, "J2": 0.27}},
        "adversarial": {"J1_jev_choice": {"flip_rate": adv_j},
                        "J4_jev_hierarchical": {"flip_rate": adv_j},
                        "B5_haiku_k5": {"flip_rate": adv_b}},
    }


def test_decide_go_inconclusive_nogo():
    from reqbench.eval import heldout

    go = heldout.decide(_results())
    assert go["bstar"] == "B5_haiku_k5" and go["best_jev"] == "J1_jev_choice"
    assert go["criterion_1"] and go["criterion_2"]["b"] and go["verdict"] == "GO"
    # Non-inferiority fails (lower bound below -δ), no AURC advantage: inconclusive.
    inc = heldout.decide(_results(j1=0.845, lo=-0.06, aurc_hi=0.01))
    assert inc["verdict"].startswith("INCONCLUSIVE")
    assert inc["n_needed_for_non_inferiority_approx"] is not None
    # Whole CI below -δ: NO-GO.
    assert heldout.decide(_results(j1=0.75, lo=-0.15, hi=-0.05))["verdict"] == "NO-GO"
    # ECE above 0.05 blocks GO.
    assert heldout.decide(_results(ece=0.08))["verdict"] != "GO"
