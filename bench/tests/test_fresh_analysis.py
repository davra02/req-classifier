import numpy as np
import pandas as pd
from reqbench.eval import fresh_analysis as fa
from reqbench.models.baselines import LABELS


def _frame(gold, pred, prefix="r"):
    return pd.DataFrame({"req_id": [f"{prefix}{i:03d}" for i in range(len(gold))],
                         "project_id": 1, "fold": -1, "gold": gold, "is_example": False,
                         "pred": pred})


def _systems(gold, good, noisy, prefix):
    out = {n: _frame(gold, noisy, prefix) for n in fa.REPORTED}
    out[fa.JEV] = _frame(gold, good, prefix)
    return out


def test_r1_r2_and_kappa_on_synthetic_sets():
    rng = np.random.default_rng(0)
    gold = list(rng.choice(LABELS, 150))
    noisy = [g if rng.random() < 0.6 else LABELS[(LABELS.index(g) + 1) % 6] for g in gold]
    fresh = _systems(gold, gold, noisy, "f")  # J3 perfect, B5x 60 % right
    test = {n: _frame(gold, gold, "t") for n in fa.REPORTED}  # every system perfect on test
    out = fa.analyse(fresh, test, double={f"f{i:03d}" for i in range(40)}, n_resamples=200)
    assert out["R1_jev_clearly_better"]["met"]
    assert out["per_system"][fa.JEV]["fresh_kappa"] == 1.0
    # B5x and B4 drop alike: no contamination signal.
    assert abs(out["R2_contamination_signal"]["value"]) < 1e-9
    assert not out["R2_contamination_signal"]["met"]
    assert out["n_double_annotated"] == 40
    # Only B5x drops: R2 fires.
    fresh2 = {n: _frame(gold, gold, "f") for n in fa.REPORTED}
    fresh2[fa.LLM1] = _frame(gold, noisy, "f")
    assert fa.r2(test, fresh2, 200)["met"]
