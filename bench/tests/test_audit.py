import pytest
from reqbench.eval import audit_phase1


def test_sample_size_scales_with_power_and_gap():
    n50 = audit_phase1._n_needed(171, 0.04, 0.015, 0.5)
    n80 = audit_phase1._n_needed(171, 0.04, 0.015, 0.8)
    assert n50 == pytest.approx(171 * (0.04 / 0.015) ** 2, rel=0.01)
    assert n80 > 2 * n50
    assert audit_phase1._n_needed(171, 0.04, 0.0, 0.8) is None


def test_fixed_jev_lr_has_a_single_grid_point():
    model = audit_phase1.FixedJevLR(table=None, c=30.0)
    assert model.grid == {"C": [30.0], "class_weight": [None]}
