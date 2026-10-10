import itertools

import pandas as pd
import pytest
from reqbench.data.promise import parse_arff
from reqbench.data.splits import MIN_PER_LEAF, make_splits

ARFF = """% comment
@RELATION nfr
@ATTRIBUTE ProjectID {1,2}
@DATA
1,'The system shall refresh the display every 60 seconds.',PE
2,'The user\\'s password shall be encrypted.',SE
"""


def test_parse_arff_reads_rows_and_unescapes_quotes():
    rows = parse_arff(ARFF)
    assert [(r.project_id, r.label) for r in rows] == [(1, "PE"), (2, "SE")]
    assert rows[1].text == "The user's password shall be encrypted."


def test_parse_arff_rejects_malformed_rows():
    with pytest.raises(ValueError):
        parse_arff("@DATA\nnot a row\n")


def _synthetic(n_projects=15):
    leaves = ["functional", "performance", "security", "usability", "operational_constraint",
              "other"]
    rows = []
    cycle = itertools.cycle(leaves * 2 + ["functional"] * 4)
    for p in range(1, n_projects + 1):
        for i in range(20 + 3 * p):  # unequal project sizes, like PROMISE
            rows.append({"req_id": f"r{p:02d}-{i:02d}", "text_hash": f"h{p}{i}",
                         "project_id": p, "leaf": next(cycle)})
    return pd.DataFrame(rows)


def test_splits_are_project_grouped_deterministic_and_balanced():
    df = _synthetic()
    first, meta = make_splits(df, cross_project_pairs=[])
    second, _ = make_splits(df, cross_project_pairs=[])
    pd.testing.assert_frame_equal(first, second)

    test_projects = set(first.loc[first["split"] == "test", "project_id"])
    dev_projects = set(first.loc[first["split"] == "dev", "project_id"])
    assert test_projects and not test_projects & dev_projects
    assert 0.27 <= (first["split"] == "test").mean() <= 0.33
    assert first.loc[first["split"] == "test", "leaf"].value_counts().min() >= MIN_PER_LEAF
    assert (first.loc[first["split"] == "dev", "fold"] >= 0).all()
    assert (first.loc[first["split"] != "dev", "fold"] == -1).all()
    # Folds are grouped by project too.
    dev = first[first["split"] == "dev"]
    assert (dev.groupby("project_id")["fold"].nunique() == 1).all()
    assert meta["test_projects"] == sorted(test_projects)


def test_cross_project_pair_straddling_split_loses_test_member():
    df = _synthetic()
    base, _ = make_splits(df, cross_project_pairs=[])
    test_id = base.loc[base["split"] == "test", "req_id"].iloc[0]
    dev_id = base.loc[base["split"] == "dev", "req_id"].iloc[0]
    out, meta = make_splits(df, cross_project_pairs=[[dev_id, test_id, 0.93]])
    assert out.set_index("req_id").at[test_id, "split"] == "excluded"
    assert meta["excluded_from_test"] == [test_id]
