import itertools
import json

import pandas as pd
import pytest
from reqbench.fresh import agreement
from reqbench.fresh.agreement import (
    ASSIGN_COLUMN,
    apply_reply,
    assign_subsets,
    checklist_message,
    parse_reply,
)
from reqbench.fresh.agreement import (
    agreement as kappa_report,
)

# Same mix as annotator A's real labels: rare labels must still reach the subsamples.
MIX = {"functional": 73, "other": 41, "security": 21, "operational_constraint": 19,
       "usability": 9, "performance": 4, "discard": 3}  # fmt: skip


def _frame():
    labels = [label for label, n in MIX.items() for _ in range(n)]
    order = sorted(range(len(labels)), key=lambda i: (i * 37) % len(labels))  # interleave
    return pd.DataFrame(
        [
            {"req_id": f"fresh-{i + 1:04d}", "doc_id": "d", "location": "l",
             "text": f"Sentence number {i + 1} shall be classified.",
             "annotator_a": labels[j], "annotator_b": "", "adjudicated": "",
             "adjudication_note": ""}
            for i, j in enumerate(order)
        ]
    )  # fmt: skip


def test_assign_draws_disjoint_stratified_subsamples_deterministically():
    first = assign_subsets(_frame(), seed=7)
    assert first.equals(assign_subsets(_frame(), seed=7))
    counts = first[ASSIGN_COLUMN].value_counts().to_dict()
    assert counts == {"": 110, "b": 30, "c": 30}
    chosen = first[first[ASSIGN_COLUMN] != ""]
    per_label = chosen["annotator_a"].value_counts()
    assert set(per_label.index) == set(MIX)
    assert per_label.min() >= 2
    for label in MIX:  # both annotators get a similar mix
        mine = chosen[chosen["annotator_a"] == label][ASSIGN_COLUMN].value_counts()
        assert abs(mine.get("b", 0) - mine.get("c", 0)) <= 1
    assert list(first.columns) == agreement.COLUMN_ORDER
    assert not first.equals(assign_subsets(_frame(), seed=8))


def test_assign_refuses_before_a_finishes_or_after_b_starts():
    unfinished = _frame()
    unfinished.loc[0, "annotator_a"] = ""
    with pytest.raises(RuntimeError, match="has not finished"):
        assign_subsets(unfinished)
    started = assign_subsets(_frame())
    started.loc[started[ASSIGN_COLUMN] == "b", "annotator_b"] = "functional"
    with pytest.raises(RuntimeError, match="already started"):
        assign_subsets(started)


def test_message_lists_own_sentences_in_csv_order_without_labels():
    df = assign_subsets(_frame())
    message = checklist_message(df, "b")
    mine = df[df[ASSIGN_COLUMN] == "b"]["text"].tolist()
    positions = [message.index(f"{n}. {text}") for n, text in enumerate(mine, 1)]
    assert positions == sorted(positions) and len(positions) == 30
    assert "F = functional" in message and "D = discard" in message
    for text in df[df[ASSIGN_COLUMN] == "c"]["text"]:
        assert text not in message
    body = message.split("1. ", 1)[1]
    assert not any(label in body for label in ("operational_constraint", "performance"))


def test_parse_reply_accepts_whatsapp_formats_and_rejects_mistakes():
    df = assign_subsets(_frame())
    ids = df[df[ASSIGN_COLUMN] == "b"]["req_id"].tolist()
    letters = itertools.cycle("FPSUCOD")
    reply = "\n".join(f"{n}{sep}{next(letters)}" for n, sep in
                      zip(range(1, 31), itertools.cycle([" ", "-", ": ", "", ") "]), strict=False))
    labels = parse_reply(df, "b", reply.lower())
    assert labels[ids[0]] == "functional" and labels[ids[6]] == "discard" and len(labels) == 30
    with pytest.raises(ValueError, match="missing items"):
        parse_reply(df, "b", "1F 2S")
    with pytest.raises(ValueError, match="unknown letter"):
        parse_reply(df, "b", reply + " 3Z")
    with pytest.raises(ValueError, match="answered twice"):
        parse_reply(df, "b", reply + " 1S")
    with pytest.raises(ValueError, match="does not exist"):
        parse_reply(df, "b", reply + " 31F")


def test_apply_reply_fills_own_column_once():
    df = assign_subsets(_frame())
    ids = df[df[ASSIGN_COLUMN] == "c"]["req_id"].tolist()
    out = apply_reply(df, "c", dict.fromkeys(ids, "security"))
    assert (out.loc[out[ASSIGN_COLUMN] == "c", "annotator_c"] == "security").all()
    assert (out.loc[out[ASSIGN_COLUMN] != "c", "annotator_c"] == "").all()
    assert (out["annotator_b"] == "").all()
    with pytest.raises(RuntimeError, match="already has labels"):
        apply_reply(out, "c", dict.fromkeys(ids, "usability"))


def test_kappa_reports_perfect_and_pending_pairs():
    df = assign_subsets(_frame())
    b = df[ASSIGN_COLUMN] == "b"
    df.loc[b, "annotator_b"] = df.loc[b, "annotator_a"]
    report = kappa_report(df)
    assert report["A-B"]["leaves_with_discard"]["kappa"] == 1.0
    assert report["A-B"]["disagreements"] == [] and report["A-B"]["pending"] == 0
    assert report["A-C"]["pending"] == 30 and report["A-C"]["leaves_with_discard"] == {"n": 0}
    assert report["A-second"]["leaves_with_discard"]["n"] == 30


def test_form_embeds_only_own_sentences_and_escapes_script_end():
    df = assign_subsets(_frame())
    b = df[ASSIGN_COLUMN] == "b"
    df.loc[df.index[b][0], "text"] = "The page shall not break on </script><b>x</b> input."
    html = agreement.render_form(df, "b")
    assert "</script><b>" not in html and "<\\/script><b>" in html
    assert "__DATA__" not in html and "__WHO__" not in html
    for text in df.loc[df[ASSIGN_COLUMN] == "c", "text"]:
        assert text not in html
    start = html.index('id="form-data">') + len('id="form-data">')
    data = json.loads(html[start : html.index("</script>", start)])
    assert data["who"] == "B" and len(data["items"]) == 30
    assert all(set(item) == {"n", "text"} for item in data["items"])  # no labels of anyone


def test_reply_signed_by_another_annotator_is_rejected():
    df = assign_subsets(_frame())
    reply = "Respuestas del anotador C:\n" + " ".join(f"{n}F" for n in range(1, 31))
    with pytest.raises(ValueError, match="signed by annotator"):
        parse_reply(df, "b", reply)
    assert len(parse_reply(df, "c", reply)) == 30


def test_reserve_subsample_is_disjoint_stratified_and_counted_in_kappa():
    df = assign_subsets(_frame())
    out = agreement.add_subset(df, "d")
    assert (out[ASSIGN_COLUMN] == "d").sum() == 30
    kept = out[ASSIGN_COLUMN].isin(["b", "c"])
    assert out.loc[kept, ASSIGN_COLUMN].equals(df.loc[kept, ASSIGN_COLUMN])  # B and C untouched
    assert list(out.columns).index("annotator_d") == list(out.columns).index(ASSIGN_COLUMN) - 1
    assert out.equals(agreement.add_subset(df, "d"))
    assert out[out[ASSIGN_COLUMN] == "d"]["annotator_a"].nunique() >= 5
    with pytest.raises(RuntimeError, match="already has a subsample"):
        agreement.add_subset(out, "d")
    with pytest.raises(ValueError):
        agreement.add_subset(out, "b")
    d = out[ASSIGN_COLUMN] == "d"
    out.loc[d, "annotator_d"] = out.loc[d, "annotator_a"]
    report = kappa_report(out)
    assert set(report) == {"A-B", "A-C", "A-D", "A-second"}
    assert report["A-D"]["leaves_with_discard"]["kappa"] == 1.0
    assert report["A-second"]["pending"] == 60
