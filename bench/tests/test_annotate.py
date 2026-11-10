import pandas as pd
import pytest
from reqbench.fresh.annotate import AnnotationStore


@pytest.fixture
def store(tmp_path):
    rows = [("security", "", "", "b"), ("", "usability", "", "b"), ("", "", "", "c")]
    pd.DataFrame(
        [
            {"req_id": f"fresh-000{i}", "doc_id": "fresh-doc-01", "location": f"SRS.md#L{i}",
             "text": f"The system shall do thing {i}.", "annotator_a": a, "annotator_b": b,
             "annotator_c": c, "second_annotator": who, "adjudicated": "",
             "adjudication_note": ""}
            for i, (a, b, c, who) in enumerate(rows, 1)
        ]
    ).to_csv(tmp_path / "requirements.csv", index=False)  # fmt: skip
    pd.DataFrame(
        [{"doc_id": "fresh-doc-01", "url": "https://github.com/o/r/tree/abc123"}]
    ).to_csv(tmp_path / "documents.csv", index=False)
    return AnnotationStore(tmp_path / "requirements.csv", tmp_path / "documents.csv")


def test_next_item_is_first_unlabelled_in_own_rows(store):
    assert store.item("a")["req_id"] == "fresh-0002"
    assert store.item("b")["req_id"] == "fresh-0001"
    assert store.item("c")["req_id"] == "fresh-0003"
    assert store.item("b")["total"] == 2 and store.item("a")["total"] == 3


def test_item_never_exposes_the_other_annotator(store):
    item = store.item("b", index=0)
    assert item["label"] == "" and "security" not in item.values()
    assert not {"annotator_a", "annotator_b", "annotator_c", "adjudicated"} & item.keys()
    assert item["source_url"] == "https://github.com/o/r/blob/abc123/SRS.md#L1"


def test_save_writes_only_own_column_and_advances(store):
    store.save("a", "fresh-0002", "discard")
    store.save("a", "fresh-0003", "functional")
    df = pd.read_csv(store.path, dtype=str, keep_default_na=False)
    assert list(df["annotator_a"]) == ["security", "discard", "functional"]
    assert list(df["annotator_b"]) == ["", "usability", ""]
    assert store.item("a") == {"complete": True, "total": 3, "done": 3}


def test_save_rejects_unknown_labels_rows_and_annotators(store):
    with pytest.raises(ValueError):
        store.save("a", "fresh-0002", "maintainability")
    with pytest.raises(ValueError):
        store.save("a", "fresh-9999", "functional")
    with pytest.raises(ValueError, match="for annotator 'b'"):
        store.save("b", "fresh-0003", "functional")  # assigned to C
    with pytest.raises(KeyError):
        store.save("z", "fresh-0002", "functional")
