import pandas as pd
import pytest
from reqbench.fresh import collect
from reqbench.fresh.collect import (
    Document,
    NearDuplicateIndex,
    build_document,
    document_exclusion,
    repo_exclusion,
    select,
    to_tables,
)
from reqbench.fresh.extract import Candidate, extract_candidates, is_candidate, split_sentences
from reqbench.fresh.github import is_document_path, select_files

MARKDOWN = """# 3. Functional requirements

<!-- The system shall not be extracted from a comment. -->
The key words "MUST" and "SHALL" in this document are to be interpreted as in RFC 2119.

**FR-01:** The system shall allow a registered user to reset the password by e-mail.
The system shall log every failed login, e.g. a wrong password. It must keep logs for 90 days.

```python
# The system shall never be read from code blocks either.
```

| ID | Requirement |
|----|-------------|
| NFR-2 | Pages must load in < 2 seconds for 95% of requests. |

- [ ] Description: Users must be able to export their data as CSV files.
- The app shall work.
- Shall the system support more than one language at the same time?
- El sistema debe permitir exportar los datos del usuario en formato CSV.
- 1.2.3 The [System Name] shall support single sign-on through the provider.
"""


def test_extract_keeps_requirements_verbatim_and_drops_markup():
    texts = [c.text for c in extract_candidates("docs/SRS.md", MARKDOWN)]
    assert texts == [
        "The system shall allow a registered user to reset the password by e-mail.",
        "The system shall log every failed login, e.g. a wrong password.",
        "It must keep logs for 90 days.",
        "Pages must load in < 2 seconds for 95% of requests.",
        "Users must be able to export their data as CSV files.",
    ]


def test_extract_records_line_of_the_block():
    candidates = extract_candidates("SRS.md", MARKDOWN)
    # A paragraph is one block: its sentences point to the paragraph's first line.
    assert candidates[0].location == "SRS.md#L6"
    assert candidates[1].line == candidates[2].line == 6
    assert candidates[3].line == 15


def test_extract_skips_exact_repeats():
    text = "The system shall export reports as PDF.\n\nThe system shall export reports as PDF."
    assert len(extract_candidates("a.md", text)) == 1


def test_candidate_rejects_fragments_and_lead_ins():
    assert is_candidate("The portal shall show the remaining credit of each student.")
    assert not is_candidate("Each functional requirement MUST follow this template:")
    assert not is_candidate("Must link to all sections in the file.")


def test_document_paths_skip_copies_fixtures_and_agent_files():
    assert is_document_path("docs/SRS.md", 1000)
    assert is_document_path("docs/requirements/functional.md", 1000)
    assert not is_document_path("docs/SRS.pdf", 1000)
    assert not is_document_path("trunk/3rdparty/srs-docs/blog/post-spec.md", 1000)
    assert not is_document_path("frontend/__fixtures__/golden/srs-ieee830.md", 1000)
    assert not is_document_path("skills/requirements/SKILL.md", 1000)
    assert not is_document_path("docs/requirements/service-template/nfr.md", 1000)
    assert not is_document_path(".agents/specs/requirements.md", 1000)
    assert not is_document_path("docs/SRS.md", 10_000_000)


def test_root_readme_only_when_no_other_document():
    blob = {"type": "blob", "sha": "s", "size": 10}
    with_srs = [blob | {"path": "README.md"}, blob | {"path": "docs/SRS.md"}]
    assert [f["path"] for f in select_files(with_srs)] == ["docs/SRS.md"]
    assert [f["path"] for f in select_files([blob | {"path": "README.md"}])] == ["README.md"]


def test_split_sentences_protects_abbreviations():
    assert split_sentences("Use a format, i.e. CSV. Then stop. Done") == [
        "Use a format, i.e. CSV.", "Then stop.", "Done"
    ]


def test_near_duplicate_index_matches_light_edits_only():
    index = NearDuplicateIndex(["The product shall refresh the display every 60 seconds."])
    assert index.match("The system shall refresh the display every 60 seconds.", 0.85)
    assert index.match("The system shall encrypt every stored password.", 0.85) is None


def _repo(name="alice/clinic-srs", **overrides):
    repo = {
        "full_name": name, "owner": name.split("/")[0], "description": "Clinic SRS",
        "created_at": "2025-03-01T00:00:00Z", "license": "MIT", "fork": False,
        "files": [{"path": "SRS.md", "sha": "x", "size": 10}], "head_sha": "abc",
    }  # fmt: skip
    return repo | overrides


def test_repo_exclusion_reasons():
    assert repo_exclusion(_repo()) is None
    assert repo_exclusion(_repo(fork=True)) == "fork"
    assert repo_exclusion(_repo(license="GPL-3.0")) == "license"
    assert repo_exclusion(_repo(created_at="2024-12-31T23:00:00Z")) == "created_before_2025"
    assert repo_exclusion(_repo("bob/srs-generator")) == "tool_or_template"
    assert repo_exclusion(_repo(description="An SRS template for students")) == "tool_or_template"
    assert repo_exclusion(_repo("carol/system-requirements-example")) == "example_repo"
    assert repo_exclusion(_repo(files=[])) == "no_document_files"


def test_build_document_drops_reference_near_duplicates_and_flags_templates():
    template = "The {} shall show the status of order number {} on the screen."
    refs = NearDuplicateIndex([template.format("product", i) for i in range(3)])
    lines = [template.format("system", i) for i in range(3)]
    lines += [f"The clinic app shall send reminder number {i} by SMS." for i in range(10)]
    doc = build_document(_repo(), {"SRS.md": "\n\n".join(lines)}, refs)
    assert len(doc.reference_hits) == 3 and len(doc.candidates) == 10
    assert document_exclusion(doc) == "template_copy"


def test_declared_ai_generated_document_is_excluded():
    text = "This SRS was generated by ChatGPT.\n\n" + "\n\n".join(
        f"The kiosk shall print ticket {i} for the visitor." for i in range(12)
    )
    doc = build_document(_repo(), {"SRS.md": text}, NearDuplicateIndex())
    assert document_exclusion(doc) == "declared_ai_generated"


def test_kiro_style_generated_spec_is_excluded():
    lines = [f"THE Docs_System SHALL document migration step {i} in full." for i in range(8)]
    lines += [f"The kiosk shall print ticket {i} for the visitor." for i in range(4)]
    doc = build_document(_repo(), {"SRS.md": "\n\n".join(lines)}, NearDuplicateIndex())
    assert document_exclusion(doc) == "generated_spec_style"


def _doc(name, n, words):
    # Each document gets its own vocabulary so that documents are not near-duplicates.
    texts = [f"{words[0]} {i} must {words[1]} the {words[2]} of {words[3]}." for i in range(n)]
    return Document(_repo(name), [Candidate("SRS.md", i + 1, t) for i, t in enumerate(texts)])


def _documents():
    vocab = [
        ("Invoice", "reconcile", "currency totals", "overdue billing accounts"),
        ("Gate", "photograph", "licence plates", "vehicles entering parking"),
        ("Catalogue", "index", "borrowed volumes", "branch library members"),
        ("Sensor", "irrigate", "flower beds", "greenhouse garden zones"),
        ("Booking", "confirm", "room upgrades", "returning hotel guests"),
        ("Exhibit", "describe", "painting provenance", "temporary museum shows"),
        ("Timetable", "announce", "platform changes", "delayed transit lines"),
        ("Crane", "schedule", "container unloading", "berthed harbor ships"),
    ]
    return [_doc(f"owner{i}/{w[0].lower()}-srs", 30, w) for i, w in enumerate(vocab)]


def _published(doc, chosen):
    return "2025-06-01"


def test_select_is_deterministic_and_respects_caps():
    first = select(_documents(), _published, seed=1, target=50, per_doc=12)
    second = select(_documents(), _published, seed=1, target=50, per_doc=12)
    assert [(d.name, [c.text for c in ch]) for d, ch, _ in first.documents] == [
        (d.name, [c.text for c in ch]) for d, ch, _ in second.documents
    ]
    assert first.n_requirements == 50
    assert max(len(ch) for _, ch, _ in first.documents) <= 12
    other = select(_documents(), _published, seed=2, target=50, per_doc=12)
    assert [d.name for d, _, _ in other.documents] != [d.name for d, _, _ in first.documents]


def test_select_one_document_per_owner_and_only_recent_files():
    docs = _documents()
    docs.append(_doc("owner0/second-srs", 30, ("Runway", "light", "taxiway signs", "airports")))
    dates = {"owner1/gate-srs": "2023-01-01"}
    result = select(docs, lambda d, _: dates.get(d.name, "2025-06-01"), target=500, per_doc=10)
    names = [d.name for d, _, _ in result.documents]
    assert "owner1/gate-srs" not in names
    assert len({n.split("/")[0] for n in names}) == len(names)
    reasons = {s["repo"]: s["reason"] for s in result.skipped}
    assert reasons["owner1/gate-srs"] == "file_before_2025"
    assert "same_owner" in reasons.values()


def test_select_skips_mirrors_of_chosen_documents():
    words = ("Invoice", "reconcile", "currency totals", "overdue accounts")
    docs = [_doc(f"o{i}/copy-{i}", 15, words) for i in range(6)]
    with pytest.raises(RuntimeError, match="Only 1 documents"):
        select(docs, _published, target=100, per_doc=40)
    original = _documents()[0]
    copy = _doc("mirror/copy", 30, ("Invoice", "reconcile", "currency totals", "overdue accounts"))
    result = select(_documents() + [copy], _published, target=500, per_doc=40)
    chosen = {d.name for d, _, _ in result.documents} & {original.name, copy.name}
    mirrors = {s["repo"] for s in result.skipped if s["reason"] == "mirror_of_selected"}
    assert len(chosen) == 1 and mirrors == {original.name, copy.name} - chosen


def test_select_rejects_per_doc_above_cap():
    with pytest.raises(ValueError):
        select(_documents(), _published, per_doc=41)


def test_to_tables_assigns_ids_and_empty_annotations():
    result = select(_documents(), _published, seed=3, target=50, per_doc=10)
    documents, requirements = to_tables(result, "2026-10-10", seed=3)
    assert list(documents["doc_id"]) == [f"fresh-doc-{i:02d}" for i in range(1, 6)]
    assert requirements["req_id"].iloc[0] == "fresh-0001" and len(requirements) == 50
    assert (requirements[collect.ANNOTATION_COLUMNS] == "").all().all()
    # Rows are shuffled: consecutive rows do not all come from the same document.
    assert requirements["doc_id"].ne(requirements["doc_id"].shift()).sum() > 4
    assert documents["max_requirements_taken"].sum() == 50


def test_write_refuses_to_overwrite_annotations(tmp_path, monkeypatch):
    (tmp_path / "fresh").mkdir()
    pd.DataFrame(
        [{"req_id": "fresh-0001", "doc_id": "d", "location": "l", "text": "t",
          "annotator_a": "security", "annotator_b": "", "adjudicated": "",
          "adjudication_note": ""}]
    ).to_csv(tmp_path / "fresh" / "requirements.csv", index=False)  # fmt: skip
    monkeypatch.setattr(collect, "data_dir", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="already has annotations"):
        collect.write_collection(None, seed=1, target=1, per_doc=1)


def test_only_rate_limited_or_unresolved_repos_are_retried():
    from reqbench.fresh.github import _pending

    assert _pending({"full_name": "a/b"})
    assert _pending({"error": "gh api repos/a/b: gh: API rate limit exceeded for user"})
    assert not _pending({"error": "gh api repos/a/b: HTTP 409 Git Repository is empty"})
    assert not _pending({"head_sha": "abc", "files": []})
