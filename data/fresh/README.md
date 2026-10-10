# Fresh evaluation set (plan §6.9)

About 150 requirements from public documents published **from 2025 onwards**, used only for
evaluation (never for training or prompt iteration). It answers two questions: do results hold
outside PROMISE, and were PROMISE results inflated by models having seen it?

## Collection rules

1. **Sources:** public requirement documents (SRS, specifications, technical requirements in
   tenders or open-source projects) dated 2025 or later, under a license that allows reuse
   (MIT, Apache-2.0, CC BY, CC0, or an explicit public-sector reuse license). Search ideas:
   GitHub repositories created since 2025 with `SRS`, `requirements specification` or
   `software requirements` in the README or file names (filter by license); public tender
   technical specifications in English.
2. **Spread:** at least 5 documents, **at most 40 requirements per document**.
3. **Extraction:** copy sentences that are requirements (usually contain shall/must/will or a
   numbered requirement format) **verbatim**, fixing whitespace only. Do not write or rephrase
   requirements yourself: knowing the taxonomy would bias the wording.
4. **Provenance:** one row per document in `documents.csv` (URL, publication date, license,
   retrieval date, version or commit). If a license does not allow copying text, keep only
   `location` (section/line) in `requirements.csv` and leave `text` empty; it will be fetched
   at evaluation time.

## Annotation

- Two annotators label every requirement **independently**, without seeing each other's labels
  or any model output, using the leaf definitions in `questions/taxonomy_v1.yaml`:
  functional, performance, security, usability, operational_constraint, other.
- Columns `annotator_a` / `annotator_b` hold the leaf; `adjudicated` is filled after discussing
  disagreements, with the reason in `adjudication_note`.
- Cohen's κ (level 1 and leaves) is computed from these columns before adjudication.
- Freeze the file (commit) **before** any system is evaluated on it.

## Files

- `documents.csv`: `doc_id,title,url,published_date,license,retrieved_at,version_or_commit,max_requirements_taken,notes`
- `requirements.csv`: `req_id,doc_id,location,text,annotator_a,annotator_b,adjudicated,adjudication_note`
  (`req_id` format: `fresh-0001`).
