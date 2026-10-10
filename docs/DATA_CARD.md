# Data card: `promise_v1` + `promise_v1_split_v1`

Regenerate everything with `uv run reqbench data all`. Every number below comes from
`data/manifests/promise_v1.json` and `data/manifests/promise_v1_split_v1.json`.

## Source

| Field | Value |
|---|---|
| Dataset | PROMISE NFR (`nfr/nfr.arff` inside `nfr.tar`) |
| Origin | Cleland-Huang et al., OpenScience tera-PROMISE; Zenodo [10.5281/zenodo.268542](https://doi.org/10.5281/zenodo.268542) |
| License | CC BY 4.0 (attribution required; see `DATA_LICENSES.md`) |
| SHA-256 of `nfr.tar` | `28f9328ce72f1c54f09b8dd45573eb811b7a78959b5bf290561cbad1d5adb0b5` |
| Content | 625 English requirements from 15 student software projects, one label each |
| Encoding | Windows-1252; normalized to NFKC with control and zero-width characters removed |

The raw file and the processed text are **not** committed (`data/raw/`, `data/processed/` are
git-ignored). Split files contain IDs, text hashes and project IDs only.

## Cleaning

| Step | Result |
|---|---|
| Rows parsed | 625 |
| Exact duplicates removed (lowercase alphanumeric key; first occurrence kept) | 3 |
| Rows after cleaning | **622** |

Removed rows (the same three NICE removed):

| Removed | Kept | Labels (removed / kept) | Note |
|---|---|---|---|
| promise-0107 | promise-0106 | F / F | identical text |
| promise-0459 | promise-0425 | **F / SE** | conflicting labels: "Only registered customers can purchase streaming movies." |
| promise-0504 | promise-0499 | PE / PE | differs only in spacing/punctuation |

Cross-project near-duplicates (similarity ≥ 0.90), kept in the data but handled in the split:
`promise-0035` ~ `promise-0074` (0.906) and `promise-0042` ~ `promise-0401` (0.918). Within-project
near-duplicates (16 pairs, e.g. "position ships horizontally / vertically") are kept: they are
distinct requirements and project-grouped splits keep each pair on one side.

## Labels

Taxonomy `taxonomy_v1` (`questions/taxonomy_v1.yaml`), two levels:

| Leaf | PROMISE classes | n | Level 1 |
|---|---|---|---|
| functional | F | 253 | functional |
| performance | PE (53) + SC (21) | 74 | non_functional |
| security | SE | 66 | non_functional |
| usability | US (67) + LF (38) | 105 | non_functional |
| operational_constraint | O (62) + L (13) + PO (1) | 76 | non_functional |
| other | A (21) + FT (10) + MN (17) | 48 | non_functional |

Known label issues (not corrected; test labels are never edited):
- 13 % of PROMISE requirements have both a functional and a quality aspect (Dalpiaz et al. 2019
  relabeling); single-label gold therefore caps achievable accuracy below 1.
- One exact duplicate carried conflicting labels (above).
- Projects 11–15 contain no functional requirements; a model could learn project style. The
  project-grouped split neutralizes this.

## Splits (`promise_v1_split_v1`, seed 20261010)

Test = whole projects, drawn by the script among the 2,043 project combinations holding 27–33 %
of requirements with ≥ 10 examples per leaf (rank 10 of the 50 closest to the overall
distribution). Nobody chose the projects by hand.

| Split | Projects | n |
|---|---|---|
| test (locked) | 2, 4, 7, 9, 13, 15 | 171 |
| dev | 1, 3, 5, 6, 8, 10, 11, 12, 14 | 449 |
| excluded | — | 2 (`promise-0035`, `promise-0042`: test-side members of cross-project near-duplicate pairs) |

| Leaf | dev | test |
|---|---|---|
| functional | 186 | 67 |
| usability | 76 | 28 |
| performance | 54 | 20 |
| operational_constraint | 52 | 24 |
| security | 47 | 19 |
| other | 34 | 13 |

Dev folds (StratifiedGroupKFold, 5 folds, grouped by project): fold 0 = projects 3, 14 (95);
fold 1 = 10, 12 (74); fold 2 = 5, 11 (86); fold 3 = 6 (74); fold 4 = 1, 8 (120).

Split content hash: `224a6fa374235c45dac6f0d48d83233dbadfb090022f44e5e8bdac8e3ed651ba`.

## Few-shot examples (`questions/examples_v1.yaml`)

Two examples per leaf, drawn from dev with the same seed among requirements of ≤ 25 words that
are not part of any duplicate or near-duplicate pair. They are shown identically to Jev and to
the LLM and are excluded from dev metrics. They are a random draw and were not hand-picked.

## Intended use and limitations

Benchmarking requirement-type classifiers. Not representative of industrial requirements: the
projects are student work, about 20 years old, English only, and the dataset is widely
published, so pre-trained models may have seen it (see the contamination checks in
`docs/PLAN.md` §6.8).
