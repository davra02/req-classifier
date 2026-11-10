# Jev question-set versions (phase 1, dev only)

The preregistration allows at most 15 versions of the Jev question set; each one is listed
here with its hashes and its dev results. Dev = 437 requirements (examples_v1 excluded),
project-grouped 5-fold CV for J3; paired bootstrap with 10,000 resamples. Outputs:
`runs/phase1/dev/` (`jev_answers.csv`, `oof_J*.csv`, `summary_J*.json`, `jev_report.json`),
regenerated from the inference cache with `uv run reqbench jev`.

## v1: `jev_type_v1` + `examples_v1` (2026-11-10)

- Model `jev-1.14.0` (every one of 3,059 answers reported it). Questions: `req_type` (Choice,
  6 leaves), `nfr_subtype` (Choice, 5 NFR leaves), `is_functional`, `is_requirement`,
  `addresses_ai` (Noul), all in one request per requirement.
- Question-set hashes: rotation 0 `554cbc94…`, rotation 1 `505c1700…`, rotation 2 `afe9dc44…`.
- Calls: 437 × 3 rotations + 437 × 4 repeats of rotation 0 = 3,059 (20 from the smoke test);
  cost $0.274; mean 2,132 input tokens; latency p50 163 ms, p95 207 ms at concurrency 8.

| System | Macro-F1 [95 % CI] | Level-1 F1 | Level-2 F1 | ECE (raw) | AURC | Cov. @ 95 % |
|---|---|---|---|---|---|---|
| B2 TF-IDF + LR | 0.577 [0.522, 0.627] | 0.695 | 0.641 | 0.097 | 0.256 | 0.062 |
| B3 bge-small + LR | 0.617 [0.565, 0.665] | 0.807 | 0.636 | 0.218 | 0.246 | 0.011 |
| J1 Choice | 0.830 [0.787, 0.867] | 0.900 | 0.837 | 0.078 | 0.043 | 0.652 |
| J2 rotation average | 0.829 [0.786, 0.865] | 0.891 | 0.844 | 0.066 | 0.045 | 0.666 |
| J3 LR on Jev outputs | 0.836 [0.794, 0.872] | 0.886 | 0.852 | 0.109 | 0.083 | 0.414 |
| J4 hierarchical | 0.818 [0.775, 0.855] | 0.858 | 0.818 | 0.069 | 0.080 | 0.291 |
| J4x hierarchical, rot. avg (exploratory) | 0.815 [0.772, 0.853] | 0.856 | — | 0.067 | 0.081 | 0.288 |

- Paired differences in macro-F1 vs B3: J1 +0.214 [+0.158, +0.269]; J2 +0.212; J3 +0.219;
  J4 +0.201; Holm-adjusted p < 0.001 for all four (no resample at or below 0).
- Among Jev variants (vs J1): J2 −0.001 [−0.015, +0.012]; J3 +0.006 [−0.015, +0.029];
  J4 −0.013 [−0.040, +0.016], with worse level-1 F1 (−0.042) and AURC (+0.037).
- Stability (5 samples of rotation 0): J1 label changes in 0.69 % of repeats vs sample 0
  (0.9 % of requirements change at least once); J4 0.97 %; mean SD of `is_functional` 0.008.
- Option order (3 rotations): `req_type` label changes in 2.1 % of rotation pairs (3.0 % of
  requirements); first-position lift ≈ 0 (−0.001 to +0.001) for the three leading options.
- Raw log loss is inflated by probabilities rounded to two decimals (exact zeros): J1 1.156.
- Caveat: PROMISE is public since 2007 and may be in Jev's training data; the fresh set is
  the check for that (criterion 2c).
