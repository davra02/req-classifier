# Phase 1 freeze (before the locked test)

Written on 2026-11-11, after every dev result and **before any system was run on the locked
test or the fresh set**. The machine-readable freeze is `frozen/phase1/FREEZE.json`: SHA-256
of every file that determines a result, plus versions, prompt hashes, seeds and the decision
constants. Its overall hash is the `freeze_hash`. `uv run reqbench heldout <set> --run`
refuses to start if any of those files differs from the freeze or is not committed, logs the
evaluation in `EVAL_LOG.jsonl` before reading a label, and refuses any other freeze for a set
that has already been evaluated. Deviation 3 in `docs/PREREGISTRATION.md` records the
decisions taken at this point.

## What is frozen

| Item | Value |
|---|---|
| Data | `promise_v1` (622 requirements), split `promise_v1_split_v1` (test = projects 2, 4, 7, 9, 13, 15; 171 requirements), `examples_v1` |
| Code | `packages/reqcore/**`, `bench/src/reqbench/{data,models}/*`, `eval/{metrics,bootstrap,cv,calibration,calibrate_dev,final_models,adversarial,haiku_dev,heldout}.py`, `uv.lock` |
| Jev | `jev-1.14.0`; question set `jev_type_v1`, hashes r0 `554cbc94…`, r1 `505c1700…`, r2 `afe9dc44…`; 7 calls per requirement (3 rotations + 4 repeats) |
| B5 Haiku | `claude-haiku-5-5`, prompt `llm_type_v1`, thinking disabled, effort low, `max_tokens` 256, k = 5; hashes r0 `54ab42b2…`, r1 `4f5d24e8…`, r2 `6a2d00c5…` |
| B6 Opus | `claude-opus-5-5`, same prompt, effort low (thinking cannot be disabled), `max_tokens` 1024, k = 1, Batch API; hash `5f710492…` |
| Local models | B0–B3 and J3 fitted on all dev (`frozen/phase1/final_models.json`: hyperparameters and file hashes); B4 `deberta-v3-small@a36c7390…`, 5 seeds (20261010–20261014), trained on all dev, inference on CPU in fp32; `bge-small-en-v1.5@5c38ec7c…` |
| Calibrators | `frozen/phase1/calibrators.json` (chosen by pooled log loss over the 5 project-grouped dev folds; J4 per question) |
| Review thresholds | `frozen/phase1/thresholds.json` (precision ≥ 0.95 on dev nested-CV calibrated probabilities) |
| Adversarial suite | `questions/adversarial_v1.yaml` (60 variants of 20 dev requirements) |
| Decision | δ = 0.03; ECE ≤ 0.05; flip ≤ 3 %; order ≤ 5 %; p95 ≤ 1 s; ≤ $1 per 1,000; adversarial ≤ B5; bootstrap 10,000 (seed 20261010) |

## Calibration on dev (nested CV; examples excluded, n = 437)

The calibrator is chosen by log loss; the ECE below is measured on dev probabilities whose
calibrator (and choice of calibrator) never saw their fold. The bootstrap ECE interval is
skewed upwards (ECE is biased upwards in resamples with repeated rows), so the point estimate
can sit at or below its lower bound.

| System | Calibrator | ECE raw → calibrated [95 % CI] | Log loss | Threshold | Dev coverage at 95 % |
|---|---|---|---|---|---|
| J1 Jev Choice | vector scaling | 0.078 → **0.044** [0.040, 0.087] | 1.156 → 0.462 | 0.882 | 0.664 |
| J2 rotation average | vector scaling | 0.066 → 0.051 [0.043, 0.093] | 0.965 → 0.459 | 0.892 | 0.654 |
| J3 LR on Jev | temperature (T = 0.72) | 0.109 → 0.051 [0.047, 0.100] | 0.583 → 0.543 | 0.954 | 0.332 |
| J4 hierarchical | `is_functional` Platt + `nfr_subtype` vector | 0.067 → **0.035** [0.039, 0.083] | 1.022 → 0.486 | 0.904 | 0.519 |
| B5 Haiku k = 5 | vector scaling | 0.112 → 0.065 [0.051, 0.099] | 3.967 → 0.570 | none | 0 |
| B4 seeds 1–5 | temperature (s1), vector (s2–s5) | 0.145–0.234 → 0.062–0.078 | 1.02–1.27 → 0.84–0.88 | 0.92–0.99 | 0.01–0.11 |
| B3 bge + LR | isotonic | 0.218 → 0.244 | 1.140 → 1.235 | 1.000 | 0.009 |
| B2 TF-IDF + LR | vector scaling | 0.097 → 0.097 | 1.200 → 1.137 | 0.937 | 0.048 |

Identity never won: for Jev and B5 the raw log loss is dominated by exact zeros (Jev rounds
to two decimals; B5's frequencies are multiples of 0.2). B5 cannot gain coverage: 97 % of
its answers form one unanimous block, right 89 % of the time, and calibration keeps it one
block.

## Freeze hash

`1ff9e151d1729968d1e9f155eef463d22dc9fb3e5e1caadcdb7b4da6ab60df19` (49 files; written by
`uv run reqbench freeze` on 2026-11-11, see `frozen/phase1/FREEZE.json`).

Checked before freezing: 75 tests pass; the whole held-out pipeline (`heldout.evaluate`) was
run end to end on dev rows from the cache, without API keys (B6 stubbed with Haiku, no
adversarial suite), to make sure it cannot crash after the paid test calls. No test text
reached any system and no test label was read.

## Errata (added 2026-11-11, after the locked-test evaluation and an independent audit)

The text above is kept as written at the freeze. Two statements in it are inaccurate (details
in `AUDIT.md`):

- **Calibrators and thresholds of trained systems do not match the evaluated models (AUDIT
  C1).** They were fitted on out-of-fold predictions of models whose hyperparameters were
  tuned per fold (J3: C = 0.1–1; B2, B3: C varies; B4: epochs vary), and frozen for final
  models retrained on all of dev with other hyperparameters (J3: C = 30). For J3 this makes
  the frozen temperature (T = 0.72) and threshold (0.954) wrong for the final model: on dev,
  with C = 30 out of fold, ECE 0.052 → 0.077 with T and the threshold decides 75 % at precision
  0.92, not 33 % at 0.95 as in the table above. The figures in the table are correct for the
  per-fold models they were computed on.
- **"Identity never won"** refers to the final selection only; for B3 identity won in 4 of 5
  outer folds.

Nothing frozen was changed: the fresh set will be evaluated with this same freeze, and the
fix (calibrating predictions of the model that is evaluated) is deferred to the next freeze.
