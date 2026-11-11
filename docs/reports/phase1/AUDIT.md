# Phase 1 audit (2026-11-11)

An independent review of phase 1 was carried out after the one-time locked-test evaluation
(freeze `1ff9e151…`, `EVAL_LOG.jsonl`). The owner accepted its conclusions. This file records
each finding, whether it changes the preregistered verdict (**INCONCLUSIVE**), and the
reproduced numbers. The support analyses are regenerated at no cost by `uv run reqbench audit`
(`bench/src/reqbench/eval/audit_phase1.py` → `runs/phase1/audit/audit.json`). Anything that
reads test predictions is **exploratory and post hoc**: it is computed after the one-time
evaluation and decides nothing.

**Confirmed:** no test or label leakage into development, calibration or thresholds; the
test was evaluated once; the formal verdict is correct under any reasonable reading of the
rule; 75 tests pass; ruff is clean; `reproduce --from-cache` regenerates identical tables at
no cost.

Fixes to frozen code are **deferred until the fresh set has been evaluated** with the same
freeze (any change to a frozen file would make that impossible); see "Deferred fixes" below.

## Findings

| Id | Finding | Changes the verdict? |
|---|---|---|
| C1 | J3's calibrator and threshold were fitted on a different model from the one evaluated | No |
| C2 | The tie-break ignored Haiku with one call (B5x) | No (B5x was exploratory); ADR 0001 is superseded |
| I1 | Criterion 3 for J3 used J1's stability and adversarial measures; the adversarial gap is not significant | No (ECE fails anyway) |
| I2 | ECE ≤ 0.05 with 15 bins and n = 171 is at the noise floor | No |
| I3 | Non-inferiority is fragile and rests on the best of four variants picked on test | No |
| I4 | The operational reading (deviation 3) was fixed after seeing dev; several choices favour Jev | No |
| I5 | Weak public audit trail (freeze commit pushed after the test) | No |
| I6 | The report's sample sizes were for ~50 % power | No |
| I7 | Several report sentences overstate the evidence | No |
| I8 | The fresh set can hardly detect a δ = 0.03 difference | No (affects ADR condition 4) |
| M | Minor: reproduce overwrites results.json; ECE depends on row order with ties; FREEZE.md "identity never won"; missing metrics in the report | No |

### C1. J3's calibrator and threshold do not match the evaluated model

The CV models that produced J3's out-of-fold predictions chose C = 0.1 / 0.1 / 1.0 / 0.1 / 1.0
(fold 2 also `class_weight` balanced); the final model trained on all of dev chose C = 30
(`frozen/phase1/final_models.json`; `eval/final_models.py:82`; `eval/calibrate_dev.py:82-85`).
The temperature T = 0.72 and the threshold 0.954 were fitted on the former and applied to the
latter. Reproduced on dev only (`audit.json` → `c1_dev`):

| J3 out-of-fold on dev | ECE raw | ECE with frozen T = 0.72 | Mean confidence (T) | Frozen threshold: coverage (precision) |
|---|---|---|---|---|
| C chosen per fold (what was frozen) | 0.109 | 0.051 | 0.848 | 27.7 % (0.942) |
| C = 0.1 fixed | 0.188 | 0.078 | 0.800 | 3.9 % (1.000) |
| C = 1 fixed | 0.044 | 0.070 | 0.912 | 62.5 % (0.941) |
| **C = 30 fixed (the final model)** | **0.052** | **0.077** | 0.936 | **74.8 % (0.920)** |

(The freeze itself reported 33 % at 0.952, on nested-CV calibrated probabilities.) With
C = 30 the model is already fairly calibrated and T < 1 sharpens it further. On the test the
same happened: mean calibrated confidence 0.924 against accuracy 0.819; the threshold decided
69 % at precision 0.932. **The report's explanation ("calibration did not transfer to new
projects") is wrong for J3**: the cause is the mismatch. The same design flaw (calibrating on
out-of-fold predictions of per-fold-tuned models, then applying to a final model with other
hyperparameters) affects B2 (C 3–30, `class_weight` varies), B3 (C 0.1–30 → 3) and B4
(epochs 7–15 per fold → 9–12); not quantified.

### C2. The tie-break ignored Haiku with one call (B5x)

B5x (one Haiku sample, hard label): macro-F1 0.788 on test; $0.127 per 1,000 ($0.064 with the
Batch API) against $0.27 for J3. J3 − B5x = +0.008 [−0.032, +0.045] (post hoc): **J3 is not
non-inferior to B5x.** This was visible on dev (B5x 0.851 vs B5 0.858). B5x was declared
exploratory before the test, so the verdict stands, but ADR 0001's sentence "the cheapest of
the non-inferior options" does not hold. ADR 0002 will replace it.

### I1. Criterion 3 for "J3" used J1's measures

`eval/heldout.py:597` and `:605` take run-to-run stability and the adversarial flip rate of J1
(J3 has no measurement of its own). Flips on the 60 adversarial variants (`adversarial_fisher`):
J1 3, J4 2, B5 1, B5x 0. Fisher two-sided: J1 vs B5 p = 0.62; J1 vs B5x p = 0.24. B5 votes
over 5 samples while J1 uses one. **"Jev is more easily swayed by injection" is not
demonstrated** with 60 variants.

### I2. The ECE threshold is at the noise floor

A perfectly calibrated system with J3's raw test confidences (outcomes drawn as Bernoulli of
the confidence, 10,000 simulations, `ece_noise_floor`) has mean ECE 0.070 (95th percentile
0.093) and falls below 0.05 only 6 % of the time. J3's raw ECE (0.078) is indistinguishable
from perfect calibration (p = 0.26). The calibrated ECE (0.114) is real miscalibration
(p < 0.0001 under the same simulation with its own confidences), caused by C1.

### I3. Fragile non-inferiority

Only J3 meets it (lower bound −0.026); J2 −0.034, J1 −0.057, J4 fails. J3 and B5 differ in
correctness on 9 of 171 requirements (6 J3 right / B5 wrong, 3 the reverse). The best Jev was
picked on test, although J3 was also the best on dev (0.836).

### I4. Operational reading fixed after seeing dev

Deviation 3 was written after the dev results, and several of its choices favour Jev: the
maximum over four Jev variants; B5x outside B\*; 2(a) only against local systems and with
unadjusted CIs (against B4 seed 4 the Holm p is 0.159). Excluding Opus from B\* was in the
original plan.

### I5. Weak public audit trail

The freeze commit `f9db031` was pushed after the test, together with the results; the commit
of deviation 2 falls inside the Jev dev run. The local inference cache supports the
substance (no test text sent before 20:06:18 UTC on 2026-11-11; only 20 calls to
`jev-1.13.0`; no fresh-set text sent to any model), but it is not public.

### I6. Sample sizes

The report's ≈ 1,200 and ≈ 870 correspond to ~50 % power. For 80 % power: about 2,500 test
requirements to separate J3 from B5 at the observed gap, and about 1,700–1,800 to show J1
non-inferior (`test_post_hoc.sample_size_needed_by_power`; 1,227 / 2,506 and 842 / 1,720 with
normal quantiles).

### I7. Overstatements in the report

- "Opus is clearly better than everything": against J3, Holm p 0.12, McNemar 0.077.
- "The thresholds gave 93–94 % instead of 95 %": about 110 of 118 for J3; the interval
  includes 0.95, and the figure is contaminated by C1.
- "Haiku cannot support automatic decisions": only vote frequency with k = 5 was tried.
- The project-level bootstrap with 6 projects is not a reliable interval.

### I8. Fresh set

Pooled κ 0.31, about 150 usable sentences, gold dominated by annotator A. ADR 0001's
condition 4 ("worse than B5 beyond δ = 0.03") is practically undetectable there.

### Minor

- `reproduce` overwrites `runs/phase1/test/results.json`; its hash no longer matches
  `EVAL_LOG.jsonl` (now `97a96df3…`, logged `8814c77a…`; only the local CPU latency differs).
- Top-label ECE depends on row order when confidences tie: on dev, J1's ECE ranges from 0.078
  to 0.081 over 1,000 random orders (B5 unaffected) (`ece_order_dependence_dev`).
- FREEZE.md says identity never won; for B3 it won in 4 of 5 outer folds (it lost only the
  final selection).
- Log loss, Brier, class-wise ECE and confusion matrices are in `results.json` but not in
  the report.

## Deferred fixes (after the fresh-set evaluation, with a new, documented freeze)

1. `reproduce` must not overwrite the logged `results.json` (write to a separate file).
2. Order-independent ECE with tied confidences (keep tied blocks whole, as AURC does).
3. For future freezes: fix hyperparameters before calibrating, or calibrate on predictions of
   the same model that is evaluated.
4. Measure stability and adversarial flips on the system being decided (not a proxy).
5. Replace the absolute ECE ≤ 0.05 threshold with a criterion relative to the noise floor.
