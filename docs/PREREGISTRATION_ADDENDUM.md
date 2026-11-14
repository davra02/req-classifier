# Preregistration addendum: the fresh-set evaluation

Written on 2026-11-11, **before the fresh set is adjudicated or evaluated by any system**
(no fresh-set text has ever been sent to a model; `AUDIT.md`, I5). It complements
`docs/PREREGISTRATION.md`, which is frozen and is not edited. It fixes, in advance, how the
fresh set is evaluated and reported and which results reopen ADR 0002. Any later change to
this file is a deviation and is listed at the end with its date and reason.

## Context

The locked-test verdict is INCONCLUSIVE; ADR 0002 carries forward B5x (Claude Haiku 5.5, one
call) as the type classifier. The fresh set (`data/fresh/`, 170 sentences from public
requirement documents first committed in 2025 or later) checks two things: whether the
results hold outside PROMISE and whether PROMISE results were inflated by contamination. Its
human agreement is low: pooled Cohen's κ between annotator A and the second annotators (B, C,
D; 90 double-annotated sentences, before adjudication) is 0.31 [0.18, 0.43] on the 6 leaves
plus `discard`, and 0.29 on level 1. Annotator A's labels (before adjudication) contain only 4
performance and 9 usability requirements, so per-class F1 for those leaves, and therefore
macro-F1, will be very noisy.

## What is frozen

- **Systems, code, prompts, calibrators and thresholds:** the phase-1 freeze `1ff9e151…`,
  unchanged. `uv run reqbench heldout fresh --run` refuses any other freeze.
- **Gold:** the `adjudicated` column of `data/fresh/requirements.csv`. The 90 double-annotated
  rows are adjudicated by the owner (reason in `adjudication_note`); for the other rows it is
  annotator A's label. Rows whose gold is `discard` are dropped (systems have no `discard`
  option). Adjudication is done without seeing any system output.
- **Order of events:** adjudication → commit of `data/fresh/requirements.csv` (the fresh-set
  freeze) → **push** → dry run with cost estimate → owner's approval → one run of
  `heldout fresh --run`. The log entry records the SHA-256 of the requirements file.

## What is reported

1. **Every system as on the test**, with the same metrics (macro-F1 with 95 % bootstrap CI,
   level-1 F1, level-2 macro-F1, accuracy, per-class F1, confusion matrices, log loss, Brier,
   ECE, AURC, coverage at the frozen thresholds), cost and latency. **B5x (Haiku, one call) is
   reported as a full system**, with its own CI and paired comparisons, not as exploratory.
2. **Two gold sets:** (a) primary, all non-discard rows; (b) secondary, only the
   double-annotated rows (adjudicated gold), which have the most reliable labels.
3. **Human ceiling:** for each system, Cohen's κ against the gold on the 6 leaves (non-discard
   rows), shown next to the human κ (0.31 [0.18, 0.43]; level 1: 0.29). A system whose κ with
   the gold is inside or above the human interval is "at the human ceiling"; given that the
   gold is mostly annotator A, agreement with the gold partly measures agreement with A.
4. **PROMISE-to-fresh drop** for every system: macro-F1 on the locked test minus macro-F1 on
   the fresh set.
5. Criterion 2(c) of the preregistration (best Jev ≥ δ = 0.03 better than B\* = B5) is reported
   as defined there; it cannot change the test verdict, because criterion 3 already failed.

## What reopens ADR 0002 (decided by the owner: either condition)

Both are computed on the primary gold set, with 10,000 bootstrap resamples (seed 20261010).

- **R1, Jev clearly better on uncontaminated data:** macro-F1(J3) − macro-F1(B5x) ≥ **0.10**
  and the lower bound of its paired-bootstrap 95 % CI > 0.
- **R2, contamination signal:** the drop of B5x minus the mean drop of the five B4 seeds
  (drop = test macro-F1 − fresh macro-F1) ≥ **0.10**, and the lower bound of its 95 % CI > 0
  (test and fresh resampled independently; the B4 seeds averaged within each resample).

The 0.10 thresholds are set at what this set can detect: with about 150 sentences and this
human agreement, differences of 0.03 are invisible. "Reopen" means the decision is reviewed
with the new evidence and recorded in a new ADR; it does not switch systems automatically.
If neither condition holds, ADR 0002 stands, and the fresh-set results are reported as they
come out, whatever their direction.

These two quantities are computed after the run by a new, unfrozen module that only reads the
saved predictions (`runs/phase1/fresh/pred_*.csv`) and the test results; it is written and
committed **before** the run.

## Deviations from this addendum

None yet.
