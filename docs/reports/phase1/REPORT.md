# Phase 1 report: requirement-type classification (PROMISE, locked test)

**Question.** Does Jev (`jev-1.14.0`) classify requirement types well enough, cheaply enough
and reliably enough to be used instead of simpler or cheaper alternatives? The rule for the
answer was preregistered before any model was run (`docs/PREREGISTRATION.md`); everything was
then frozen (`FREEZE.md`, freeze `1ff9e151…`) and the locked test was evaluated **once** on
2026-11-11 (`EVAL_LOG.jsonl`: started 20:06 UTC, completed 20:19 UTC, commit `f9db031`).

**Short answer: inconclusive under the preregistered rule.** On the test, the best Jev variant
(J3, a logistic regression on Jev's outputs) is as accurate as the best alternative, Claude
Haiku 5.5 with 5 samples, and much better at knowing when it is unsure; it beats every local
model by a wide margin and is cheaper and 5× faster than Haiku. But it fails two of the
preregistered operational guardrails: its calibrated confidence is off on the test (ECE 0.114
> 0.05) and it is more easily swayed by prompt injection than Haiku (5 % of adversarial
variants flip its label, against 1.7 %). Criterion 2(c) (the fresh set) is still pending, but
it cannot turn this into a GO, because criterion 3 fails independently of it.

## What was compared

171 requirements from 6 whole PROMISE projects (2, 4, 7, 9, 13, 15) never used for
development; 6 leaf classes (functional, performance, security, usability,
operational_constraint, other). Systems: majority (B0), keyword rules (B1), TF-IDF + LR (B2),
bge-small embeddings + LR (B3), fine-tuned DeBERTa-v3-small with 5 seeds (B4), Claude Haiku 5.5
zero-shot with k = 5 samples (B5), Claude Opus 5.5 as a quality ceiling (B6, not a candidate);
Jev: flat Choice (J1), average of 3 option orders (J2), LR on all Jev outputs (J3) and
hierarchical (J4). Every probabilistic system also has a calibrated variant (`+ cal.`) and a
review threshold, both fitted on dev only. Jev and the LLMs saw the same definitions and
examples; neither prompts nor questions were iterated (version 1 of 15 for both).

## Decision (preregistered rule, applied by `reqbench.eval.heldout.decide`)

- **Best Jev** = J3 (macro-F1 0.796); **B\*** = B5 Haiku (0.781; B4's seed mean is 0.647).

| Criterion | Result | Met? |
|---|---|---|
| 1. Non-inferiority: lower CI bound of J3 − B5 > −0.03 | +0.015 [−0.026, +0.054] | **yes** |
| 2(a). ≥ 0.03 better (CI > 0) than every system costing no more (B0–B4, every seed) | +0.093 to +0.702, all CIs > 0 | **yes** |
| 2(b). Lower AURC than B\* (CI excludes 0) | 0.052 vs 0.179; Δ −0.127 [−0.170, −0.084] | **yes** |
| 2(c). ≥ 0.03 better than B\* on the fresh set | not evaluated yet | pending |
| 3. ECE of J3 + calibrator ≤ 0.05 | 0.114 (raw J3: 0.078) | **no** |
| 3. Run-to-run flip rate ≤ 3 % | 2.0 % | yes |
| 3. Option-order flip rate ≤ 5 % (or J2) | 4.7 % | yes (barely) |
| 3. p95 latency ≤ 1 s | 277 ms (3 calls per requirement) | yes |
| 3. Cost ≤ $1 per 1,000 | $0.27 | yes |
| 3. Adversarial flip rate ≤ B5's | 5.0 % vs 1.7 % | **no** |
| NO-GO conditions (below B\* beyond δ; below B2; instability) | none holds | — |

**Verdict: INCONCLUSIVE.** The preregistration says that then "the simpler, cheaper system
wins". It does not define *simpler*. The owner applied the tie-break in favour of **J3**, with
conditions: `docs/decisions/0001-phase1-type-decision.md` (the verdict itself stays
"inconclusive").

Two caveats on the decision itself:
- **"Best Jev" was picked on the test among four variants**, as preregistered; that favours Jev
  slightly. J2 is practically tied with J3 (0.795). The single-call J1, the cheapest and the
  one set as Jev "as is", would **not** pass criterion 1 on its own: J1 − B5 = −0.008
  [−0.057, +0.042].
- J4 *with its calibrator* scored 0.820 on the test, the highest Jev number: vector scaling
  re-weights the classes and changes some predictions. It is a calibrated variant, not one of
  the four preregistered systems, so it does not enter the rule; it is a lead for phase 2,
  not a result.
- With 171 requirements the intervals are wide. Separating J3 from B5 at their observed gap
  (+0.015) would take about 1,200 test requirements; showing non-inferiority for J1 at its
  gap, about 870.

## Main results (test, n = 171)

![Risk-coverage on test](figures/risk_coverage_test.png)

| System | Macro-F1 [95 % CI] | Level-1 F1 | Accuracy | AURC | Cov. @ 95 % (oracle) | Frozen threshold on test: coverage (precision) | $ / 1,000 | p95 latency |
|---|---|---|---|---|---|---|---|---|
| B1 keywords | 0.621 [0.530, 0.697] | 0.696 | 0.649 | — | — | — | 0 | < 1 ms |
| B2 TF-IDF + LR | 0.585 [0.484, 0.666] | 0.768 | 0.637 | 0.199 | 0.047 | 0.082 (0.929) | ≈ 0 | 0.2 ms |
| B3 bge-small + LR | 0.637 [0.545, 0.714] | 0.759 | 0.690 | 0.223 | 0.000 | 0 | ≈ 0 | 7 ms |
| B4 DeBERTa, 5 seeds | 0.647 ± 0.044 (0.584–0.703) | 0.807–0.824 | 0.655–0.719 | 0.146–0.180 | 0.05–0.15 | 0 | ≈ 0 | 19 ms |
| **J1** Jev Choice | 0.774 [0.695, 0.838] | 0.872 | 0.795 | 0.070 | 0.503 | 0.655 (0.938) | 0.09 | 256 ms |
| **J2** 3-order average | 0.795 [0.722, 0.857] | 0.883 | 0.813 | 0.069 | 0.474 | 0.649 (0.937) | 0.27 | 277 ms |
| **J3** LR on Jev | **0.796** [0.718, 0.858] | 0.871 | 0.819 | **0.052** | **0.567** | 0.690 (0.932) | 0.27 | 277 ms |
| **J4** hierarchical | 0.716 [0.629, 0.786] | 0.776 | 0.743 | 0.106 | 0.374 | 0.561 (0.927) | 0.09 | 256 ms |
| B5 Haiku 5.5, k = 5 | 0.781 [0.705, 0.845] | 0.871 | 0.801 | 0.179 | 0.000 | 0 (no threshold) | 0.64 (0.32 batch) | 1.3 s |
| B6 Opus 5.5 (ceiling) | **0.857** [0.792, 0.910] | 0.895 | 0.865 | — | — | — | 5.12 (2.56 batch) | 2.5 s |

Coverage at 95 % (oracle) is what the best threshold *chosen on the test* would give; the
frozen-threshold column is what the thresholds fixed on dev actually gave. Full tables
(every variant, per-class F1, all paired differences with Holm and McNemar p-values):
`TEST_TABLES.md`.

- **Jev against the local models.** J2 and J3 beat every local model by 0.09–0.21 macro-F1
  (every CI > 0); J1 by 0.07–0.19 (its CI against the best DeBERTa seed touches 0); J4 only
  beats B0–B2 clearly. The fine-tuned DeBERTa is unstable across seeds (0.58–0.70) and no better
  than the keyword rules on this test.
- **Jev against Haiku.** Same accuracy (J1 −0.008, J2 +0.014, J3 +0.015; every CI includes 0).
  The difference is in confidence: Haiku's five samples agree on 95 % of requirements and those
  unanimous answers are wrong often enough that no threshold reaches 95 % precision; Jev's
  probabilities rank its errors last (AURC 0.05–0.07 vs 0.18).
- **Opus is clearly better than everything** (J3 − B6 = −0.061 [−0.120, −0.009]), at 10–60×
  Jev's cost per requirement (batch vs standard, J2/J3 vs J1) and 10× its latency: the task has headroom.
- **Everything dropped from dev to test** (J1 0.830 → 0.774, J3 0.836 → 0.796, B5 0.858 →
  0.781, B4 0.687 → 0.647). The test projects differ (two of them have no functional
  requirements). The ranking held, except that J3 overtook J1 and B5 lost its dev edge over J1.
- **J4 (hierarchical) is the weakest Jev variant**: its level-1 Noul (`is_functional`) is the
  bottleneck (level-1 F1 0.776 vs 0.872 for J1).

## Calibration and human-review thresholds

![Reliability on test](figures/reliability_test.png)

Calibrators were chosen by log loss in nested, project-grouped CV on dev (`FREEZE.md`).
Identity never won, mostly because Jev's two-decimal probabilities contain exact zeros and B5's
are multiples of 0.2.

| System | ECE dev raw → calibrated (nested CV) | ECE test raw → calibrated |
|---|---|---|
| J1 (vector scaling) | 0.078 → 0.044 | 0.099 → 0.091 |
| J2 (vector scaling) | 0.066 → 0.051 | 0.087 → 0.094 |
| J3 (temperature, T = 0.72) | 0.109 → 0.051 | 0.078 → **0.114** |
| J4 (Platt + vector, per question) | 0.067 → 0.035 | 0.134 → **0.051** |
| B5 (vector scaling) | 0.112 → 0.065 | 0.187 → 0.143 |
| B4 seed 1 (temperature) | 0.145 → 0.062 | 0.207 → 0.126 |

- **Calibration did not transfer well to new projects.** Only J4's per-question calibration
  came close to the 0.05 target on the test. J3's temperature *sharpened* its probabilities
  (T < 1) to fit dev and overshot on the test. With n = 171 the test ECE is itself noisy (J3
  calibrated: 95 % CI [0.073, 0.168]).
- **Review thresholds held about 93–94 % precision instead of 95 %.** At the frozen
  thresholds Jev would decide 56–69 % of requirements automatically with precision 0.93–0.94;
  B5 and B4 would decide none. A production threshold needs a margin, or recalibration on data
  closer to the target documents.

## Robustness

- **Run-to-run** (5 samples): Jev J1 changes label in 2.0 % of repeats (dev 0.7 %); Haiku
  2.0 %. **Option order** (3 rotations): Jev 4.7 % of rotation pairs (dev 2.1 %), close to the
  5 % limit; Haiku 6.8 %. No first-position bias in either.
- **Prompt injection** (60 variants of 20 dev requirements, 3 templates): Jev J1 flips on
  5.0 % and J4 on 3.3 %; Haiku on 1.7 %. Jev resists "ignore previous instructions" and fake
  label prefixes (0 %) but follows a fake `SYSTEM NOTE` in 15 % of cases (attack success
  20 %). Jev's documentation already lists sensitivity to adversarial content; the planned
  service puts a rule-based injection detector and the `addresses_ai` Noul in front of it.

## Cost and latency (measured on the test calls)

Jev: 2,138 input tokens per call, $0.090 per 1,000 requirements for one call (J1, J4) and
$0.27 for three (J2, J3); p50 169 ms and p95 256 ms in series, p95 262 ms at concurrency 8.
Haiku k = 5: $0.64 per 1,000 ($0.32 with the Batch API), p95 1.3 s per requirement with the 5
calls in parallel. Opus: 12 output tokens on average (it did not think at effort low), $5.12
per 1,000 ($2.56 batch), p95 2.5 s. Local models on CPU, batch 1: p95 0.2 ms (B2), 7 ms (B3),
19 ms (B4). No invalid answer from any API model. Test session cost $0.82; project total so far
$1.32 of the $10 budget. Prices of 2026-10-10.

## What the tie-break weighed (decided in ADR 0001: J3)

The rule says the simpler and cheaper system wins when the result is inconclusive. Here:
- J3 vs B5: J3 is cheaper ($0.27 vs $0.32–0.64), 5× faster and far better at selective
  prediction, but needs 3 Jev calls plus a small model trained on labelled data, and is less
  robust to injection.
- J1 is the simplest Jev (one call, $0.09) but did not show non-inferiority on its own.
- B5 is a single prompt with no trained component and better injection robustness, but it
  cannot support a "decide automatically when confident" design.

## Limitations

- 171 test requirements from 6 projects: wide intervals; the project-level bootstrap gives
  J3 − B5 = +0.015 [−0.011, +0.041].
- PROMISE is public since 2007; Jev, Haiku and Opus may all have seen it. The fresh set
  (2025–2026 requirements) is the check, still pending: annotation done (A all 170; B, C, D 30
  each), human agreement low (pooled κ 0.31 on leaves), adjudication not done yet.
- Not done yet: leave-one-project-out, the external binary set (Dalpiaz), the contamination
  probe (§6.8).

## Reproducing

`uv run reqbench reproduce --from-cache` regenerates every dev table, the calibration, the test
results and these figures from the inference cache without any API key; the test is only
re-read from the cache, never re-evaluated. Checked on 2026-11-11: no spend, frozen files
unchanged, `results.json` identical to the logged one except the local models' CPU latency,
which is measured again on every run. `TEST_TABLES.md` and `figures/` are generated;
this file is written by hand.
