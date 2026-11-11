# ADR 0001: Phase 1 decision for requirement-type classification

- **Status:** **superseded** by `0002-phase1-type-decision-b5x.md` (2026-11-11). It
  overlooked Claude Haiku with one call (B5x), which is as accurate and cheaper
  (`docs/reports/phase1/AUDIT.md`, finding C2). Originally accepted on 2026-11-11.
- **Decided by:** the project owner, after the one-time locked-test evaluation.
- **Evidence:** `docs/reports/phase1/REPORT.md`, `TEST_TABLES.md`, `EVAL_LOG.jsonl`
  (freeze `1ff9e151…`, commit `f9db031`).

## Context

The preregistered rule (`docs/PREREGISTRATION.md`) gave **INCONCLUSIVE** on the locked test
(171 requirements, 6 projects). The best Jev variant, J3 (logistic regression on the outputs of
three `jev-1.14.0` calls), was non-inferior to the best alternative, B5 Claude Haiku 5.5 with
k = 5 (macro-F1 0.796 vs 0.781, Δ +0.015 [−0.026, +0.054]). It was clearly better than every
local model (criterion 2a) and at selective prediction (AURC 0.052 vs 0.179, criterion 2b).
It failed two operational guardrails of criterion 3: calibrated ECE on the test (0.114 > 0.05)
and adversarial robustness (5.0 % label flips vs 1.7 % for B5). Criterion 2(c), the fresh set,
is pending, but it cannot produce a GO while criterion 3 fails.

For an inconclusive result the preregistration says "the simpler, cheaper system wins",
without defining *simpler*. The candidates were J3, J1 (Jev with one call, simplest and
cheapest, but not shown non-inferior on its own: −0.008 [−0.057, +0.042]) and B5 (one prompt,
no trained component, but no usable confidence: coverage at 95 % precision = 0).

## Decision

**J3 (Jev + logistic regression) is the type classifier carried forward.**

Reasons: it is the cheapest of the non-inferior options ($0.27 per 1,000 requirements vs
$0.32 batch / $0.64 standard for B5), about 5× faster (p95 277 ms vs 1.3 s), and the only one
whose confidence supports the planned "decide automatically when confident, send the rest to
a human" design (it decided 69 % of the test automatically at the frozen threshold). The extra
complexity over B5 (three Jev calls plus a small trained model) is accepted in exchange.

This is the owner's application of the tie-break, **not a GO**: the result stays
"inconclusive" in the report, and the two failed guardrails are conditions on its use.

## Conditions

1. **Human review with a safety margin.** On the test, the thresholds frozen on dev gave
   93–94 % precision instead of 95 %. The service needs a more conservative threshold, or
   recalibration on data closer to the target documents (J3's dev temperature, T = 0.72,
   over-sharpened its probabilities on new projects).
2. **Injection defence in front of Jev.** The rule-based injection detector and the
   `addresses_ai` Noul planned in §5.5 are mandatory; Jev followed a fake `SYSTEM NOTE` in
   15 % of those variants.
3. **The J3 model is part of the release.** Its logistic regression is trained on labelled
   dev data (hash in `frozen/phase1/final_models.json`); a new Jev version means re-running
   dev, refitting J3, recalibrating and re-testing (plan §6.4).
4. **Fresh-set check.** The fresh set is evaluated once, with the same freeze, after its
   adjudication is committed. If J3 is clearly worse than B5 there (beyond δ = 0.03, or a
   large PROMISE-to-fresh drop that the trained models do not show), this decision is
   reopened.

## Consequences

Phase 2 (quality checks) and the service design use J3 for the type question. Opus 5.5
(0.857) shows the task has headroom; a stronger option can be reconsidered if its cost falls
or if the fresh set changes the picture.
