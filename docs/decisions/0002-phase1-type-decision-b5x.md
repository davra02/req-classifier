# ADR 0002: Phase 1 decision for requirement-type classification (supersedes ADR 0001)

- **Status:** accepted (2026-11-11). Supersedes `0001-phase1-type-decision.md`.
- **Decided by:** the project owner, after the one-time locked-test evaluation and the
  independent audit (`docs/reports/phase1/AUDIT.md`).
- **Evidence:** `docs/reports/phase1/REPORT.md` (with errata E1–E8), `AUDIT.md`,
  `runs/phase1/audit/audit.json` (`uv run reqbench audit`).

## Context

The preregistered verdict on the locked test is **INCONCLUSIVE** (unchanged by the audit). For
that case the preregistration says "the simpler, cheaper system wins". ADR 0001 applied that
tie-break in favour of J3 (Jev × 3 + logistic regression) as "the cheapest of the
non-inferior options". The audit showed that this overlooked Claude Haiku 5.5 with a single
call (B5x), declared exploratory before the test:

| | J3 | B5x |
|---|---|---|
| Macro-F1 (test) | 0.796 | 0.788; J3 − B5x = +0.008 [−0.032, +0.045] (post hoc) |
| Cost per 1,000 requirements | $0.27 | $0.13 standard, $0.064 Batch |
| p95 latency | 0.28 s | 1.16 s |
| Usable confidence for automatic decisions | yes (AURC 0.052), but its frozen threshold is miscalibrated (AUDIT C1) | no (one hard label) |
| Components | 3 Jev calls + a trained model, retrained whenever Jev's version changes | one prompt, nothing trained |
| Prompt-injection flips (60 dev variants) | not measured for J3 (J1: 3/60) | 0/60 (difference not significant) |

## Decision

**B5x (Claude Haiku 5.5, one call, the frozen prompt `llm_type_v1`) is the type classifier
carried forward.** This **follows the literal tie-break**: among the systems that cannot be
told apart in accuracy, it is the simplest and the cheapest. It is not a GO for any system.

What is given up: J3's selective prediction (deciding the confident cases automatically and
sending the rest to a person) and its lower latency. Jev was not shown to be a differentiator
for this task: its accuracy ties with a single LLM call, and its one clear advantage, a
confidence that ranks its own errors, matters only for a design that is not being built for
type classification.

## Conditions and consequences

1. **No automatic-decision threshold for type.** B5x gives no probability. If the service
   needs one, a confidence method for Haiku (for example asking for a stated confidence, or
   k > 1 with a different aggregation) has to be measured on dev first and frozen; k = 5 vote
   frequency did not work (coverage at 95 % precision = 0).
2. **Injection defence stays.** The rule-based detector of plan §5.5 sits in front of the
   classifier regardless of the model.
3. **Fresh-set check.** B5x is reported on the fresh set as a full system. What result
   reopens this decision is fixed before that evaluation, in
   `docs/PREREGISTRATION_ADDENDUM.md`, taking into account the low human agreement there
   (pooled κ 0.31).
4. **Pinning.** `claude-haiku-5-5`, thinking disabled, effort low, `max_tokens` 256, output
   constrained to the 6-label enum; a model change means re-running dev and the held-out sets.
5. **Jev's role moves to phase 2 as a hypothesis**: binary quality criteria with a probability
   may suit it better than a 6-way type label. That is a bet without data yet; it will be
   tested with Haiku (one call) as the baseline from the start and a stopping rule fixed
   before spending.
