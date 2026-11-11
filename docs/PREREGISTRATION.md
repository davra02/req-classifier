# Preregistration: phase 1, requirement-type benchmark

Written on 2026-10-10, **before any model was run on the data**. Any later change to this file
is a deviation and must be listed in the "Deviations" section with its date and reason. The
git history of this file is the audit trail.

## Question

Does Jev (`jev-1.14.0`; `jev-1.13.0` until deviation 2) classify requirement types well enough, cheaply enough and reliably
enough to be used instead of simpler or cheaper alternatives? A negative or inconclusive
answer is a valid outcome.

## Data (frozen)

- Dataset `promise_v1`: PROMISE NFR, 622 requirements after removing 3 duplicates
  (content hash `01c163fc28a20857b38a0e62fddeee4536b7bf69e02973bc0ba3cf85123e6973`).
- Taxonomy `taxonomy_v1`: functional / non-functional → performance, security, usability,
  operational_constraint, other (`questions/taxonomy_v1.yaml`).
- Split `promise_v1_split_v1` (seed 20261010, content hash
  `224a6fa374235c45dac6f0d48d83233dbadfb090022f44e5e8bdac8e3ed651ba`):
  locked test = projects 2, 4, 7, 9, 13, 15 (171 requirements); dev = 449 in 5 project-grouped
  folds; 2 excluded.
- Few-shot examples `examples_v1`: 2 per leaf from dev, excluded from dev metrics.
- Fresh set (plan §6.9): ~150 requirements from public documents published from 2025 onwards,
  double-annotated. Frozen before any system is evaluated on it.

## Systems

B0 majority · B1 keyword rules · B2 TF-IDF + logistic regression · B3 `bge-small-en-v1.5`
embeddings + logistic regression · B4 `deberta-v3-small` fine-tuned (5 seeds) · B5 Claude Haiku
5.5 zero-shot (thinking disabled, effort low, k = 5 samples) · B6 Claude Opus 5.5 zero-shot
(effort low, k = 1, test and fresh set only) · J1 Jev flat Choice · J2 Jev averaged over 3
cyclic option orders · J3 Jev probabilities → logistic regression · J4 Jev hierarchical (Noul
`is_functional` + Choice `nfr_subtype`). Calibrated variants are fitted on dev out-of-fold
predictions only.

## Development rules

- All prompt, question, rule and hyperparameter work uses dev only.
- At most **15 versions** of the Jev question set and 15 of the LLM prompt; every version is
  committed with its hash and its dev results.
- Jev and the LLM receive identical class definitions and examples (rendered from the same
  YAML).
- The locked test and the fresh set are evaluated **once**, after code, questions, prompts,
  calibrators, thresholds and seeds are frozen and committed. Each evaluation is logged.

## Metrics

- **Primary:** macro-F1 over the 6 leaves on the locked test.
- Always reported: level-1 F1 (functional vs non-functional); level-2 macro-F1 over the 5 NFR
  leaves on gold-NFR items; per-class precision/recall/F1; confusion matrix; log loss; Brier;
  top-label and classwise ECE (15 equal-mass bins) with reliability diagrams; risk-coverage
  curve, AURC and coverage at 95 % precision; cost per 1,000 requirements; latency p50/p95
  (excluding cache hits); run-to-run flip rate (5 repeats); option-order flip rate (3
  rotations); adversarial flip rate (~60 variants).
- Inference: paired bootstrap (10,000 resamples over requirements; project-level resampling as
  sensitivity analysis) for 95 % CIs of differences vs Jev; McNemar for accuracy; Holm
  correction across comparisons.

## Decision criteria (δ = 0.03 macro-F1)

Let B\* be the best non-Jev system by test macro-F1.

**GO (use Jev for type classification)** only if all hold:
1. Non-inferiority: lower 95 % CI bound of [macro-F1(best Jev) − macro-F1(B\*)] > −δ.
2. Added value, at least one of: (a) ≥ δ better (CI > 0) than every system that costs no more;
   (b) lower AURC than B\* (CI excludes 0); (c) ≥ δ better than B\* on the external or fresh set.
3. Operational: post-calibration ECE ≤ 0.05; run-to-run flip rate ≤ 3 %; option-order flip rate
   ≤ 5 % (or removed by J2); p95 latency ≤ 1 s; cost ≤ $1 per 1,000 requirements; adversarial
   flip rate ≤ that of B5.

**NO-GO** if any holds: best Jev is below B\* by more than δ with a CI that rules out
non-inferiority; best Jev is below B2 (TF-IDF + LR) with no advantage (b) or (c); instability or
order bias exceeds the thresholds and J2 does not fix it.

**Inconclusive** (CIs overlap without meeting GO): the simpler, cheaper system wins, and the
report states the sample size that would be needed to separate them.

## Version policy

Jev is pinned to `jev-1.14.0` (see deviation 2); a response reporting another model aborts the run. If the
pinned version becomes unavailable, all Jev results are rerun on dev, recalibrated and
re-evaluated as a new, separately logged evaluation.

## Deviations

1. **2026-11-10, fresh-set annotation** (decided before any system was evaluated on the fresh
   set). Planned: two annotators label all ~150 sentences. Done instead: annotator A labels all
   170 sentences; two further annotators, B and C, each label a **disjoint subsample of 30**
   (60 sentences double-annotated). Subsamples are drawn with seed 20261010, stratified by A's
   label (proportional allocation, at least 2 per label) and dealt alternately to B and C
   (`reqbench fresh assign`). B and C receive the same leaf definitions and never see A's
   labels. Cohen's κ (A–B, A–C, pooled; leaves and level 1, bootstrap 95 % CI) is reported on
   these 60; their disagreements are adjudicated. For the other 110 sentences, A's label is
   final. Reason: annotator availability. Consequence: the human agreement ceiling has a wider
   interval, and 110 labels rest on a single annotator; both are stated with the results.
   A **reserve** subsample of 30 for annotator D was drawn the same way from the remaining
   110 rows (`reqbench fresh assign --who d`, seed 20261010). It is used only if a further
   annotator is available; if used, its rows join the double-annotated set and the pooled κ.

2. **2026-11-10, Jev version** (decided before any Jev result on dev, test or the fresh set
   existed; the only Jev calls made with `jev-1.13.0` were the 20 smoke-test calls of phase 0).
   Planned: `jev-1.13.0`. Done instead: `jev-1.14.0`, pinned by its versioned ID with the same
   abort rule. Reason: TypeSafe released Jev 1.14 on 2026-10-14 and the `jev-latest` and
   `jev-preview` aliases now point to it; measuring the superseded version would make the
   result less relevant, and a later retirement of 1.13 would force a full rerun, including
   the test. Price ($0.042 per million input tokens) and API are unchanged; the changelog lists
   no change to Choice or Noul and the documented failure modes (including first-option bias)
   are the same, so the robustness measures stay as planned. Only one version is evaluated:
   1.13 is not run on dev, so the version is not chosen by its results. A phase-0-style smoke
   test on 1.14 (20 dev calls) confirmed that the API answers with `jev-1.14.0`.

3. **2026-11-11, order of the held-out evaluations and operational details** (decided after
   the dev results and before the locked test or the fresh set was read by any system; the
   freeze is `frozen/phase1/FREEZE.json`, explained in `docs/reports/phase1/FREEZE.md`).
   - *Order.* Planned (plan §6.9): the fresh set is evaluated once, together with the locked
     test. Done instead: the locked test is evaluated first; the fresh set is evaluated later,
     once, with the **same freeze** (same code, questions, prompts, calibrators, thresholds and
     models; the evaluation runner refuses any other freeze), after its adjudication is
     committed. Reason: the fresh-set annotation finished on 2026-11-11 and its adjudication
     is still pending. Consequence: criterion 2(c) stays "pending" until the fresh set is
     evaluated, and the decision on the test alone can be GO only through 2(a) or 2(b).
   - *Adversarial suite.* `questions/adversarial_v1.yaml`: 20 dev requirements (seed
     20261012, proportional to the gold leaf, ≥ 2 each) × 3 fixed prompt-injection templates
     = 60 variants, generated by code; measured for J1, J4 and B5 (k = 5) in the same session
     as the test. Only dev texts are used, so the test stays untouched by it.
   - *Opus `max_tokens` = 1024* (was 4096 in the smoke test): Opus 5.5 cannot disable
     thinking and the budget guard reserves the full `max_tokens` per call; 1024 keeps the
     worst case of the 171-call test batch within the $3 Opus bucket. Truncated or refused
     answers count as errors and are reported.
   - *Operational reading of the decision rule* (no threshold changes; the reading is coded
     in `reqbench.eval.heldout.decide` and covered by tests): "best Jev" is the best of the
     raw J1-J4 by test macro-F1; B\* is chosen among B0-B5 (B6 Opus is a quality ceiling,
     plan §6.2; exploratory variants are excluded), with B4 ranked by the mean of its 5 seeds
     and compared through its best seed; 2(a) = difference ≥ δ **and** lower CI bound > 0
     against every local system (B0-B4, every seed); 2(b) uses raw probabilities; criterion 3's
     ECE uses the best Jev with its frozen calibrator; NO-GO for "below B\*" requires the whole
     CI below −δ.
