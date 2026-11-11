# LLM prompt versions (phase 1, dev only)

The preregistration allows at most 15 versions of the LLM prompt; each one is listed here with
its hashes and its dev results. Dev = 437 requirements (examples_v1 excluded); paired
bootstrap with 10,000 resamples. Outputs: `runs/phase1/dev/` (`haiku_answers.csv`,
`haiku_latency.csv`, `oof_B5*.csv`, `summary_B5*.json`, `haiku_report.json`), regenerated
from the inference cache with `uv run reqbench haiku`.

## v1: `llm_type_v1` + `examples_v1` (2026-11-11)

Not iterated: Jev was evaluated with its first question set only, so the LLM gets the same.

- Model `claude-haiku-5-5` (every one of 3,209 answers reported it), thinking disabled
  (`{"type": "disabled"}`), effort `low`, `max_tokens` 256, output constrained to the 6-label
  enum with `output_config.format` (labels compared case-insensitively, as the
  structured-outputs docs advise). Same definitions and examples as Jev (rendered from
  `taxonomy_v1`); the requirement goes only in the user turn, inside `<requirement>` tags.
- Prompt hashes (`question_set_hash`, prompt + config): rotation 0 `54ab42b2…` (identical to
  the phase-0 smoke test), rotation 1 `4f5d24e8…`, rotation 2 `6a2d00c5…`. A rotation shifts
  both the order of the definitions and the order of the enum.
- B5 = label frequencies over k = 5 samples of rotation 0 (Haiku 5.5 rejects non-default
  temperatures); prediction = most frequent label, ties to the label drawn first.
- Calls: 437 × (5 samples + 2 rotations) = 3,059 (20 from the smoke test, 3,039 through the
  Message Batches API, 3 batches, no errors) + 150 standard calls for latency (30 random
  requirements × 5 concurrent samples). Every answer ended with `end_turn` and a valid label.
  Cost $0.213 (estimate $0.211). Mean 1,210 input and 10.4 output tokens per call.

| System | Macro-F1 [95 % CI] | Level-1 F1 | Level-2 F1 | Accuracy | ECE (raw) | AURC | Cov. @ 95 % |
|---|---|---|---|---|---|---|---|
| B3 bge-small + LR | 0.617 [0.565, 0.665] | 0.807 | 0.636 | — | 0.218 | 0.246 | 0.011 |
| B4 DeBERTa-v3-small (5 seeds) | 0.687 ± 0.017 | 0.814 ± 0.055 | — | — | 0.18 | 0.146–0.226 | 0.025–0.144 |
| J1 Jev Choice | 0.830 [0.787, 0.867] | 0.900 | 0.837 | 0.849 | 0.078 | 0.049 | 0.641 |
| **B5 Haiku 5.5, k = 5** | **0.858 [0.817, 0.894]** | 0.930 | 0.862 | 0.879 | 0.112 | 0.109 | **0.000** |
| B5x Haiku, one sample (exploratory) | 0.851 [0.809, 0.888] | 0.932 | 0.851 | 0.876 | — | — | — |

- Paired differences B5 − reference (Holm over the 7 references for macro-F1):
  - J1: macro-F1 **+0.028 [−0.001, +0.058]** (Holm p 0.059); level-1 F1 +0.030 [+0.014,
    +0.048]; accuracy +0.030 [+0.009, +0.053]; AURC **+0.060 [+0.038, +0.082]** (J1 better).
  - B3: macro-F1 +0.241 [+0.188, +0.294]; AURC −0.137 [−0.190, −0.085].
  - B4 seeds 1-5: macro-F1 +0.142 to +0.185 (every CI > 0); AURC vs seed 1 −0.037
    [−0.079, +0.002].
- Per-class F1 (B5 / J1): functional 0.920 / 0.889, performance 0.920 / 0.878, security
  0.874 / 0.874, usability 0.788 / 0.752, operational_constraint 0.883 / 0.857, other
  0.762 / 0.732. Main confusion: usability → functional (14 of 74).
- **Selective prediction:** the five samples agree on 423 of 437 requirements (96.8 %), and
  those unanimous answers are right only 89.1 % of the time (the other 14: 50 %). No
  confidence threshold reaches 95 % precision, so coverage at 95 % is 0 and AURC is twice
  J1's. Calibration cannot fix this: it is monotonic and the unanimous block stays one block.
  Raw log loss is 3.97 because wrong unanimous answers get probability 0.
- Stability (5 samples of rotation 0): label changes in 1.8 % of repeats vs sample 0 (Jev
  0.7 %); 3.2 % of requirements change at least once.
- Definition order (3 rotations, sample 0): label changes in 4.0 % of rotation pairs (5.7 % of
  requirements), against 2.3 % between two samples with the same order; no first-position
  bias (share predicted as the leading label: functional 0.442 vs 0.463, performance 0.103 vs
  0.108, security 0.096 vs 0.096).
- Cost per 1,000 requirements (prices of 2026-10-10, checked unchanged on 2026-11-11):
  B5 (k = 5) $0.63 standard / $0.315 batch; one call $0.126 / $0.063. Jev J1: $0.089.
- Latency (standard calls, no cache hits, from this Mac): per call p50 1,093 ms, p95
  1,910 ms; per requirement with the 5 samples concurrent p50 1,171 ms, p95 2,146 ms. Jev:
  p50 163 ms, p95 207 ms.
- Caveat: PROMISE is public since 2007 and may be in Haiku's training data as well as Jev's;
  the fresh set checks both.
