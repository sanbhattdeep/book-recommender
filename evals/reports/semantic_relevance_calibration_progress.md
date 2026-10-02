# Semantic Relevance Judge Calibration Progress Report

**Project:** Book Recommender — Semantic Relevance Evaluation  
**Calibration lineage:** `semantic_relevance_v0.29.0`  
**Current candidate at pause:** `semantic_relevance_v0.29.0-r5`  
**Checkpoint date:** 2026-10-02  
**Status:** **CALIBRATION PAUSED — SAFE TO RESUME FROM THIS REPORT**  
**Independent v0.29 qualification evidence:** **NOT CREATED / NOT CONSUMED**

---

## 1. Purpose of this report

This document is the handoff/checkpoint for the semantic-relevance LLM-as-a-judge calibration work.

It has two purposes:

1. Provide an interview-ready explanation of the evaluation engineering process already completed.
2. Preserve enough technical and evidence-lifecycle detail to resume calibration later without reconstructing the history from terminal logs or old chats.

This report distinguishes:

- calibration/development evidence
- targeted regression evidence
- stability evidence
- independent validation/final-holdout evidence
- release/qualification gates

The current candidate, `v0.29.0-r5`, has passed targeted consumed-development regression but **has not yet completed a full-240 r5 development run, stability qualification, candidate freeze, or fresh independent v0.29 evaluation**.

> **Do not describe v0.29.0-r5 as final-qualified.**

---

## 2. Evaluation problem

The recommender uses an LLM-based judge to score whether a candidate book is semantically relevant to a user query.

The judge returns an ordinal relevance score:

| Score | Meaning |
|---:|---|
| 4 | Strong match — clearly satisfies the central intent and is an obvious/direct recommendation |
| 3 | Good/clear match — relevant overall, but one important aspect is secondary, implicit, weaker, or missing |
| 2 | Partial match — meaningfully satisfies one central part while missing another |
| 1 | Weak/incidental match — genuine overlap exists but is peripheral or superficial |
| 0 | No match — supplied evidence does not establish meaningful relevance or contradicts the request |

Important calibration rules:

- Use the supplied book description as the evidence basis.
- Do not use external book knowledge to rescue missing evidence.
- When between adjacent scores, prefer the lower score unless the higher anchor is clearly supported.
- A direct concept mention can still be only incidental.
- Missing a central query component normally prevents a strong score.

The core challenge is that the judge contains LLM-generated semantic judgments, so the system is **non-deterministic** and can fail through:

- over-inference
- under-inference
- unstable component verification
- unsupported relationship inference
- cross-span composition mistakes
- deterministic guard gaps
- rubric/label tension
- prompt/model variance

---

## 3. Calibration architecture

The judge is intentionally hybrid rather than purely prompt-based.

At a high level:

1. Each query is decomposed into semantic facets.
2. Candidate evidence spans are selected.
3. The LLM verifies whether each facet/component is supported.
4. Deterministic guards prevent known classes of invalid inference.
5. Deterministic scoring converts facet support into the final 0–4 score.
6. Human labels remain immutable gold evidence unless explicitly adjudicated under a recorded label-review process.
7. Targeted regression sets protect earlier fixes from being broken by later calibration changes.

Current semantic identity at the pause point:

| Artifact | Version |
|---|---|
| Candidate | `semantic_relevance_v0.29.0-r5` |
| Development dataset | `6.1.0` |
| Facet specification | `0.10.1` |
| Judge config | `0.29.0-r5` |
| Regression manifest | `2.4.0` |
| Rubric | `0.1.0` |
| Scoring module | unchanged from earlier v0.29 candidates |

Development dataset SHA-256:

```text
cfcc465497079944fec6da17f14bbe5f3350d9385ea9ddc646610b7751d72da7
```

---

## 4. Evidence lifecycle and why it matters

A major goal of this project has been to avoid leaking evaluation evidence back into calibration.

### Consumed development evidence

Cases already seen during calibration can be used for:

- debugging
- regression tests
- neighborhood checks
- targeted stability tests
- comparing candidate versions

They **cannot** be treated as fresh proof of generalization.

Current v0.29 development set:

```text
240 consumed cases
```

### Fresh independent evidence

Fresh unseen human-labelled cases are required to qualify the final v0.29 candidate.

At this checkpoint:

```text
Fresh independent v0.29 evidence:
NOT CREATED
NOT CONSUMED
```

This is deliberate. The independent set must be created **after the candidate is frozen**.

---

## 5. Historical evidence that must never be reused as fresh qualification evidence

### v0.27 historical final holdout

Consumed once. Historical diagnostic/development evidence only.

### v0.28 r4 independent validation

Consumed once.

Historical decision:

```text
REVIEW_STOP_FINAL_HOLDOUT
```

It may be used as consumed regression/diagnostic evidence only.

### v0.28 r8 final holdout

Consumed exactly once.

Canonical run:

```text
evals\runs\semantic_relevance_v0_28_final_holdout\20261001T044506Z_final_r8
```

Final decision:

```text
FINAL_HOLDOUT_REVIEW
```

Closeout status:

```text
EVALUATED_NOT_FINAL_QUALIFIED
```

The v0.28 final holdout:

- must not be rerun for qualification
- must not be relabelled for v0.28
- must not be reused as independent v0.29 evidence
- may be used as v0.29 development/diagnostic regression evidence

---

## 6. Frozen quantitative gates

| Gate | Threshold |
|---|---:|
| Within-one agreement | `>= 0.95` |
| Exact agreement | `>= 0.50` |
| Quadratic weighted kappa | `>= 0.80` |
| Linear weighted kappa | `>= 0.60` |
| Severe disagreements (`|human - judge| >= 2`) | `<= 1` |
| Binary relevance precision (`0` vs `>0`) | `>= 0.90` |
| Binary relevance recall (`0` vs `>0`) | `>= 0.80` |
| Deterministic-cue severe false positives | `= 0` |

These gates measure different risks and should not be collapsed into a single score.

---

## 7. How to interpret the metrics

### Exact agreement

Percentage of cases where:

```text
judge_score == human_score
```

This is the strictest agreement metric.

A human score of 3 and judge score of 2 counts as incorrect even though the scores are adjacent.

### Within-one agreement

Percentage where:

```text
abs(human_score - judge_score) <= 1
```

Useful for a 0–4 ordinal rubric because it separates ordinary adjacent disagreement from materially wrong judgments.

### Mean Absolute Error (MAE)

```text
mean(abs(human_score - judge_score))
```

Lower is better.

An MAE of `0.32` means the judge differs from the human label by about one-third of a score point on average.

### Linear weighted kappa

Measures ordinal agreement beyond chance and penalizes larger disagreements more than smaller ones.

### Quadratic weighted kappa

Like linear kappa, but large ordinal differences receive much stronger penalties. This is useful because `0 -> 3` is far more serious than `3 -> 2`.

### Binary precision

Collapse the scale into:

```text
0 = irrelevant
1–4 = some relevance
```

Precision asks:

> When the judge says a book is relevant, how often is the human label also positive?

Low precision means the system recommends too many irrelevant books.

### Binary recall

Recall asks:

> Of books humans consider relevant, how many does the judge identify as relevant?

Low recall means valid recommendations are missed.

### F1

Harmonic mean of precision and recall. Useful as a summary, but not sufficient as the only release gate.

### Severe disagreement count

Defined as:

```text
abs(human_score - judge_score) >= 2
```

Examples:

```text
human 0, judge 2
human 0, judge 3
human 4, judge 2
```

A system can have strong averages while still containing dangerous severe failures, which is why this is explicitly gated.

---

## 8. v0.28 r8 — important historical qualification result

Before v0.29 began, `v0.28.0-r8` performed strongly on consumed development evidence.

### Full 210-case consumed development

```text
Exact agreement:           69.0%
Within ±1 agreement:       99.0%
MAE:                       0.319
Linear weighted kappa:     0.775
Quadratic weighted kappa:  0.909

Precision:                 94.4%
Recall:                    87.5%
F1:                        90.8%

Severe disagreements:      2
```

However, the one-time independent final holdout exposed generalization problems.

### r8 independent final holdout — 30 cases

```text
Exact agreement:           66.7%
Within ±1 agreement:       83.3%
MAE:                       0.500
Linear weighted kappa:     0.575
Quadratic weighted kappa:  0.702

Precision:                 75.0%
Recall:                    81.8%
F1:                        78.3%

Severe disagreements:      5
```

Gate result:

```text
PASS exact
PASS recall
PASS deterministic-cue severe FP

FAIL within-one
FAIL quadratic kappa
FAIL linear kappa
FAIL severe disagreements
FAIL precision
```

Decision:

```text
FINAL_HOLDOUT_REVIEW
```

This is a concrete example of why evaluating only on calibration/development data is unsafe.

---

## 9. v0.29 lineage — why it was created

v0.29 began only after closing v0.28.

The five severe r8 final-holdout cases were reviewed as **consumed evidence**.

Human-label adjudication concluded:

```text
U4_Q03_T01: 0 -> 2
U4_Q03_T02: keep 2
U4_Q04_T02: keep 0
U4_Q06_T01: 2 -> 0
U4_Q09_T04: 1 -> 2
```

Only three human labels were changed.

The resulting v0.29 development dataset became:

```text
semantic_relevance_v0.29_development.v6.1.0.csv
240 cases
```

Human-label adjudication and semantic repair were intentionally separated.

---

## 10. v0.29 repair lineage

### r1

Targeted repairs:

- Q03 false-zero / finding-purpose behavior
- Q04 journey/movement guard
- Q09 resistance behavior and score cap

Targeted regression exposed:

- Gulliver's Travels regression
- Animal Farm and 1984 regression

### r2

Added:

- Q04 cross-span movement recovery
- Q09 same-span opposition + authoritarian framing recovery

Q09 repair held. Gulliver still failed because `shipwrecked` was not being deterministically recognized as a danger cue.

### r3

Added narrow deterministic Q04 shipwreck-danger handling.

Targeted r3:

```text
58 / 58 gated cases PASS
2 diagnostics INFO
```

Important repaired examples:

```text
U2_Q04_T70  Gulliver's Travels              human 3 / judge 3
U2_Q09_T02  Animal Farm and 1984            human 4 / judge 3
U4_Q03_T01  Bridget Jones's Diary           human 2 / judge 2
U4_Q03_T02  Discover Your Destiny...         human 2 / judge 2
U4_Q04_T02  Banker                          human 0 / judge 0
U4_Q06_T01  Veronika Decides to Die         human 0 / judge 0
U4_Q09_T04  Brief and Frightening Reign...  human 2 / judge 2
```

---

## 11. Latest complete aggregate calibration metrics

The most recent complete **240-case aggregate run** available at this checkpoint is r3:

```text
evals\runs\semantic_relevance_v0_29_r3_development\20261001T135129Z
```

Overall metrics:

```text
Cases:                     240
Exact agreement:           70.0%
Within ±1 agreement:       98.3%
MAE:                       0.321
Linear weighted kappa:     0.768
Quadratic weighted kappa:  0.897

TP=95
FP=6
FN=12
TN=127

Precision:                 94.1%
Recall:                    88.8%
F1:                        91.3%

Severe disagreements:      4
Over-promotions >=2:       1
Under-promotions >=2:      3
```

### Interpretation

The aggregate metrics are strong:

- exact agreement comfortably exceeds 50%
- within-one is very high
- both weighted kappas show substantial ordinal agreement
- precision and recall are both above the frozen thresholds

But these are **not current r5 qualification metrics**.

They came from r3 and still contained important case-level defects later addressed in r4/r5.

Correct interpretation:

> r3 demonstrated strong aggregate calibration but was not stable enough at the case/facet level to freeze.

---

## 12. r3 full-run defects that drove later work

### U_Q04_T10 — The Valkyries

```text
Human: 4
r3 full run: 2
```

The description explicitly said the characters:

```text
take off on a forty day adventure into the ... Mojave Desert
```

but the Q04 movement guard did not recognize it as movement/travel.

Focused stability:

```text
2
3
2
```

The `dangerous journeys` facet remained absent in all three successful runs.

Conclusion:

```text
stable deterministic movement-anchor gap
+ LLM variance in fantasy-adventure verification
```

### U4_Q07_T03 — By the River Piedra I Sat Down and Wept

Human:

```text
0
```

The description is about two former lovers.

Observed scores before r5:

```text
r3 full run:      3
focused stability:1
focused stability:1
focused stability:1
r4 targeted:      3
```

The high-scoring runs hallucinated a parent-child relationship from generic:

```text
difficulties
blame
resentment
```

Conclusion:

```text
bimodal stochastic grounding failure
```

---

## 13. r4 repair

r4 made **one semantic change only**:

Q04 movement/travel gained a narrow directional-departure rule for constructions such as:

```text
take off on a forty day adventure into the Mojave Desert
```

The rule deliberately required:

- take/takes/took/taking off on
- a journey/adventure/trip/trek/voyage/expedition noun
- directional continuation such as into/through/across/toward/to

It intentionally avoided idioms such as:

```text
sales took off
the story takes off
take off his coat
```

No Q07 semantic change was made in r4.

### r4 targeted result

```text
59 / 60 gated PASS
```

Q04 succeeded:

```text
U_Q04_T10  The Valkyries  human 4 / judge 3
```

Only failure:

```text
U4_Q07_T03  human 0 / judge 3
```

---

## 14. r5 repair — current candidate

Current candidate:

```text
semantic_relevance_v0.29.0-r5
```

r5 added a **precision-only negative grounding guard** for Q07.

For:

```text
parent_child_relationship
```

a positive component result cannot survive unless its supporting supplied text contains an explicit kinship anchor such as:

```text
parent
mother
father
son
daughter
child
```

The guard:

- never creates positive evidence
- does not modify scoring
- does not modify the facet spec
- does not modify human labels
- does not modify Q04
- only rejects unsupported parent-child identity inference

---

## 15. Latest calibration result — r5 targeted regression

Canonical targeted run:

```text
evals\runs\semantic_relevance_v0_29_r5_development\20261002T091035Z_targeted_r5
```

Result:

```text
60 / 60 gated cases PASS
2 diagnostics INFO
TARGETED REGRESSION: PASS
```

Critical checks:

```text
U_Q04_T10    The Valkyries                  human 4 / judge 3   PASS
U4_Q07_T03   By the River Piedra...         human 0 / judge 0   PASS

U2_Q04_T70   Gulliver's Travels             human 3 / judge 3   PASS
U2_Q09_T02   Animal Farm and 1984           human 4 / judge 3   PASS

U4_Q03_T01   Bridget Jones's Diary          human 2 / judge 2   PASS
U4_Q03_T02   Discover Your Destiny...       human 2 / judge 2   PASS
U4_Q04_T02   Banker                         human 0 / judge 0   PASS
U4_Q06_T01   Veronika Decides to Die        human 0 / judge 0   PASS
U4_Q09_T04   Brief and Frightening Reign... human 2 / judge 2   PASS
```

Known diagnostic-only cases:

```text
U3_Q10_T01  Principle Centered Leadership
U3_Q11_T04  New Moon
```

These are not current tuning targets.

---

## 16. How to interpret the r5 targeted PASS

The r5 targeted result means:

> All known targeted regression conditions currently pass on consumed development evidence.

It does **not** mean:

- full-development aggregate metrics have been recomputed for r5
- r5 is stable across repeated runs
- r5 is frozen
- r5 has passed independent validation
- r5 is production-qualified

A concise interview phrasing:

> "Targeted regression tells me whether known failure modes are still controlled. Independent evaluation tells me whether the judge generalizes beyond the cases used to calibrate it."

---

## 17. Current calibration checkpoint

```text
Candidate: semantic_relevance_v0.29.0-r5

Preparation:                PASS
Localized semantic tests:   PASS
Semantic/provenance tests:  PASS
Targeted regression:        PASS (60/60)
Diagnostics:                2 INFO

Full 240 r5 run:            NOT RUN
r5 stability matrix:        NOT RUN
Candidate freeze:           NOT DONE

Fresh v0.29 independent set:
                            NOT CREATED
                            NOT CONSUMED

Independent qualification:  NOT RUN
```

This is the exact point from which calibration should resume.

---

## 18. Resume calibration from here

Do not rebuild or rerun historical consumed holdouts.

Do not start fresh independent v0.29 evaluation yet.

### Step 1 — full r5 consumed-development evaluation

From repo root:

```powershell
uv run python .\evals\run_judge_v0_29_r5_development.py 2>&1 |
  Tee-Object -FilePath .\v029_r5_full240_output.txt
```

Expected evidence role:

```text
consumed development evidence
```

If the run reaches `240/240`, do not rerun it merely because analysis tooling needs correction.

### Step 2 — analyze full r5 results

Report:

- overall 240 metrics
- frozen development reference metrics
- historical diagnostic subgroup metrics
- severe disagreements
- targeted expectation status
- new severe-regression guards
- known diagnostics

Preserve the historical provenance methodology. Do not accidentally treat old consumed holdouts as fresh qualification evidence.

### Step 3 — classify failures before tuning

For every new severe or gate failure classify it first as:

```text
human-label problem
rubric/spec ambiguity
deterministic implementation defect
LLM stochastic instability
unsupported inference
scoring defect
analysis-tooling defect
```

### Step 4 — stability evaluation

If the full r5 development gate passes, run a deliberately selected stability matrix multiple times.

Include:

- repaired Q03 cases
- Q04 movement/journey boundary cases
- Q09 resistance cases
- Q11 cross-span mythology cases
- Q07 parent-child grounding cases
- deterministic negative controls

Compare:

- final score
- facet support
- evidence spans
- component grounding
- severe-error stability

### Step 5 — freeze candidate

Only after:

```text
targeted regression PASS
full consumed-development gate PASS
stability PASS
```

Pin:

- judge SHA
- scoring SHA
- facet-spec SHA
- judge-config SHA
- regression-manifest SHA
- rubric SHA
- development-dataset SHA
- stability evidence
- release gates
- evidence boundary

### Step 6 — create fresh independent v0.29 evidence

Only after freeze:

1. create unseen candidate pool
2. human-label blind to judge output
3. freeze labels
4. split/lock evaluation evidence before judge execution
5. pin hashes
6. execute according to preregistered protocol

No U4 case already consumed by v0.28 may count as fresh v0.29 independent evidence.

---

## 19. Interview demonstration flow

Calibration can now be paused.

A compact interview evaluation demonstration can show:

```text
1. Define evaluation objective
2. Choose/create labelled evaluation dataset
3. Run system/judge
4. Compute metrics
5. Inspect severe errors
6. Apply preregistered release gates
7. Produce PASS / REVIEW / FAIL
8. Save reproducibility artifacts
```

Recommended metrics:

```text
exact agreement
within-one agreement
MAE
linear weighted kappa
quadratic weighted kappa
precision
recall
F1
severe disagreement count
```

Recommended release-gate principle:

> Do not gate only on an average metric. Combine overall quality, ordinal agreement, binary relevance behavior, and severe-error controls.

The historical r8 result is a strong interview example:

```text
consumed development looked strong
but independent final holdout failed 5/8 gates
```

This demonstrates:

```text
overfitting to calibration evidence
the value of independent holdouts
the importance of preregistered gates
```

---

## 20. Suggested actual-evaluation architecture for the interview

Keep **calibration** and **evaluation** separate.

A clean structure is:

```text
evals/
├── datasets/
├── runs/
├── metrics/
├── reports/
├── releases/
└── regression/
```

A completed evaluation run should produce artifacts such as:

```text
judge_results.csv
human_vs_judge.csv
confusion_matrix.csv
largest_disagreements.csv
summary.json
release_gate_result.json
run_metadata.json
```

Example machine-readable release decision:

```json
{
  "decision": "PASS",
  "checks": {
    "within_one_ge_0_95": true,
    "exact_ge_0_50": true,
    "quadratic_kappa_ge_0_80": true,
    "linear_kappa_ge_0_60": true,
    "severe_disagreements_le_1": true,
    "precision_ge_0_90": true,
    "recall_ge_0_80": true,
    "deterministic_severe_fp_eq_0": true
  }
}
```

This shows that evaluation is an engineering system, not simply "ask an LLM whether the answers look good."

---


## 21. Worked example — how one query is actually scored by the LLM judge

This section walks through a real case from the calibration dataset end-to-end.

### Example query

```text
A fantasy adventure involving magic and dangerous journeys
```

Query ID:

```text
Q04
```

Candidate:

```text
The Valkyries — Paulo Coelho
```

Human label:

```text
4
```

The human reviewer considered this a strong match for:

- fantasy adventure
- magic
- dangerous journey

A key part of the supplied description is:

```text
Paulo and his wife, Cristina, drop everything, pack their bags,
and take off on a forty day adventure into the starkly beautiful
and sometimes dangerous Mojave Desert...
```

The same description also refers to:

```text
a magical tale
a curse
a guardian angel
a modern-day adventure
a metaphysical odyssey
```

The important point is that the judge does **not** simply ask the LLM:

```text
"How relevant is this book from 0 to 4?"
```

Instead, the score is assembled through a structured evaluation pipeline.

### Code map for this worked example

The links below are relative to this report's recommended repo location,
`evals/reports/semantic_relevance_calibration_progress.md`, so they can be opened directly while browsing the repository in GitHub.

### Execution-type legend

| Type | Meaning |
|---|---|
| **LLM** | The model makes a bounded semantic judgment from supplied text and a structured prompt/schema. |
| **Deterministic** | Python/configuration code makes the decision with no model judgment at runtime. |
| **Hybrid** | The LLM proposes/selects/verifies semantic evidence, then deterministic validation, guards, composition, or scoring constrain the result. |
| **Frozen config** | Human-authored/versioned evaluation policy that is loaded deterministically at runtime; the LLM does not redefine it per case. |

| Pipeline responsibility | Execution type | What to inspect when debugging | Current r5 implementation |
|---|---|---|---|
| Run orchestration: load dataset/config/spec, call judge, call deterministic scorer | **Deterministic** | Confirm the intended dataset/config/spec versions and that the expected case was executed exactly once | [`run_judge_v0_29_r5_development.py` — evaluation loop](../run_judge_v0_29_r5_development.py#L1339-L1450) |
| Frozen Q04 query, facets, required components, and negative boundaries | **Frozen config / deterministic at runtime** | Verify the query decomposition itself is correct before blaming the model | [`semantic_relevance_query_facets.v0.10.1.json` — Q04](../facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json#L211-L280) |
| Split the exact description into numbered source spans | **Deterministic** | Check whether the expected phrase is present verbatim and which span ID contains it | [`build_description_spans()`](../semantic_relevance_facet_judge.py#L370-L398) |
| Select candidate evidence spans for one facet | **Hybrid: LLM selection + deterministic schema/span validation** | Inspect `facet_evidence_json` and the candidate span IDs. Ask whether the best supplied span was missed or a weak span was selected | [`build_evidence_selection_prompt()`](../semantic_relevance_facet_judge.py#L466-L531) and [`select_candidate_evidence()`](../semantic_relevance_facet_judge.py#L582-L639) |
| Verify one frozen semantic component against one exact span | **Hybrid: LLM verification + deterministic validation/guards** | Inspect `component_evidence_ledger_json`: `grounding_relation`, `negative_boundary_applied`, `external_knowledge_required`, `reason`, and supporting span IDs | [`build_isolated_component_prompt()`](../semantic_relevance_facet_judge.py#L2892-L2949) and [`verify_isolated_component()`](../semantic_relevance_facet_judge.py#L2969-L3096) |
| Assemble component results for a span / compare ranked candidates | **Mostly deterministic assembly; candidate verifications originate from LLM/hybrid stages** | Check which required component is missing and whether the final relation follows mechanically from the component states | [`_assemble_single_span_from_component_results()`](../semantic_relevance_facet_judge.py#L3097-L3167), [`verify_candidate_evidence()`](../semantic_relevance_facet_judge.py#L3168-L3277), [`verify_ranked_candidates()`](../semantic_relevance_facet_judge.py#L3309-L3395) |
| Q04 directional-departure detection added in r4 | **Deterministic** | Test the exact phrase against the regex/helper. If it does not match, this is an implementation/coverage issue, not LLM variance | [`_Q04_R4_DIRECTIONAL_DEPARTURE_RE` and `_q04_r4_has_directional_departure_anchor()`](../semantic_relevance_facet_judge.py#L1878-L1889) |
| Q04 movement guard + cross-span movement recovery | **Deterministic guard/recovery around LLM component output** | Compare raw semantic evidence with the post-guard component result. Look for a guard rejecting or recovering an otherwise positive/negative component | [`_q04_r1_movement_text_anchor_guard()`](../semantic_relevance_facet_judge.py#L2591-L2629), [`_q04_r1_recovery_movement_text_anchor_guard()`](../semantic_relevance_facet_judge.py#L2632-L2667), [`_q04_r2_cross_span_movement_component_check()`](../semantic_relevance_facet_judge.py#L2671-L2702) |
| Q04 shipwreck danger recovery | **Deterministic** | Confirm the shipwreck cue and travel anchor are both present before expecting recovery | [`_q04_r3_shipwreck_danger_component_check()`](../semantic_relevance_facet_judge.py#L2706-L2735) |
| Orchestrate the complete pipeline for one facet | **Hybrid orchestration** | Follow one facet in order: precheck → deterministic cue → candidate selection → component verification → composition/recovery → prominence | [`evaluate_one_facet()`](../semantic_relevance_facet_judge.py#L4432-L4936) |
| Orchestrate every frozen facet for one query/book pair | **Hybrid orchestration** | Check whether the problematic score comes from one facet or from multiple facet assessments | [`generate_validated_semantic_verdict()`](../semantic_relevance_facet_judge.py#L4939-L5015) |
| Convert verification relation + prominence into `absent/incidental/meaningful/strong` support | **Deterministic** | If semantic verification looks right but facet support looks wrong, debug this mapping rather than the LLM | [`derive_facet_support()`](../semantic_relevance_facet_scoring.py#L518-L559) |
| Convert all facet supports into the final 0–4 score | **Deterministic** | Start with `scoring_explanation`, then compare the facet counts/coverage and caps/rules with `compute_facet_score()` | [`compute_facet_score()`](../semantic_relevance_facet_scoring.py#L586-L790) |
| Persist score, evidence, facet assessments, component ledger, and scoring diagnostics | **Deterministic** | Confirm the audit fields in `judge_results.csv` were populated and are internally consistent | [`run_judge_v0_29_r5_development.py` — persisted audit fields](../run_judge_v0_29_r5_development.py#L1437-L1605) |
| Define targeted regression expectations | **Frozen config / deterministic** | Check whether the expected range is appropriate and whether it predates the run being evaluated | [`semantic_relevance_v0.29_regression_manifest.v2.4.0.json`](../datasets/semantic_relevance_v0.29_regression_manifest.v2.4.0.json#L5-L208) |
| Compare targeted judge output with human labels and return PASS/FAIL | **Deterministic** | If case scoring looks correct but the gate is wrong, inspect expectation lookup/comparison logic rather than the judge | [`analyze_v0_29_r5_targeted.py`](../analyze_v0_29_r5_targeted.py#L20-L69) |

> **Note:** these line links are pinned to the current `v0.29.0-r5` source layout. If the source files are edited later, function names are the more durable reference and line numbers may move.

---

### Step 1 — decompose the query into semantic facets

**Execution:** **Frozen config / deterministic at runtime.** The facet structure is versioned before the run; the LLM does not invent a new decomposition for each book.

**Debug first:** If the judge seems to be solving the wrong semantic problem, inspect the frozen Q04 definition and required components before inspecting model output.


**Code reference:** [`semantic_relevance_query_facets.v0.10.1.json` — Q04](../facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json#L211-L280)

Q04 is represented by three core facets:

```text
F1: fantasy adventure
F2: magic
F3: dangerous journeys
```

This decomposition is defined once in the frozen facet specification and reused for every book evaluated against Q04.

The judge therefore evaluates three separate questions rather than one vague overall question:

```text
Does the evidence establish a fantasy adventure?
Does the evidence establish magic?
Does the evidence establish a dangerous journey?
```

---

### Step 2 — select candidate evidence spans

**Execution:** **Hybrid.** Span creation is deterministic; the LLM chooses candidate evidence spans; Python validates that returned span IDs actually exist and retries invalid structured output.

**Debug first:** Inspect `facet_evidence_json` and the chosen `candidate_evidence_span_ids`. If the most important text was never selected, later verification cannot rescue it unless a full-context recovery stage explicitly does so.


**Code references:** [`build_description_spans()`](../semantic_relevance_facet_judge.py#L370-L398), [`build_evidence_selection_prompt()`](../semantic_relevance_facet_judge.py#L466-L531), [`select_candidate_evidence()`](../semantic_relevance_facet_judge.py#L582-L639)

The description is split into exact evidence spans.

For this case, the relevant spans included evidence such as:

```text
"take off on a forty day adventure into the ... Mojave Desert"

"a magical tale"

"a modern-day adventure and a metaphysical odyssey"
```

The judge records which exact spans are candidates for each facet.

This is important for auditability because the final judgment can later be traced back to specific supplied text rather than an unexplained LLM opinion.

---

### Step 3 — verify each facet/component

**Execution:** **Hybrid.** The LLM decides semantic grounding for one frozen component against one exact span; deterministic validation and guards can reject inconsistent or disallowed positives.

**Debug first:** Inspect `component_evidence_ledger_json`, especially `grounding_relation`, `negative_boundary_applied`, `external_knowledge_required`, `reason`, and `supporting_span_ids`.


**Code references:** [`build_isolated_component_prompt()`](../semantic_relevance_facet_judge.py#L2892-L2949), [`verify_isolated_component()`](../semantic_relevance_facet_judge.py#L2969-L3096), [`verify_ranked_candidates()`](../semantic_relevance_facet_judge.py#L3309-L3395)

Some facets are simple enough to verify directly.

Others are represented as semantic components.

#### F2 — magic

For `magic`, the supplied text contains an explicit lexical cue:

```text
"magical tale"
```

The judge therefore has a deterministic direct cue for this facet.

Observed r5 targeted result:

```text
F2 = meaningful / direct / substantive
```

This means:

- the evidence directly supports the facet
- the support is substantive to the book rather than incidental
- no external book knowledge is needed

---

### Step 4 — composite facets are broken into required components

**Execution:** **Deterministic composition over hybrid component results.** The LLM/hybrid verifier evaluates the individual components, but Python decides whether all required components are present.

**Debug first:** Identify the exact missing component. Do not debug the final 0–4 score yet; first ask why `movement_or_travel`, `danger_or_threat`, or another required component is missing.


**Code references:** [Q04 `required_components`](../facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json#L255-L280), [`_assemble_single_span_from_component_results()`](../semantic_relevance_facet_judge.py#L3097-L3167)

`dangerous journeys` is not treated as one fuzzy phrase.

It is represented as two required semantic components:

```text
movement_or_travel
+
danger_or_threat
```

Both components must be established.

This prevents a description containing only danger from being treated as a dangerous journey.

For example:

```text
"There is an explosion and the protagonist is threatened."
```

contains danger, but not necessarily travel.

Likewise:

```text
"They travel across the country."
```

contains movement, but not necessarily danger.

---

### Step 5 — LLM verification is constrained by deterministic guards

**Execution:** **Hybrid.** The LLM provides semantic verification; deterministic guards can reject, cap, or recover results when frozen boundaries or explicit textual anchors require it.

**Debug first:** Compare the supplied evidence with the post-guard component result. If the model reason is plausible but the guard changes it, debug the guard. If the guard is not involved and the semantic reason is unsupported, investigate LLM variance/prompting.


**Code references:** [`_q04_r1_movement_text_anchor_guard()`](../semantic_relevance_facet_judge.py#L2591-L2629), [`_q04_r1_recovery_movement_text_anchor_guard()`](../semantic_relevance_facet_judge.py#L2632-L2667)

This case exposed an important calibration defect.

In r3, the evidence contained:

```text
take off on a forty day adventure into the ... Mojave Desert
```

and:

```text
sometimes dangerous
```

The LLM correctly recognized the danger component.

However, the deterministic movement guard did **not** recognize the particular wording:

```text
take off on ... adventure into ...
```

as an explicit travel anchor.

The r3 component result therefore became approximately:

```text
movement_or_travel = missing
danger_or_threat   = entailed
```

Because both components are required:

```text
F3 dangerous journeys = absent
```

The resulting r3 full-run score was:

```text
Human = 4
Judge = 2
```

This was not treated as evidence that the human label was wrong.

Instead, the component trace exposed a specific implementation gap.

---

### Step 6 — repair the semantic guard, not the individual example

**Execution:** **Deterministic repair.** r4 changed a reusable text-pattern rule; it did not tell the LLM that a particular book should receive a particular score.

**Debug first:** Write positive and negative localized tests around the failure class. Confirm both the intended match and nearby false-positive boundaries before running expensive LLM regressions.


**Code references:** [`_Q04_R4_DIRECTIONAL_DEPARTURE_RE`](../semantic_relevance_facet_judge.py#L1878-L1884), [`_q04_r4_has_directional_departure_anchor()`](../semantic_relevance_facet_judge.py#L1887-L1889), [`_q04_r2_cross_span_movement_component_check()`](../semantic_relevance_facet_judge.py#L2671-L2702)

The r4 calibration change did **not** hard-code:

```text
The Valkyries = relevant
```

Instead, it added a general movement pattern for directional departure constructions such as:

```text
take off on a ... adventure into ...
take off on a ... journey through ...
took off on an ... expedition toward ...
```

The rule deliberately rejects unrelated uses such as:

```text
sales took off
the story takes off
he took off his coat
```

This is an important evaluation-engineering principle:

> Fix the semantic failure class, not the individual test case.

After that repair, the same supplied description establishes:

```text
movement_or_travel = established
danger_or_threat   = established
```

Therefore:

```text
F3 dangerous journeys = meaningful / entailed / substantive
```

---

### Step 7 — fantasy adventure is also evaluated compositionally

**Execution:** **Hybrid.** The component meanings are frozen; the LLM verifies semantic support; deterministic code validates and assembles the result.

**Debug first:** If repeated runs flip between `incidental` and `meaningful`, compare component/evidence traces across runs. That is a stability issue rather than automatically a scoring-rule defect.


**Code references:** [Q04 F1 component definition](../facets/semantic_relevance/semantic_relevance_query_facets.v0.10.1.json#L214-L236), [`evaluate_one_facet()`](../semantic_relevance_facet_judge.py#L4432-L4936)

The `fantasy adventure` facet itself combines ideas such as:

```text
fantasy_or_supernatural_setting
+
adventure / quest / journey / action-driven exploits
```

The judge evaluates whether the supplied text establishes those ideas without relying on outside knowledge of the book.

By the r5 targeted run, the observed facet result was:

```text
F1 = meaningful / entailed / substantive
```

The supplied evidence supporting the overall judgment included references to:

```text
magical tale
curse
guardian angel
forty day adventure
metaphysical odyssey
```

---

### Step 8 — deterministic scoring converts facet support into the 0–4 score

**Execution:** **Deterministic.** Once the facet assessments are frozen for that execution, the final numeric score is calculated by Python, not by a final LLM call.

**Debug first:** Read `scoring_explanation`, `meaningful_core_count`, `meaningful_core_coverage`, `strong_core_count`, `direct_core_count`, `qualifier_cap_applied`, and `clear_rule_applied`. If these inputs are right but the numeric score is wrong, the bug is in scoring code.


**Code references:** [`derive_facet_support()`](../semantic_relevance_facet_scoring.py#L518-L559), [`compute_facet_score()`](../semantic_relevance_facet_scoring.py#L586-L790), [runner hand-off from semantic verdict to deterministic scoring](../run_judge_v0_29_r5_development.py#L1373-L1394)

Once all facet assessments are complete, Python scoring — not a final free-form LLM opinion — calculates the ordinal score.

For the r5 targeted run, the facet state was:

```text
F1 fantasy adventure   = meaningful / entailed / substantive
F2 magic               = meaningful / direct / substantive
F3 dangerous journeys  = meaningful / entailed / substantive
```

The resulting judge output was:

```text
level = clear
score = 3
```

Human label:

```text
4
```

So this case is currently an **adjacent disagreement**:

```text
|4 - 3| = 1
```

It therefore:

- fails exact agreement
- passes within-one agreement
- is **not** a severe disagreement

This is a useful example of why the project tracks more than exact accuracy.

---

### Step 9 — how this one case affects the aggregate metrics

**Execution:** **Deterministic.** Human and judge scores are compared mechanically to update agreement/error metrics and gates.

**Debug first:** Verify the human label, judge score, and case identity being joined. A bad merge/provenance split can produce a wrong metric even when the underlying judge result is correct.


**Code references for the current targeted regression check:** [`semantic_relevance_v0.29_regression_manifest.v2.4.0.json`](../datasets/semantic_relevance_v0.29_regression_manifest.v2.4.0.json#L200-L207), [`analyze_v0_29_r5_targeted.py`](../analyze_v0_29_r5_targeted.py#L20-L69)

For this case:

```text
Human = 4
Judge = 3
```

Metric impact:

| Metric | Effect |
|---|---|
| Exact agreement | Fail |
| Within-one agreement | Pass |
| Absolute error | `1` |
| Severe disagreement | No |
| Binary relevance precision/recall | Correct positive classification |
| Weighted kappa | Small ordinal penalty |

Compare that with the old r3 result:

```text
Human = 4
Judge = 2
```

That result had:

```text
absolute error = 2
```

and therefore counted as a **severe disagreement**.

The r4 repair improved the failure from:

```text
4 -> 2
```

to:

```text
4 -> 3
```

without changing the human label.

---


### Score-debugging playbook — work backwards from the final score

When a score looks wrong, debug it as a pipeline rather than immediately editing the prompt.

#### 1. Confirm the case and final deterministic score

Open the case row in:

```text
<run-directory>\judge_results.csv
```

Start with:

```text
case_id
judge_score
match_level
judge_reason
scoring_explanation
```

If `judge_score` does not follow from `scoring_explanation`, investigate deterministic scoring/persistence before inspecting LLM reasoning.

#### 2. Inspect deterministic scoring inputs

Check:

```text
core_facet_count
incidental_core_count
meaningful_core_count
meaningful_core_coverage
strong_core_count
direct_core_count
entailed_core_count
adjacent_core_count
unsupported_core_count
qualifier_count
satisfied_qualifier_count
qualifier_cap_applied
clear_rule_applied
```

These are the inputs Python used to compute the final score.

Question to ask:

> Given these facet-support counts, is the 0–4 score mechanically correct?

If **no**, debug [`compute_facet_score()`](../semantic_relevance_facet_scoring.py#L586-L790).

If **yes**, continue backward.

#### 3. Find the facet that caused the score

Inspect:

```text
facet_assessments_json
```

For each facet look at:

```text
facet_id
facet_text
verification_relation
prominence
derived_support
winning_evidence_span_id
verification_reason
```

Question to ask:

> Which facet is unexpectedly absent, incidental, meaningful, or strong?

If the verification relation looks correct but `derived_support` is wrong, debug [`derive_facet_support()`](../semantic_relevance_facet_scoring.py#L518-L559).

#### 4. For a composite facet, inspect the component ledger

Inspect:

```text
component_evidence_ledger_json
```

Look for:

```text
component_id
grounding_relation
supporting_span_ids
negative_boundary_applied
external_knowledge_required
reason
```

For `dangerous journeys`, for example, explicitly check:

```text
movement_or_travel
danger_or_threat
```

Question to ask:

> Is one required component missing, and if so, why?

This is where the original Valkyries defect became obvious:

```text
movement_or_travel = missing
danger_or_threat   = entailed
```

#### 5. Verify that the right evidence was selected

Inspect:

```text
facet_evidence_json
evidence_text
```

Then compare the selected span IDs with the original description.

Question to ask:

> Did the relevant phrase exist in the supplied description but never reach the verifier?

If yes, debug evidence selection/full-context recovery.

If no, the label may be asking the judge to infer something not established by the supplied text.

#### 6. Identify who made the questionable decision

Use this rule of thumb:

```text
Wrong candidate span?            -> LLM/hybrid evidence selection
Wrong semantic entailment?       -> LLM/hybrid component verification
Correct LLM result rejected?     -> deterministic guard
Missing cross-span composition?  -> deterministic recovery/composition
Wrong facet support?             -> deterministic derive_facet_support
Correct facets, wrong 0-4 score? -> deterministic compute_facet_score
Correct score, wrong gate?       -> deterministic analyzer/provenance logic
```

This prevents wasting time tuning prompts when the actual bug is deterministic code — or adding deterministic rules to compensate for ordinary model variance.

#### 7. Decide whether rerunning the case is informative

A rerun is useful mainly when the disputed stage is **LLM-owned or hybrid**.

For example, the Q07 failure showed:

```text
3, 1, 1, 1, 3
```

across executions before the grounding guard was introduced.

That variation revealed a stochastic semantic-verification problem.

By contrast, if the same deterministic regex fails the same exact phrase every time, repeated LLM runs add little value; write a localized deterministic test instead.

#### 8. Classify the failure before repairing it

Use one of the project failure categories:

```text
human-label problem
rubric/spec ambiguity
deterministic implementation defect
LLM stochastic instability
unsupported inference
scoring defect
analysis-tooling/provenance defect
```

Only after classification should calibration change.

#### 9. Add the failure to regression protection

After a repair:

1. add a localized unit/contract test if deterministic logic changed
2. add/retain the real consumed case in the targeted regression manifest
3. run targeted regression
4. check neighboring positive and negative controls
5. only then consider a broader/full run

This is how one discovered failure becomes permanent evaluation coverage.

#### Useful PowerShell inspection snippet

For a known case and run:

```powershell
$run = ".\evals\runs\<run-directory>"
$case = "U_Q04_T10"

$row = Import-Csv "$run\judge_results.csv" |
  Where-Object { $_.case_id -eq $case }

$row | Select-Object `
  case_id,
  judge_score,
  match_level,
  judge_reason,
  scoring_explanation,
  meaningful_core_count,
  meaningful_core_coverage,
  strong_core_count,
  direct_core_count,
  unsupported_core_count,
  qualifier_cap_applied,
  clear_rule_applied |
  Format-List

"`nFACET ASSESSMENTS"
$row.facet_assessments_json |
  ConvertFrom-Json |
  ConvertTo-Json -Depth 30

"`nCOMPONENT LEDGER"
$row.component_evidence_ledger_json |
  ConvertFrom-Json |
  ConvertTo-Json -Depth 30

"`nCANDIDATE EVIDENCE"
$row.facet_evidence_json |
  ConvertFrom-Json |
  ConvertTo-Json -Depth 30
```

For interview purposes, the debugging strategy can be summarized as:

> **Start with the deterministic score explanation and walk backward through facet support, component grounding, and selected evidence until you reach the first incorrect decision. Then fix the owner of that decision — LLM behavior, deterministic logic, the frozen spec, the human label, or the analysis pipeline.**

---

### Step 10 — why the process is useful in practice

**Execution:** **End-to-end hybrid system.** LLM semantic reasoning is deliberately surrounded by deterministic evidence boundaries, composition, scoring, persistence, regression checks, and release logic.

**Debug first:** Work backwards from the final score until the first stage where the trace diverges from what the supplied evidence supports.


**Code references for the audit trail:** [`generate_validated_semantic_verdict()`](../semantic_relevance_facet_judge.py#L4939-L5015), [persisted facet/component/scoring diagnostics](../run_judge_v0_29_r5_development.py#L1437-L1605)

For an interview, this example illustrates why the evaluation system is more defensible than a simple one-shot LLM score.

The system can answer:

```text
What query facet failed?
Which exact evidence span was used?
Which semantic component was missing?
Was the conclusion produced by the LLM or by a deterministic rule?
Was a negative boundary triggered?
How did the facet states become the final score?
Does the resulting error count as adjacent or severe?
Did a later repair fix the general failure class?
```

That gives the judge a trace roughly like:

```text
Query
  ↓
Frozen query facets
  ↓
Candidate evidence spans
  ↓
LLM semantic verification
  ↓
Deterministic component/negative-boundary guards
  ↓
Facet support + prominence
  ↓
Deterministic 0–4 scoring
  ↓
Compare with human gold label
  ↓
Update metrics / regression gates / release decision
```

The key design idea is:

> The LLM performs bounded semantic inference, while deterministic code controls composition, known invalid inferences, scoring, regression expectations, and release decisions.

This separation makes failures diagnosable and makes calibration changes testable.

---

## 22. Key lessons demonstrated so far

### 1. High aggregate metrics are not enough

A system can have roughly:

```text
98% within-one agreement
94% precision
89% recall
```

and still contain severe semantic hallucinations.

### 2. Independent holdouts matter

r8 passed consumed-development gates but failed 5/8 independent final-holdout gates.

### 3. Non-deterministic systems require stability testing

`U4_Q07_T03` demonstrated:

```text
3
1
1
1
3
```

before the deterministic grounding guard was added.

### 4. Regression sets must grow with discovered failures

Every fixed failure should become evidence preventing future regressions.

### 5. Deterministic controls complement LLM reasoning

Use the LLM where semantic inference is valuable, but deterministic guards where particular inference classes are categorically invalid.

### 6. Human labels are also reviewable

Some severe disagreements were label/spec tensions. Human labels changed only through a separate recorded adjudication process.

### 7. Analysis tooling must itself be tested

During r3 full-run review, analysis tooling initially used incorrect schema/provenance assumptions.

The completed judge run remained valid.

The evaluator, datasets, lifecycle locks, analysis pipeline, and release logic all need tests — not only the model under evaluation.

---

## 23. Important files and runs to preserve

### Current development dataset

```text
evals\datasets\semantic_relevance_v0.29_development.v6.1.0.csv
SHA-256:
cfcc465497079944fec6da17f14bbe5f3350d9385ea9ddc646610b7751d72da7
```

### Current candidate artifacts

```text
evals\judge_configs\semantic_relevance_judge.v0.29.0-r5.json
evals\facets\semantic_relevance\semantic_relevance_query_facets.v0.10.1.json
evals\datasets\semantic_relevance_v0.29_regression_manifest.v2.4.0.json
evals\semantic_relevance_facet_judge.py
evals\semantic_relevance_facet_scoring.py
```

### Current targeted run

```text
evals\runs\semantic_relevance_v0_29_r5_development\20261002T091035Z_targeted_r5
```

### Most recent complete 240-case run

```text
evals\runs\semantic_relevance_v0_29_r3_development\20261001T135129Z
```

### Historical one-time r8 final holdout

```text
evals\runs\semantic_relevance_v0_28_final_holdout\20261001T044506Z_final_r8
```

Never rerun it for qualification.

---

## 24. Current candidate hashes

```text
Judge:
41b3b6e9650e7e4a1f1ad9693c2b0a145c844c3f813852d6c53ed8896ea9384a

Scoring:
8d40b3f773e3c763d9b6436ff036e63966ddc2b3877ecf4bf5f8989664d2587b

Facet spec 0.10.1:
083c88fe4de381e8c2e51e30af75e999c9d09cff6cfb03b134472acf881629a0

Judge config 0.29.0-r5:
47c2bc8013d34939a699b590f33b521ef3aa17074b288f3bfcf09e2892572275

Regression manifest 2.4.0:
095c222d05cffd18e7e5f2b41d5409cee143760182ccc11d77f0a84c9b920af6
```

Verify these before generating additional r5 evidence.

---

## 25. Resume checklist

```text
[ ] Read this report
[ ] Confirm current candidate is still v0.29.0-r5
[ ] Verify current candidate hashes
[ ] Confirm v0.29 independent evidence still does not exist
[ ] Confirm historical locks/finality markers remain unchanged
[ ] Do NOT rerun old independent holdouts
[ ] Run one full r5 240-case consumed-development pass
[ ] Analyze it with the corrected provenance model
[ ] Classify any new failures before tuning
[ ] If full gate passes, run stability matrix
[ ] If stability passes, freeze release candidate
[ ] Only then create fresh unseen v0.29 evaluation evidence
```

---

## 26. One-sentence current status

> **Semantic relevance calibration is paused at `v0.29.0-r5`: all 60 targeted consumed-development regression gates pass, the latest complete aggregate run (r3) shows 70.0% exact / 98.3% within-one / 0.897 quadratic kappa / 94.1% precision / 88.8% recall, but r5 still requires a full 240-case development run, stability qualification, candidate freeze, and fresh independent evaluation before it can be considered final-qualified.**
