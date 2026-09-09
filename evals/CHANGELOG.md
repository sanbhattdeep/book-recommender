# Evaluation Changelog

## Judge Config v0.5.0
Date: 2026-09-08

### Changed
- Added explicit evidence grounding for every positive relevance judgment.
- Added `evidence_text` containing an exact phrase from the supplied book description.
- Added `matched_concept` identifying the requested concept supported by the evidence.
- Added deterministic validation that `evidence_text` actually occurs in the supplied description.
- Added consistency validation between:
  - has_relevant_evidence
  - evidence_text
  - matched_concept
  - match_level
  - score

### Why
Judge Config v0.4.0 achieved 80% exact agreement on the 5-case
development set but still produced one speculative 0 -> 1 error.

The remaining failure occurred because the judge treated a possible
interpretation of personal struggle as evidence of redemption.

v0.5.0 requires every positive relevance judgment to be grounded in
specific text from the supplied description.

### Validation

5-case development smoke test:

- Exact agreement: 100% (5/5)
- Within-one agreement: 100% (5/5)

Results:

- Q01_R01: Human 3 -> Judge 3
- Q01_R05: Human 3 -> Judge 3
- Q01_R20: Human 0 -> Judge 0
- Q01_R50: Human 0 -> Judge 0
- Q01_NEG: Human 0 -> Judge 0

All structured verdict fields were internally consistent.

### Decision
Freeze Judge Config v0.5.0.

Do not tune further using these five development cases.

Evaluate v0.5.0 against the remaining 55 untouched holdout cases.

### Unchanged
- Rubric version: 0.1.0
- Calibration dataset version: 0.1.0
- Provider: Ollama
- Model: jeffnyman/ts-evaluator
- Temperature: 0.0

## Judge Config v0.4.0
Date: 2026-09-08

### Changed
- Replaced boundary-specific prompt patches with a single ordered
  five-level decision ladder:
  - none -> 0
  - incidental -> 1
  - partial -> 2
  - clear -> 3
  - strong -> 4
- Added structured verdict fields:
  - has_relevant_evidence
  - match_level
  - score
  - reason
- Added deterministic validation of match_level-to-score mapping.
- Added validation that has_relevant_evidence=False must map to
  match_level=none and score=0.

### Why
Judge Config v0.3.0 regressed badly because the 0-vs-1-specific
guardrails distorted interpretation of scores 2-4.

The new approach evaluates all five relevance levels using one
consistent decision framework.

### Validation

5-case development smoke test:

- Exact agreement: 80% (4/5)
- Within-one agreement: 100% (5/5)

Results:

- Q01_R01: Human 3 -> Judge 3
- Q01_R05: Human 3 -> Judge 3
- Q01_R20: Human 0 -> Judge 0
- Q01_R50: Human 0 -> Judge 0
- Q01_NEG: Human 0 -> Judge 1

### Observation
The decision ladder corrected the major regression seen in v0.3.0.

One remaining inconsistency exists in Q01_NEG. The judge returned
has_relevant_evidence=True based on reasoning that the description
"could be seen as" redemption, despite the judge configuration
explicitly stating that speculative language such as "could be seen
as" does not constitute positive evidence.

### Decision
Do not yet run the untouched 55-case holdout.

Create Judge Config v0.5.0 that requires the judge to identify the
specific supplied-text evidence and requested concept supporting any
has_relevant_evidence=True decision.

### Unchanged
- Rubric version: 0.1.0
- Calibration dataset version: 0.1.0
- Provider: Ollama
- Model: jeffnyman/ts-evaluator
- Temperature: 0.0

## Judge Config v0.3.0
Date: 2026-09-08

### Changed
- Added an explicit evidence-first decision rule before the judge may
  assign any score above 0.
- Score 1 now requires positive evidence that at least one requested
  concept, or a close semantic equivalent, is actually present in the
  supplied description.
- Explicitly prohibited hypothetical or speculative semantic overlap
  from justifying score 1.
- Added guidance that phrases such as:
  - "could be related"
  - "could be seen as"
  - "might represent"
  - "may imply"
  - "possibly reflects"

  are not sufficient evidence for semantic relevance.
- Added a consistency rule: if the judge's own reasoning says there is
  "no explicit evidence", "no meaningful evidence", or only a
  hypothetical connection, the score must be 0.
- Added a decision sequence for the 0-vs-1 boundary:

  1. Is there positive evidence for a requested concept or close
     semantic equivalent?
  2. If no -> score 0.
  3. If yes, but only incidental/superficial -> score 1.
  4. Otherwise consider scores 2-4 normally.

### Why
Judge Config v0.2.0 improved exact agreement from 40% to 60% on the
5-case development smoke set, but still produced two 0->1 errors.

In both remaining failures, the judge's reasoning acknowledged that
direct evidence for the requested themes was absent but still assigned
score 1 based on speculative interpretation.

The purpose of v0.3.0 is to make the scoring decision consistent with
the judge's own evidence assessment.

### Unchanged
- Rubric version: 0.1.0
- Calibration dataset version: 0.1.0
- Judge provider: Ollama
- Judge model: jeffnyman/ts-evaluator
- Temperature: 0.0
- Native scoring scale: 0-4

### Validation

5-case development smoke test:

- Exact agreement: 20% (1/5)
- Within-one agreement: 60% (3/5)
- Human 0 -> Judge 0: 1/3
- Human 0 -> Judge 1: 2/3
- Human 3 -> Judge 1: 2/2

### Observation

The evidence-first 0-vs-1 rules did not eliminate speculative
0-to-1 scoring and introduced a regression on higher relevance scores.

Both human-score-3 cases were downgraded to 1 despite the judge
identifying explicit evidence for redemption.

The additional 0-vs-1 instructions appear to have over-emphasized the
lowest scoring boundary and distorted interpretation of scores 2-4.

### Decision

Do not proceed to the 55-case holdout with Judge Config v0.3.0.

Create Judge Config v0.4.0 using a single ordered decision ladder
covering all five relevance levels rather than adding further
boundary-specific guardrails.

## Judge Config v0.2.0
Date: 2026-09-08

### Changed
- Added explicit guidance for the 0-vs-1 scoring boundary.
- Clarified that generic adjacent concepts such as change, hardship,
  recovery, self-improvement, or emotional experience must not be
  treated as semantic relevance unless explicitly connected to the
  user's requested theme.
- Added guidance not to infer a requested theme merely because it could
  plausibly occur in the story.

### Why
Judge Config v0.1.0 showed systematic score inflation on the initial
5-case development smoke test:

- Human score 0 -> Judge score 1 on 3/3 zero-score cases.

### Smoke-test result
5 development cases:

- Exact agreement: 60% (3/5)
- Within-one agreement: 100%
- Human 0 -> Judge 0: 1/3 cases
- Human 0 -> Judge 1: 2/3 cases

The new guidance corrected Q01_R50.

However, Q01_R20 and Q01_NEG were still scored as 1 even though the
judge's own explanations described the connection as unsupported or
speculative.

Examples from the judge reasoning included:

- "there is no explicit evidence"
- "could be seen as a form of redemption"

### Decision
The remaining failures indicate that the judge understands that direct
evidence is absent but still treats hypothetical semantic connections
as sufficient for score 1.

Create Judge Config v0.3.0 with an explicit evidence-first decision rule
for the 0-vs-1 boundary.

### Unchanged
- Rubric version: 0.1.0
- Calibration dataset version: 0.1.0
- Judge provider: Ollama
- Judge model: jeffnyman/ts-evaluator

## Judge Config v0.1.0
Date: 2026-09-08

### Added
- Initial semantic relevance judge configuration.
- Ollama provider.
- Model: jeffnyman/ts-evaluator.
- Native 0-4 structured scoring.
- Blind evaluation using query, title, authors, and description only.


## Calibration Dataset v0.1.0
Date: 2026-09-07

### Added
- Initial 60-case human-labelled semantic relevance calibration dataset.
- 12 queries x 5 candidate books.
- Human scores on the 0-4 rubric scale.
- Human reasons for each label.


## Rubric v0.1.0
Date: 2026-08-31

### Added
- Initial Semantic Recommendation Relevance rubric.
- Native 0-4 scale.
- Anchor examples for each score.
- Boundary examples for adjacent score decisions.
- Tie-break rule.