# Evaluation Changelog

# Semantic relevance judge v0.21.1

v0.21.1 is a targeted correction to the v0.21.0 role/precheck architecture. Frozen facet spec v0.8.0, rubric v0.1.0, and deterministic 0-4 scoring remain unchanged.

Changes:

- Keep the v0.21 full-description hard-exclusion precheck before positive candidate selection.
- Treat hard exclusions as semantic guards rather than lexical vetoes; genuine positive facet evidence is not suppressed merely because exclusion-associated words occur.
- Replace monolithic role output with decomposed signals and deterministic Python resolution.
- Resolve `is_substantively_examined=true` ahead of `is_background_cause_or_factor=true`, preventing background-cause from suppressing genuine substantive treatment.
- Keep meta-discussion and example/illustration as strong incidental signals.
- Clarify fiction semantics: world-defining, plot-driving, character-arc, or consequential content may be substantive without academic analysis.
- Clarify background-cause semantics: a facet is background only when it primarily explains some OTHER main subject/event/situation.
- Add Q11/Q12 positive controls to the targeted gate.

No title-, author-, ISBN-, or case-specific judge behavior is added.

# v0.21.0

Structural correction for the remaining Q11/Q12 v0.20 over-promotions.

- Added full-description hard-exclusion precheck before every positive stage.
- Replaced single LLM `context_role` classification with five independent,
  source-grounded role signals.
- Added deterministic Python role precedence:
  meta/example/background → incidental; primary → central;
  substantive → substantive; otherwise → incidental.
- Added precheck and role-signal audit serialization.
- Preserved frozen facet spec v0.8.0, rubric v0.1.0, inference contract,
  support derivation, and final deterministic 0–4 scoring.
- Retained all seven v0.20 regression gates.
- Added Q11 and Q12 positive controls to prevent blanket suppression.
- Added deterministic contract tests for precheck ordering, signal precedence,
  source-span grounding, positive-control registration, and serialization.

# v0.20.0

Behavioral stabilization after v0.19 targeted regression failure.

- Added first-class hard-exclusion contract.
- Tightened Q02 suspense and Q05 political-conflict boundaries.
- Removed Q11 context-insensitive adventure/quest deterministic cue.
- Added Q11 meta-literary/symbolic hard exclusions.
- Reworked core prominence as primary-subject-first context-role classification with Python-only prominence derivation.
- Fixed v0.19 audit serialization omission for inference/context fields.
- Preserved Q04 within-span semantics, Q06 composite recovery, Q10 polarity guard, and deterministic 0-4 scoring.
- Rubric remains v0.1.0 and is fully populated.

# v0.19.0 changelog

## Behavioral changes

- Added structured `inference_kind` to single-span and composite verification.
- `ENTAILED` is mechanically restricted to `necessary_semantic_inference`.
- Configured possibility-language backstop rejects speculative ENTAILED reasons.
- Added structured `context_role` to core prominence.
- Background causes, examples/illustrations, meta-discussion, and incidental mentions deterministically map to `INCIDENTAL` prominence.
- Q05 war boundary now blocks terrorism/violent-mission => war without established warfare.
- Q11 boundaries distinguish narrative/subject content from literary-analysis examples and symbolic speculation.
- Q04 definitions allow within-scenario composition of fantastical movement plus explicit threat/danger.

## Non-behavioral / packaging repair

- Restored complete rubric v0.1.0 JSON; previous generated package contained only the version field.
- Added a rubric integrity regression test.
- Added a deterministic builder for the 90-case consumed-development dataset; it never reads the locked holdout.

## Unchanged

- Rubric semantics/version: 0.1.0
- Final 0–4 aggregation
- Qualifier caps
- v0.16 polarity guard
- v0.18 full-context composite architecture
- Ollama model/provider/temperature

# v0.18.0 changelog

## Changed: composite architecture

Removed the separate full-description composition selector introduced in v0.17.

The composite verifier now sees the full numbered description and performs span
selection and relation classification in one isolated call.

This prevents a useful span from being excluded because an upstream selector
failed to pass it forward.

## Changed: ENTAILED vs ADJACENT contract

Composite `ENTAILED` now explicitly means:

- all required semantic components are supplied by the cited spans;
- the facet follows through a short necessary inference;
- exact facet wording is not required;
- no single span needs to state the whole concept.

Composite `ADJACENT` now requires
`missing_semantic_component` to name the specific required component that is
still absent or only plausible.

## Added audit fields

Per facet:

- `composite_verification_attempted`
- `composite_evidence_span_ids`
- `composite_verification_relation`
- `composite_combined_evidence_summary`
- `composite_missing_semantic_component`
- `composite_verification_reason`

## Preserved

- facet spec v0.6.0
- rubric v0.1.0
- v0.16 Q02 suspense boundary
- polarity-aware deterministic cues
- standalone evidence selection/verification
- prominence behavior
- composite ceiling = ENTAILED
- deterministic support derivation
- deterministic final scoring
- model/provider/temperature

# v0.17.1 changelog

## Fixed

Composition-selector structural noncompliance no longer aborts an otherwise
valid evaluation case.

The selector still requires either:

- 2-4 unique real span IDs, or
- exactly `["NONE"]`.

After bounded repair attempts, continued structural invalidity now falls back
to `NONE/no-composition`.

## Unchanged

- v0.17 independent full-description composition-selection architecture
- Q02 suspense semantics
- polarity-aware deterministic cues
- facet spec v0.6.0
- verifier relation semantics
- prominence
- support derivation
- final 0-4 aggregation
- model/provider/temperature

This patch changes fault tolerance for malformed optional-stage output only; it
does not change valid-output semantic behavior.

# v0.17.0 changelog

## Judge Config v0.17.0

### Added

A dedicated composition-specific evidence selector over the full numbered book
description.

The new selector runs only for core facets whose best standalone relation is
`unsupported` or `adjacent`. It chooses the smallest complementary 2-4 span
set and is intentionally independent of Stage-A standalone ranking.

### Why

v0.16 proved that conservative multi-span verification could execute yet still
miss a facet because Stage A omitted a span that was weak alone but important
when combined with an earlier event/impact span.

The architecture therefore changes from:

```text
Stage-A candidates
      ↓
composite verifier
```

to:

```text
Stage-A standalone path
      ↓
weak result
      ↓
full-description composition selector
      ↓
composite verifier
```

### Preserved

- facet spec v0.6.0
- rubric v0.1.0
- Q02 suspense boundary
- polarity-aware deterministic cues
- isolated verifier semantics
- core prominence semantics
- composite relation ceiling = ENTAILED
- deterministic support derivation
- deterministic final scoring
- model/provider/temperature

## Audit additions

Per-facet:

- `composite_selection_attempted`
- `composite_candidate_span_ids`
- `composite_selection_reason`
- existing positive `composite_evidence_span_ids`
- existing composite relation/reason

Per-case:

- `composite_selection_attempt_count`
- `composite_verification_attempt_count`
- `composite_verification_count`

This distinguishes selector invocation, verifier invocation, and successful
positive composite recovery.

# v0.16.0 changelog entry

## Judge Config v0.16.0

### Added — polarity-aware deterministic DIRECT cue guard

Deterministic cue matches are now checked for explicit local negation, absence,
or contrast before they can establish `verification_relation=direct`.
High-precision patterns cover constructions such as `no X`, `without X`,
`lack of X`, `instead of X`, `rather than X`, and `X is absent/lacking`.

A suppressed cue does not make the facet absent. It falls through to Stage A so
the semantic pipeline can still evaluate the surrounding meaning.

### Added — core-only multi-span composition recovery

After Stage-A selection and isolated single-span verification, v0.16 may run a
new composition verifier when:

- the facet is core;
- at least two candidate spans were selected; and
- the strongest single-span relation is `unsupported` or `adjacent`.

The composition verifier can use only 2–3 selected exact spans and may return
`unsupported`, `adjacent`, or `entailed`. `direct` is mechanically rejected.
Positive composite output must identify at least two contributing source-span
IDs. The composed relation replaces the single-span relation only when it is
strictly stronger.

### Added — audit fields

- `deterministic_cue_polarity_blocked_count`
- `composite_verification_count`
- per-facet composite span IDs / relation / reason in `facet_assessments_json`

### Preserved

- deterministic final scoring
- v0.14 two-core aggregation guard
- support derivation
- qualifier behavior
- model/provider/temperature
- rubric v0.1.0
- original calibration provenance v0.5.0

---

## Facet Spec v0.6.0

### Changed — Q02 F1 `suspense`

Clarified that suspense requires story-level narrative tension / anticipation.
A difficult choice, moral dilemma, illness, deadline, interpersonal conflict,
high stakes, danger, or generic uncertainty is not sufficient by itself.

### Preserved

All query decompositions and every non-Q02-F1 semantic definition remain
unchanged from v0.5.0.

---

## Development Dataset v1.0.0

Created `semantic_relevance_v0.16_development.v1.0.0.csv` by combining the
previously consumed 30-case validation split and previously consumed 30-case
final holdout.

This 60-case set is explicitly development/regression data. Results on it are
not unseen generalization evidence.

# v0.15.0 changelog entry

## Facet Spec v0.5.0

### Changed

Only Q03 semantic-definition boundaries changed; facet decomposition is
unchanged.

`personal growth` now explicitly distinguishes actual personal development from
nearby spiritual/devotional concepts. Prayer, faith, spirituality, devotion,
character, virtues, wisdom, knowing a deity more deeply, or discovering how a
deity/external force works in a person do not themselves establish personal
growth. Actual development/change must be independently stated or necessarily
entailed by the exact evidence.

`finding purpose` now requires the exact evidence itself to establish meaning,
life direction, vocation/calling, goals, or values guiding life choices.
Spiritual understanding or discovering how a deity/external force works in
someone is insufficient unless that required connection is independently
present in the evidence.

### Why

The retained v0.14 targeted run fixed the aggregation failure but
`U_Q03_T70` still produced human 1 -> judge 3.

The trace showed that the verifier read an exclusion-with-exception as
permission to invent the exception condition:

```text
excluded near-concept
    -> assumed downstream self-development / purpose
    -> ENTAILED
```

The new definitions express generic semantic boundaries and contain no
candidate title or case ID.

---

## Judge Config v0.15.0

### Changed

Strengthened generic verifier precedence:

- explicit exclusions and exception clauses are hard gates;
- if a definition says `X does not establish the facet unless Y`, Y must be
  stated or necessarily entailed by the exact candidate evidence;
- `unless` / `only when` / `provided that` cannot be satisfied by assuming a
  plausible downstream benefit, likely outcome, interpretation, or real-world
  association;
- semantic-definition content is specification, not evidence;
- ENTAILED explicitly excludes merely plausible consequences/associations.

Stage-A ranking wording now clarifies that terms appearing only in
negative/exclusion clauses are not positive literal matches. They can still be
selected as lower-priority candidates so recall is preserved.

### Preserved

- calibration dataset v0.5.0
- rubric v0.1.0
- model `jeffnyman/ts-evaluator`, temperature 0
- deterministic DIRECT cue rules
- top-3 candidate architecture
- prominence stage
- support derivation
- qualifier behavior
- v0.14 two-core aggregation guard
- final 0-4 scale

# v0.14.0 changelog entry

## Calibration Dataset v0.5.0

### Changed

Blind re-adjudication of `Q12_R05` from human score `2` to `1` is now captured
in the versioned development dataset. All 60 rows carry `dataset_version=0.5.0`.

New reason:

> The description relates to inequality through its discussion of equality and social justice, but this connection is incidental to the requested focus on inequality, poverty, and wealth distribution. Poverty and wealth distribution are not established.

Historical runs remain pinned to the dataset version recorded in their own run
metadata.

---

## Facet Spec v0.4.0

### Changed

Only Q03 semantic-definition boundaries changed; facet structure is unchanged.

- `personal growth`: spiritual/devotional/philosophical/character/wisdom content
  alone is insufficient unless actual development, improvement, change, or
  increased self-understanding is established.
- `finding purpose`: discovering how a deity, belief system, external force, or
  circumstance works in a person's life alone is insufficient unless connected
  to meaning, direction, vocation, values, goals, or life direction.

### Why

v0.13 unseen validation produced a severe over-promotion on
`U_Q03_T70` by treating generic devotional discovery as both personal growth and
finding purpose. The validation set is now development data for v0.14 tuning;
the added boundaries are intentionally generic and candidate-independent.

---

## Judge Config v0.14.0

### Changed

For exactly two core facets, the historical `one strong facet => score 3`
exception now requires **no absent core facet**.

```text
strong + absent      -> 2
strong + incidental  -> 3 (guarded two-core exception)
strong + meaningful  -> 3 (normal two-thirds coverage)
strong + strong      -> 4
```

The audit rule name changes from `two_core_one_strong` to
`two_core_one_strong_no_absent` when the guarded exception fires.

### Why

Two independent validation cases showed the same structural failure:

- `U_Q03_T30` Learned Optimism: human 1, judge 3
- `U_Q01_T02` Lila's Child: human 2, judge 3

In both, one core facet was strong and the other completely absent. Calling that
an overall clear match was too permissive for a two-part query.

### Preserved

- rubric v0.1.0
- Ollama model `jeffnyman/ts-evaluator`, temperature 0
- deterministic positive-only DIRECT cue layer
- Stage-A candidate selection
- isolated precedence verifier
- core prominence stage
- support derivation
- qualifier behavior
- final 0-4 scale

## Calibration Dataset v0.4.0

### Changed

Re-adjudicated `Q02_NEG` (`Son of a Witch`) from human score `0` to `1`.

New reason:
> The description contains incidental suspense through unresolved questions, danger, and uncertainty, but crime and investigation are not established. The overlap is too weak for the book to be a useful match to the overall request.

All rows carry `dataset_version = 0.4.0`. Existing run metadata is not rewritten.

### Why

Blind re-review against the frozen rubric found genuine but incidental suspense; crime and investigation remain absent. This fits score 1 better than score 0.

---

## Judge Config v0.13.0

### Changed

Added a deterministic, positive-only, high-precision DIRECT cue layer before Stage A. Cue matches are grounded in exact description spans and are defined from frozen query/facet semantics rather than individual candidate books.

A cue may establish only `verification_relation=direct`. Core prominence still goes through the existing LLM prominence stage, and final support/0-4 scoring remains deterministic Python. When no cue fires, the v0.12 Stage-A -> isolated verifier -> prominence pipeline runs unchanged.

Added `deterministic_direct_cue_count` to run results and deterministic cue reasons to per-facet audit output.

### Why

v0.12 reduced severe errors but still repeatedly missed obvious lexical/concrete evidence despite explicit prompt instructions, notably `orphaned` -> loss and `introduction to astronomy` -> astronomy subject matter. This indicated diminishing returns from further prompt wording.

The cue layer handles only high-confidence lexical/concrete cases and leaves nuanced inference to the isolated LLM verifier.

### Preserved

- rubric v0.1.0
- facet spec v0.3.0
- adjacent semantics
- qualifier behavior
- candidate precedence for LLM-selected evidence
- prominence semantics
- deterministic final 0-4 aggregation

## Calibration Dataset v0.4.0

### Changed

Re-adjudicated `Q02_NEG` (`Son of a Witch`) from human score `0` to `1`.

Previous reason:
> no match to any theme or plot aspects

New reason:
> The description contains incidental suspense through unresolved questions,
> danger, and uncertainty, but crime and investigation are not established.
> The overlap is too weak for the book to be a useful match to the overall request.

### Why

A blind re-review against the frozen semantic relevance rubric found genuine but
incidental suspense in the supplied description. Crime and investigation remain
unsupported. Under the rubric, this is better represented by score 1 (weak /
incidental relevance) than score 0 (no meaningful relevance).

No candidate identity, query, description, rubric version, or other human gold
label was changed.

All rows now carry `dataset_version = 0.4.0`.

### Provenance

Existing judge runs that used dataset v0.3.0 remain recorded as v0.3.0 runs.
Their metadata must not be rewritten. Any comparison against v0.4.0 is a
post-run re-analysis against the re-adjudicated gold dataset.

## Judge Config v0.12.0
Date: 2026-09-11

### Changed

Added explicit candidate-ranking and verification-precedence contracts while preserving the v0.11 isolated multi-candidate architecture.

Stage A now ranks evidence candidates using:

1. literal / lexical / morphological facet evidence
2. unmistakable concrete instances
3. full-facet entailment
4. specific adjacent connections
5. indirect context

The isolated verifier now applies the relation decision in a fixed order:

A. excluded-near-concept / wrong-sense gate
B. direct
C. entailed
D. adjacent
E. unsupported

Added an explicit guardrail that a concept rejected by the frozen semantic definition cannot be promoted to `adjacent` when that excluded near-concept is the evidence's only connection to the facet.

Strengthened direct-recognition guidance for ordinary lexical and concrete instances, including:
- orphaned -> loss
- drug pusher -> crime
- fingerprints / who-why pursuit -> investigation
- introduction to astronomy -> astronomy subject matter

Strengthened Stage-A guidance for relational and process facets so directly relational/action-oriented spans outrank unrelated backstory.

Clarified that multiple facets may simultaneously be central when they are defining or recurring organizing elements.

### Preserved

No change to:
- dataset v0.3.0
- rubric v0.1.0
- facet spec v0.3.0
- multi-candidate maximum of 3
- deterministic candidate precedence: direct > entailed > adjacent > unsupported
- adjacent-core -> incidental support mapping
- qualifier semantics introduced in v0.11
- deterministic core-coverage aggregation
- final 0-4 scoring rules

### Why

The v0.11 20-case diagnostic successfully made score 1 reachable and reduced false zeros, but exposed three remaining systematic failures:
- Stage A sometimes selected indirect backstory instead of direct relational evidence.
- The verifier sometimes downgraded obvious lexical/concrete instances such as `orphaned`, `pusher`, investigative actions, and `introduction to astronomy`.
- `adjacent` sometimes bypassed explicit exclusion boundaries, allowing generic adaptation to count weakly toward redemption and generic injustice to count weakly toward social/economic inequality.

v0.12.0 addresses these failure classes without redesigning the pipeline or changing deterministic aggregation.

### Validation

Run:
- `uv run python evals/test_facet_scoring.py`
- `uv run python evals/test_evidence_candidate_selection.py`
- `uv run python evals/test_semantic_definition_prompts.py`

Then rerun the frozen 20-case diagnostic slice and assess against `v0.12.0_acceptance_checklist.md`.

## Calibration Dataset v0.3.0
Date: 2026-09-10

### Changed

Re-adjudicated Q09_R50 under the description-only evidence policy.

- human_score: 2 -> 1
- classification changed from partial relevance to incidental relevance

Reason:

The supplied description provides a genuine but weak semantic connection
through imprisonment and a journey to freedom. It does not provide sufficient
description-grounded evidence for authoritarian control, a dystopian setting,
or explicit political resistance.

All other candidate identities and human labels remain unchanged.

## Judge Config v0.9.3
Date: 2026-09-10

### Changed

Updated calibration dataset provenance:

- dataset v0.2.0 -> v0.3.0

No changes were made to:

- evidence candidate selection
- isolated evidence verification
- prominence assessment
- qualifier handling
- support derivation
- deterministic 0-4 aggregation
- model or temperature

Judge behavior is therefore unchanged from v0.9.2.

## Judge Config v0.9.2
Date: 2026-09-10

### Changed

Replaced the single-candidate evidence-selection stage introduced in v0.9.0
with multi-candidate evidence retrieval.

For each frozen facet, Stage A can now return up to three ranked candidate
description spans rather than exactly one candidate.

The pipeline is now:

```text
frozen facet
    ↓
select up to 3 candidate evidence spans
    ↓
verify each candidate independently
    ↓
Python selects strongest verified candidate
    ↓
assess prominence for supported core facet
    ↓
Python derives facet support
    ↓
Python computes overall 0-4 score

## Judge Config v0.9.0
Date: 2026-09-10

### Changed

Redesigned facet-level semantic evaluation to isolate evidence verification
from the broader query and description context.

v0.8.0 demonstrated that separating semantic relation from prominence improved
interpretability, but the judge continued to over-classify requested concepts
as `explicit` or `entailed`.

The v0.9.0 architecture therefore introduced an isolated per-facet pipeline:

```text
frozen facet
    ↓
candidate evidence selection
    ↓
isolated facet-vs-evidence verification
    ↓
core-only prominence assessment
    ↓
Python derives facet support
    ↓
Python computes overall 0-4 score

## Judge Config v0.8.0
Date: 2026-09-10

### Changed

Judge v0.8.0 redesigned facet-level semantic judgment to separate semantic
relationship from prominence.

The LLM no longer directly assigns:

```text
absent
incidental
meaningful
strong

## Judge Config v0.7.2
Date: 2026-09-10

### Changed

Judge v0.7.2 added a structured-output recovery mechanism for local-model
failures observed during the v0.7.1 full development run.

Semantic judging behavior remained unchanged from v0.7.1.

Deterministic score aggregation remained unchanged from v0.7.0.

The purpose of v0.7.2 was execution reliability rather than another semantic
or scoring redesign.

### Added Per-Facet Semantic Fallback

The normal semantic path continues to request all frozen facet assessments in
one structured response.

Recovery behavior is:

```text
batch semantic verdict
        |
        | invalid
        v
batch repair attempt 1
        |
        | invalid
        v
batch repair attempt 2
        |
        | invalid
        v
per-facet fallback


```markdown

## Judge Config v0.7.1
Date: 2026-09-09

### Changed

Judge v0.7.1 is a targeted semantic patch to v0.7.0.

The following remain unchanged:

- frozen facet specification
- facet support scale
- deterministic scoring algorithm
- `two_core_one_strong` rule
- two-thirds coverage rule
- qualifier behavior
- evidence span selection
- model
- provider
- temperature
- rubric
- calibration dataset

The change is limited to how the semantic judge handles polarity.

### Added Negative-Polarity Guidance

The judge is explicitly instructed not to confuse:

```text
concept is absent from the story

## Judge Config v0.7.0
Date: 2026-09-09

### Changed

Judge v0.7.0 revised the deterministic aggregation rules introduced by the
facet-based v0.6 architecture.

The semantic facet architecture remains unchanged:

1. Frozen query facets are loaded for each query.
2. The LLM independently assesses each facet as:
   - absent
   - incidental
   - meaningful
   - strong
3. Python computes the final 0-4 score.
4. Grounded evidence is selected only after semantic support is frozen.

The primary v0.7.0 change is how partial facet coverage maps to score 3.

### Added Two-Core Strong Exception

The original v0.6 scoring rule required at least two-thirds of core facets to
reach `meaningful` or `strong` support before score 3 could be assigned.

For a two-core query this meant:

- 1/2 supported = 50%
- required coverage = 2/2

This made the aggregation too strict for cases where one requested concept was
missing but another was directly and centrally satisfied.

The rubric explicitly permits score 3 when:

> the recommendation is clearly relevant overall, but one important aspect of
> the request is secondary, implicit, weaker, or missing.

v0.7.0 therefore added the following narrow rule:

- if a query has exactly TWO core facets
- and at least ONE core facet is `strong`
- the overall score may be 3 even if the other core facet is absent

Example:

```text
forgiveness -> absent
redemption  -> strong

overall -> 3

## Judge Config v0.6.1
Date: 2026-09-09

### Changed

Tightened the semantic facet-support criteria introduced in Judge Config
v0.6.0.

The facet architecture, frozen facet specification, deterministic scoring
algorithm, and source-span evidence mechanism remain unchanged.

The change is limited to the LLM semantic-assessment behavior.

### Added Existence Gate

Before assigning:

- incidental
- meaningful
- strong

the judge must first answer:

> Is this facet itself, or a close semantic equivalent, genuinely supported by
> the supplied description?

If the answer is no, the facet must be classified as:

`absent`

The existence gate is evaluated independently for every frozen facet.

### Clarified Incidental Support

`incidental` no longer means:

> somewhat related to the requested concept

It now requires that the requested facet itself, or a close semantic
equivalent, is genuinely present but peripheral or brief.

The following are explicitly insufficient:

- thematic adjacency
- causal association
- broad similarity
- plausible but unstated themes
- concepts that commonly co-occur with the requested facet

Examples added to the judge instructions include:

- personal growth is not redemption
- adapting to change is not forgiveness
- relationship difficulty is not forgiveness
- recovery is not redemption
- crime is not investigation
- poverty is not wealth distribution

These relationships may be related in ordinary language, but they do not by
themselves establish the requested semantic facet.

### Added Facet Isolation Rule

Evidence or reasoning for one facet must not automatically be reused as
evidence for another facet.

For example:

`forgiveness`

and:

`redemption`

are related concepts, but evidence supporting redemption does not automatically
establish forgiveness.

Each facet must independently pass the semantic existence gate.

### Clarified Semantic Entailment

Reasonable implicit support remains allowed.

The description does not need to contain the exact wording of the facet.

However, implicit support must entail the facet itself.

The judge must not promote a merely adjacent concept using reasoning such as:

- "could be related to"
- "could be seen as"
- "in a broad sense"
- "may imply"
- "is similar to"

unless the supplied description actually supports the requested facet.

### Unchanged

The following remain unchanged from v0.6.0:

- Calibration dataset: `v0.2.0`
- Rubric: `v0.1.0`
- Facet specification: `v0.1.0`
- Provider: Ollama
- Model: `jeffnyman/ts-evaluator`
- Temperature: `0.0`
- Evaluation unit: one query × one recommended book
- Facet support scale:
  - absent
  - incidental
  - meaningful
  - strong
- Deterministic 0-4 scoring algorithm
- Two-thirds core-facet threshold for score 3
- Qualifier cap
- Lower-score tie-break rule
- Source-span evidence selection
- Python-owned final score
- Blind evaluation
- Per-case persistence
- Resume support
- Run provenance tracking

### Why

The v0.6.0 Q01 smoke test produced:

- Q01_R20: human 0 -> judge 1
- Q01_R50: human 0 -> judge 1
- Q01_NEG: human 0 -> judge 1

Inspection showed that the judge was interpreting semantic adjacency as
incidental relevance.

This is different from the v0.5 problem.

v0.5 was too binary and frequently converted partial matches directly to 0.

The goal of v0.6.1 is therefore NOT to make the judge more conservative in
general.

The goal is to distinguish:

genuine-but-peripheral support
    -> incidental

from:

related but non-equivalent concept
    -> absent

This preserves score 1 for legitimate incidental matches while preventing
thematically adjacent concepts from receiving positive relevance.

### Validation Status

Development validation pending.

The first validation step is another five-case Q01 smoke test:

`uv run python evals/run_judge_calibration.py --limit 5`

The expected behavior is not hard-coded as a required result, but the primary
question is whether the previous false-positive incidental judgments for:

- Q01_R20
- Q01_R50
- Q01_NEG

are corrected without destroying genuine positive matches for:

- Q01_R01
- Q01_R05

If Q01_R01 or Q01_R05 changes because one of the two frozen facets does not
independently pass the tighter existence gate, that should be investigated as
a possible deterministic scoring/facet-coverage issue rather than immediately
loosening the semantic prompt.

### Decision

Judge Config v0.6.1 remains in development pending the Q01 smoke test.

Do not run the complete 60-case development dataset until the five-case smoke
result has been inspected.

## Judge Config v0.6.0
Date: 2026-09-09

### Changed

Judge v0.6.0 introduced a new facet-based semantic relevance architecture.

The previous v0.5.x design asked the LLM to make a holistic relevance judgment
and directly choose one of:

- none
- incidental
- partial
- clear
- strong

v0.6.0 separates semantic reasoning from final scoring.

The new evaluation pipeline is:

1. Decompose each query into a frozen set of semantic facets.
2. Ask the LLM to assess each facet independently.
3. Represent facet support using:
   - absent
   - incidental
   - meaningful
   - strong
4. Compute the final 0-4 relevance score deterministically in Python.
5. Extract grounded evidence only after semantic support and the final score
   have already been decided.

The LLM no longer directly assigns the overall numeric relevance score.

### Added

Added frozen query facet specification:

`evals/facets/semantic_relevance/semantic_relevance_query_facets.v0.1.0.json`

The facet specification:

- contains frozen decompositions for Q01-Q12
- is derived from query text rather than individual candidate books
- distinguishes:
  - `core` facets
  - `qualifier` facets
- is reused unchanged for every candidate belonging to the same query

Examples:

Q10:

- leadership — core
- teamwork — core
- building effective organizations — core

Q08:

- astronomy and the universe — core
- beginner-friendly explanation — qualifier

Q07 is intentionally represented as the single relational concept:

- complicated parent-child relationship — core

rather than splitting "complicated relationship" and "parent and child" into
independent facets.

### Added Deterministic Scoring

Added:

`evals/semantic_relevance_facet_scoring.py`

Python now owns the mapping from facet support to the final rubric score.

Base scoring rules:

- Score 0 — None
  - all core facets are absent

- Score 1 — Incidental
  - at least one core facet is incidental
  - no core facet is meaningful or strong

- Score 2 — Partial
  - at least one core facet is meaningful or strong
  - fewer than two-thirds of core facets are meaningful or strong

- Score 3 — Clear
  - at least two-thirds of core facets are meaningful or strong
  - but all core facets are not strong

- Score 4 — Strong
  - all core facets are strong

Qualifier rule:

- qualifiers cannot promote a weak core match
- if the provisional score is 4 and any qualifier is below meaningful,
  the score is capped at 3

The existing lower-score tie-break principle remains:

> Choose the lower score unless the conditions for the higher score are
> clearly satisfied.

### Added Facet-Level Judge

Added:

`evals/semantic_relevance_facet_judge.py`

The LLM now returns one semantic assessment per frozen facet.

The structured semantic verdict deliberately contains no final numeric score.

Each facet receives:

- facet_id
- support
- reason

The runner validates that:

- every frozen facet is returned
- each facet is returned exactly once
- no facets are invented
- no facets are omitted

### Evidence Grounding

The first v0.6.0 smoke implementation asked the LLM to copy an exact evidence
excerpt for each positive facet.

This exposed a reliability problem.

For Q01_R50 the semantic stage completed, but the evidence extractor repeatedly
returned the paraphrased sentence:

> "The book's focus is more on personal growth and adaptation to change."

The sentence did not occur verbatim in the supplied description, so Python
correctly rejected the evidence and the run failed.

Increasing extraction retries was not considered a satisfactory solution
because the problem was architectural rather than transient.

Evidence grounding was therefore changed to source-span selection.

Python now:

1. splits the supplied description into exact numbered source spans
   such as S1, S2, S3
2. gives those source spans to the LLM
3. asks the LLM to select one span ID for each non-absent facet
4. retrieves the exact source text itself

The LLM therefore never generates or paraphrases persisted evidence text.

Evidence remains grounded by construction.

The evidence stage is still not allowed to change:

- facet support
- facet reasoning
- final numeric score

### Runner Changes

`evals/run_judge_calibration.py` was updated for the v0.6 architecture.

The runner preserves:

- blind evaluation
- version pinning
- `--limit`
- `--resume`
- per-case persistence
- run metadata
- Git SHA provenance
- Ollama model configuration

It additionally records:

- facet spec version
- facet scoring module
- facet judge module

The result CSV retains compatibility fields:

- case_id
- has_relevant_evidence
- evidence_text
- matched_concept
- match_level
- judge_score
- judge_reason

and adds v0.6 audit fields:

- core_facet_count
- meaningful_core_count
- meaningful_core_coverage
- strong_core_count
- qualifier_count
- qualifier_cap_applied
- scoring_explanation
- facet_assessments_json
- facet_evidence_json

### Why

Judge v0.5.1 showed a structural failure when evaluating multi-facet queries.

Two opposite patterns were repeatedly observed:

1. Missing one requested concept could cause an otherwise relevant book to
   collapse to score 0.

2. Strong support for one requested concept could cause the entire query-book
   pair to be promoted to score 3 or 4 even when other important requested
   concepts were absent.

The judge also almost completely failed to use intermediate scores:

- score 1 predictions: 0
- score 2 predictions: 2

on the 55-case Q02-Q12 development slice.

Facet decomposition was introduced so that partial semantic coverage becomes
an explicit intermediate representation rather than something the model must
infer while simultaneously choosing an overall score.

### Validation

Deterministic scoring tests were added in:

`evals/test_facet_scoring.py`

Synthetic cases verify expected behavior for:

- no support -> 0
- incidental-only support -> 1
- one strong facet out of several -> 2
- partial mythology/adventure coverage -> 2
- broad historical-war coverage -> 3
- all central facets strong -> 4
- qualifier preventing a score of 4

The deterministic scoring tests passed.

### Smoke Test

Run:

`20260909T114825Z`

Scope:

- Q01 only
- 5 cases
- development data
- `--limit 5`

Results:

- Q01_R01 -> 3
- Q01_R05 -> 3
- Q01_R20 -> 1
- Q01_R50 -> 1
- Q01_NEG -> 1

Human labels:

- Q01_R01 -> 3
- Q01_R05 -> 3
- Q01_R20 -> 0
- Q01_R50 -> 0
- Q01_NEG -> 0

Smoke-test agreement:

- Exact agreement: 40%
- Within ±1 agreement: 100%

The run completed successfully using source-span evidence selection.

### Observations

The architecture successfully produced intermediate score-1 judgments rather
than collapsing all weak cases to 0 or strong cases to 3/4.

However, inspection of Q01_R20, Q01_R50, and Q01_NEG showed that the judge was
using `incidental` too broadly.

The judge sometimes treated concepts such as:

- personal growth
- adapting to change
- recovery
- general relationship difficulty

as incidental evidence for:

- forgiveness
- redemption

This revealed a semantic-support calibration issue rather than a deterministic
scoring issue.

The facet architecture and score computation were therefore retained.

The semantic definition of positive facet support required further tightening.

### Decision

Retain the v0.6 architecture.

Do not accept Judge Config v0.6.0 as the final calibrated semantic judge.

Advance to v0.6.1 with tighter semantic facet-support criteria.

### Development Note

An early v0.6.0 smoke attempt used free-form exact-quote extraction and failed
because the local model paraphrased evidence.

The evidence mechanism was changed to source-span selection before broader
v0.6 calibration.

This implementation change is recorded explicitly here because a v0.6.0 smoke
run had already occurred before the evidence mechanism was finalized.

## Judge Config v0.5.1
Date: 2026-09-09

### Changed
- Updated the judge configuration version from `0.5.0` to `0.5.1`.
- Updated the calibration dataset dependency from:
  - `semantic_relevance_calibration.v0.1.0.csv`
  - to `semantic_relevance_calibration.v0.2.0.csv`
- Kept the semantic judge behavior unchanged:
  - same rubric
  - same model
  - same temperature
  - same decision ladder
  - same evidence-grounding rules
  - same score mapping
- Used the repaired calibration dataset in which:
  - original candidate identities were restored
  - exact ISBN-13 values were recovered from the canonical book catalog
  - mixed encoding issues were repaired
  - human scores and human reasons were preserved
  - semantic retrieval was not rerun
- Fixed runner-side persistence so the final `judge_results.csv` stores the verified `grounded_evidence` returned by the evidence-extraction stage rather than the original paraphrased `verdict.evidence_text`.

### Why
The previous v0.5.0 calibration run was performed against dataset v0.1.0, which was later found to contain candidate drift.

Several `case_id` values had become associated with different books after candidate regeneration while retaining the original human labels. This made the resulting human-vs-judge comparison unreliable as a calibration result.

A repaired dataset, `v0.2.0`, was created from the original human-labelled candidate set without rerunning retrieval.

The repair process:

- recovered all 60 original candidates
- matched all 60 uniquely against the canonical book catalog
- restored exact 13-digit ISBN values
- restored canonical title, author, and description text
- preserved original human labels and reasons
- repaired mixed UTF-8 / Windows-1252 encoding issues

Because the semantic judge itself was not changed, this was treated as a patch-level judge-config revision rather than a new semantic judge version.

### Validation

Calibration configuration:

- Judge Config: `v0.5.1`
- Calibration Dataset: `v0.2.0`
- Rubric: `v0.1.0`
- Provider: Ollama
- Model: `jeffnyman/ts-evaluator`
- Temperature: `0.0`

Development set:

- Q01
- 5 cases
- Previously used for judge development and therefore excluded from primary calibration metrics

Primary holdout:

- Q02-Q12
- 55 cases

Holdout results:

- Exact agreement: **49.1%**
- Within ±1 agreement: **72.7%**
- Linear weighted Cohen's kappa: **0.501**
- Quadratic weighted Cohen's kappa: **0.649**

Judge prediction distribution across the 55-case holdout:

- Score 0: 29
- Score 1: 0
- Score 2: 2
- Score 3: 19
- Score 4: 5

Human-label distribution across the same holdout:

- Score 0: 15
- Score 1: 5
- Score 2: 9
- Score 3: 19
- Score 4: 7

Agreement by human score:

- Human 0:
  - Cases: 15
  - Exact agreement: 100%
  - Within ±1: 100%
  - Mean absolute difference: 0.00

- Human 1:
  - Cases: 5
  - Exact agreement: 0%
  - Within ±1: 60%
  - Mean absolute difference: 1.40

- Human 2:
  - Cases: 9
  - Exact agreement: 0%
  - Within ±1: 11.1%
  - Mean absolute difference: 1.89

- Human 3:
  - Cases: 19
  - Exact agreement: 57.9%
  - Within ±1: 78.9%
  - Mean absolute difference: 0.84

- Human 4:
  - Cases: 7
  - Exact agreement: 14.3%
  - Within ±1: 85.7%
  - Mean absolute difference: 1.00

### Observations

The repaired dataset slightly improved agreement compared with the earlier corrupted v0.1.0 run:

- Exact agreement:
  - v0.5.0 / dataset v0.1.0: 47.3%
  - v0.5.1 / dataset v0.2.0: 49.1%

- Linear weighted kappa:
  - previous: 0.477
  - repaired run: 0.501

- Quadratic weighted kappa:
  - previous: 0.617
  - repaired run: 0.649

However, the main calibration problem remained.

The judge showed strong score-distribution collapse:

- no score-1 predictions
- only two score-2 predictions
- most non-zero predictions concentrated at scores 3 and 4

This indicates that the judge is not reliably distinguishing:

- incidental relevance
- partial relevance
- clear relevance
- strong relevance

The strongest performance was at score 0:

- all 15 human score-0 cases were correctly judged as 0

This suggests the current judge behaves more like a conservative binary relevance detector than a calibrated five-level ordinal evaluator.

A binary interpretation of the holdout approximately gives:

- Precision: 100%
- Recall: 65%
- F1: 78.8%

where human score 0 is treated as irrelevant and scores 1-4 as relevant.

Several large disagreements revealed a recurring semantic-composition failure.

The judge sometimes treated:

- one missing requested facet as meaning no relevance at all
- one strongly matching facet as meaning the entire multi-part query was strongly satisfied

Examples included:

- historical / war / political-conflict queries being scored 0 despite descriptions clearly containing historical warfare
- partial mythology/adventure matches being scored 0 when some important requested facets were present
- organizational-effectiveness books being scored 4 even when leadership and teamwork were weak or absent

The evidence-grounding stage itself behaved correctly in this run.

All positive verdicts persisted grounded evidence spans that occurred in the supplied description.

### Decision
**Reject Judge Config v0.5.1 for production use as a five-level ordinal semantic-relevance judge.**

Do not tune v0.5.1 further.

Retain the following successful design elements for the next iteration:

- blind evaluation
- structured verdicts
- deterministic score mapping
- evidence grounding
- exact evidence extraction
- resumable execution
- run provenance/version tracking

Redesign the semantic classification architecture for the next judge version.

The next judge should explicitly decompose multi-part user requests into semantic facets before assigning an overall relevance level.

The 55 cases from Q02-Q12 have now been inspected in detail and should no longer be treated as an untouched holdout for future judge versions.

Use them as development/calibration data for the next iteration.

Create a new unseen human-labelled holdout before making any generalization claim about the next judge version.

### Unchanged
- Rubric version: `0.1.0`
- Provider: `ollama`
- Model: `jeffnyman/ts-evaluator`
- Base URL: `http://localhost:11434`
- Temperature: `0.0`
- Evaluation unit: one user query × one recommended book
- Semantic decision ladder:
  - none -> 0
  - incidental -> 1
  - partial -> 2
  - clear -> 3
  - strong -> 4

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

Calibration outcome: REJECTED

Reason:
Insufficient ordinal agreement on untouched holdout.
The judge substantially under-detects partial and implicit semantic
relevance and collapses the 1–2 region of the rubric.

Holdout size             55
Exact agreement          47.3%
Within ±1                72.7%
Linear weighted κ        0.477
Quadratic weighted κ     0.617

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