# Evaluation Changelog

# Semantic Relevance v0.29.0 r3

r3 repairs the sole remaining r2 gated failure: `U2_Q04_T70`.

## Diagnosis

r2 successfully recovered Q04 `movement_or_travel` from the exact supplied
`travel book` span, but the model verifier rejected `shipwrecked` as
`danger_or_threat`.

Facet spec 0.10.1 already states that an explicitly shipwrecked named traveler
in the supplied travel narrative is a hazard/danger cue. r3 therefore changes
no facet semantics. It deterministically implements that existing component
contract.

## r3 rule

For Q04 `dangerous journeys` / `danger_or_threat` only:

- an exact supplied span must explicitly contain `shipwreck` / `shipwrecked`; AND
- the supplied description must independently contain an explicit travel/movement anchor.

This preserves:
- Banker: danger/action but no travel anchor -> no dangerous journey.
- ordinary travel without danger -> no dangerous journey.
- isolated/meta shipwreck mention without a travel narrative -> no deterministic recovery.

## Versions

- judge config: 0.29.0-r3
- facet spec: unchanged 0.10.1
- regression manifest: 2.2.0
- development dataset: unchanged 6.1.0
- rubric: unchanged 0.1.0
- scoring logic: unchanged from r2

No independent v0.29 evidence has been created or consumed.

# Semantic Relevance v0.29.0 r2

r2 repairs exactly the two r1 gated regressions without changing labels or thresholds.

- U2_Q04_T70: recover explicit movement from `travel book` in another exact span; danger remains independently grounded.
- U2_Q09_T02: same-span `overthrow` + `totalitarianism` can establish resistance.

Banker remains protected because danger/time progression without an explicit movement anchor is insufficient. U4_Q09_T04 remains protected because authoritarian/tyrannical framing without an opposition action leaves resistance absent and the Q09 cap at partial.

Version pins: judge config 0.29.0-r2; facet spec 0.10.1; regression manifest 2.1.0; development dataset unchanged 6.1.0; rubric unchanged 0.1.0.
No independent v0.29 evidence has been created or consumed.

# Semantic Relevance v0.29.0 — adjudication overlay r1

This overlay applies the three human-label revisions explicitly approved after
review of the five consumed v0.28 final-holdout severe cases.

Approved revisions:

- `U4_Q03_T01`: `0 -> 2`
- `U4_Q06_T01`: `2 -> 0`
- `U4_Q09_T04`: `1 -> 2`

Two reviewed labels remain unchanged:

- `U4_Q03_T02`: human `2`; classified as a judge defect.
- `U4_Q04_T02`: human `0`; classified as a judge defect.

`U4_Q09_T04` remains a semantic repair target after adjudication because the
frozen r8 score is `3` while the adjudicated human score is `2`.

The source `v6.0.0` dataset is never overwritten. The overlay creates
`semantic_relevance_v0.29_development.v6.1.0.csv`, an adjudication manifest,
an adjudicated five-case review sheet, and a recalculated frozen-r8 baseline.

No judge calls or semantic changes occur in this step.

# Semantic Relevance v0.29.0 bootstrap r1

This overlay starts the new lineage **without tuning**.

It performs no judge calls and changes no human labels.

It creates a 240-case consumed-development dataset:

- 210 cases already consumed during v0.28 development;
- 30 cases from the now-consumed v0.28 r8 final holdout.

It then reconstructs the frozen r8 baseline across all 240 cases using already
existing judge results. The former final-holdout results are not rerun.

Finally, it creates a five-case severe-review CSV for:

- U4_Q03_T01
- U4_Q03_T02
- U4_Q04_T02
- U4_Q06_T01
- U4_Q09_T04

These are review targets only. No repair should be proposed until each is
classified against the frozen rubric/facet semantics as judge defect,
label/spec tension, or ambiguity.

Any future v0.29 independent qualification must use a newly created unseen
human-labelled evidence set after the v0.29 candidate is frozen.

# Semantic Relevance v0.28.0 r8 — final evaluation closeout r1

This overlay performs closeout only. It makes no judge calls and makes no semantic changes.

It verifies the completed one-time final-holdout lifecycle and writes:

`evals/releases/semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json`

The closeout records `EVALUATED_NOT_FINAL_QUALIFIED`, final decision
`FINAL_HOLDOUT_REVIEW`, all final metrics/gate outcomes, the five severe
final-holdout case IDs, immutable evidence hashes, and the evidence boundary
for v0.29.

The 30 r8 final-holdout cases are permanently consumed final evidence. They
may be used for diagnostics or regression development in v0.29+, but never
again as independent validation/final-holdout evidence.

Any v0.29 candidate must be frozen before creating/consuming a new unseen
human-labelled independent evidence set for qualification.

# Semantic Relevance v0.28.0 r8

r8 follows the failed r7 targeted gate.

## What failed in r7

`U2_Q11_T02` (*Sirens and Sea Monsters*) remained human=4 / judge=2.
The r7 semantic rule was correct, but its implementation required the classical
myth source/retelling signal and supernatural mythic-content signal to occur in
the same isolated evidence span.

The supplied description distributes those signals across multiple exact spans.

## r8 change

No label, threshold, scoring, or query meaning changes.

For Q11 mythology only, r8 deterministically composes exact description spans
when:

1. one span explicitly ties the narrative to classical mythic/epic source
   material or retelling; and
2. another supplied span explicitly depicts supernatural mythic figures,
   creatures, or events; and
3. the description is not analytical legend-origin/provenance prose.

No case ID, title, author, ISBN, or external entity knowledge is used.

Version identities:

- judge config: `0.28.0-r8`
- facet spec: `0.9.13`
- regression manifest: `1.7.0`
- targeted gate: 52 gated + 2 diagnostic-only cases
- development dataset: unchanged v5.0.0 / 210 consumed cases

Semantic hashes:

- judge: `e4efec54d03601e3016b376741c7f14fc1259ab3d63d1fdff75925cc2bfb43c1`
- scoring: `8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808`
- facet spec: `0952d80ebed8fbdebfb3b625f5648be9ccc08148d87d6753adc83d010953e289`
- judge config: `bdc4157b457e3453ff94b860cb45022a7c7c83f311432ef9841a1c1fe5a9bb8d`
- regression manifest: `da8d7ffe0f9131dd5ab976a0af72d7c478ee592b4cf1b495e105e483d908c585`

The independent final holdout remains `LOCKED_DO_NOT_RUN`.


# Semantic Relevance v0.28.0 r7

r7 follows the canonical r6 full-210 consumed-development gate.

r6 targeted regression passed, but the full gate failed because the original
150-case reference subset had two severe errors instead of the allowed one:

- `U_Q03_T10` — historical tolerated severe error.
- `U2_Q11_T02` — new r6 severe under-score: human 4, judge 2.

The r7 change is intentionally narrow:

- for Q11 mythology only, explicit classical mythic/epic source or retelling
  language plus explicit supernatural mythic narrative content grounds the
  mythology facet;
- the r5 legend-origin/provenance analytical exclusion remains unchanged;
- r6 whole-work Q01 and explicit myth/hero-content guards remain unchanged.

New regression target:

`U2_Q11_T02` — human 4; r7 required judge score 3-4.

Version identities:

- judge config: `0.28.0-r7`
- facet spec: `0.9.12`
- regression manifest: `1.6.0`
- gated targeted cases: 52
- diagnostic-only cases: 2
- development dataset remains 210 consumed cases / v5.0.0

The independent 30-case final holdout remains LOCKED_DO_NOT_RUN.

# Semantic Relevance v0.28.0 r6

r6 is a new semantic revision after the r5 targeted gate failed on two inherited positive controls. r5 is not modified retroactively.

## r6 repairs

1. **Q01 whole-work thematic enumeration** — a deterministic direct cue in a sentence explicitly characterizing the story/book/work as being of/about themes is forced to substantive/defining role. This makes the r4 prompt-level rule robust to model prominence variation.
2. **Q11 explicit myth/legend hero content** — if the supplied description itself presents myths/legends as stories involving heroes, legendary/mythic hero identity is positively grounded. The r5 analytical origin/provenance exclusion remains authoritative.

All five r5 independent-validation repairs remain unchanged. The final holdout remains locked/unseen.

Judge SHA-256: `aab36e98b9faa7a9dd771e3732b59ba685ff073e72d0bdc855cccebf447fafa8`
Scoring SHA-256: `8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808`
Facet spec 0.9.11 SHA-256: `e734cb73fdf5dd5847e78193c20930c165ce2771b79c2ffffce181f7cc9c4b53`
Judge config 0.28.0-r6 SHA-256: `b1be0036fcddfd644f3f2ec02ad6ea47af0e75c08efd3eb46de1d7f52d5df462`
Regression manifest 1.5.0 SHA-256: `ccc5aa2814c4fc991ffa97bbeaeed4c5f8dd60e073d8bf767905d4f9c2ebbe8d`

# Semantic Relevance v0.28.0 r5

r5 starts **after** the one-time r4 independent validation failed its preregistered gate.

The r4 validation result is immutable and is now consumed development evidence. The
separate 30-case U4 final holdout remains locked and is not included in this package's
development dataset.

## Localized semantic changes

1. **Q02** — interpersonal truth/secret disclosure and relationship consequences do not
   become suspense without an independent story-level suspense anchor.
2. **Q06** — war/trauma/injury recovery cannot invent grief or significant loss.
3. **Q10** — explicit organization-level dysfunction -> alignment/productivity/goal
   improvement positively establishes building effective organizations.
4. **Q11** — research about the origin/provenance of a legend is analytical/meta content,
   not automatically mythology or legendary-hero narrative content.
5. **Q12** — background social-class contrast without developed inequality/resource
   analysis is capped at incidental prominence.

## Versioning

- judge implementation SHA-256: `7bcbbbd44ffcc2c38dd8fb3ea7327fd0c79da7b777bb8f2381ff464992cb36dc`
- frozen scoring SHA-256: `8447a75aee29d1ebf34147e625a7fe8c1f8ccf4b409029e917ec4c47285e8808`
- facet spec: `0.9.10` / `a51d288bfd84c8c200692a92484e5e3a79ac05e2d99274fb3fb2d4d53ef30d7b`
- judge config: `0.28.0-r5` / `1ec3c6350cb106bf678020f81d64045ca4b29005dd563c892a492a86cd53750a`
- regression manifest: `1.4.0` / `1467ef1c48d72d5a6d200e8dcb807c2cc07c103de71e35e451e313703a072d94`
- post-validation adjudication manifest: `68f5d85fd5905034d813234b9f9df8202252ef1598e9676b5247902e55a7b893`

# Semantic relevance v0.28 unseen-pool build r1

This overlay starts the first genuinely independent evidence cycle after
freezing **v0.28.0 r4**.

It performs only two data-preparation actions:

1. build a 60-case `U4_` pool from raw application retrieval while excluding all
   historically consumed candidate identities; and
2. export a blind Excel workbook for human labeling.

It does **not** import completed labels, split the pool, run validation, run the
final holdout, or invoke the semantic-relevance judge.

Run from the repository root:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\run_v028_unseen_pool_builder.ps1 2>&1 |
  Tee-Object -FilePath .\v028_unseen_pool_build_output.txt
```

After a PASS, human-label all 60 workbook rows. Do not run any U4 judge calls.

# Semantic relevance v0.28.0 r4

Development-only revision after r3 passed targeted/full-180 but failed stability on `U_Q01_T02`.

## Why r4 exists

The r3 stability run was perfectly deterministic across all 13 cases, but `U_Q01_T02` produced `1,1,1` instead of its frozen positive-control range `2-3`. The source explicitly characterizes the whole work as a story of several themes including redemption. The r2 prominence instruction had overgeneralized list demotion from character-side-detail lists to whole-work thematic lists.

## Semantic change

- Whole-work thematic enumerations such as `this story of ... redemption` may be substantive.
- Character possession/relation/obligation enumerations such as `his king, his lover, his friends, his gods` remain incidental unless independently developed.
- Judge implementation, facet spec 0.9.9, scoring, and rubric are unchanged from r3.

## Gates

1. `prepare_v028_r4.ps1` (no judge calls)
2. `run_v028_r4_targeted.ps1` — 41 gated + 2 diagnostic = 43 cases
3. `run_v028_r4_full_180.ps1` — fresh 180-case consumed-development run
4. `run_v028_r4_stability.ps1` — restart-safe 3 fresh runs x 13 cases

No independent v0.28 holdout exists yet.


# Semantic Relevance v0.28.0 r2

Development-only repair after the v0.28.0 r1 targeted gate failed.

## r1 diagnosis
- `U3_Q03_T01` was repaired successfully and must remain `0`.
- The global speculative-entailment prompt/guard was over-broad and regressed:
  - `U_Q06_T02`
  - `U_Q07_T02`
  - `U2_Q04_T70`
- `U3_Q11_T02` improved from `3` to `2`, proving the localized adventure and
  legendary-hero boundaries worked, but the direct lexical `gods` cue was still
  promoted from concept presence to substantive whole-book prominence.

## r2 changes
1. Roll back the global r1 speculative-entailment prompt and deterministic
   wording-based guard to the r7 behavior.
2. Preserve the localized Q03 personal-growth boundary from r1.
3. Preserve the localized Q11 adventure and legendary-hero boundaries from r1.
4. Add deterministic direct-cue prominence isolation for a narrow structural
   pattern: a cue occurring only as one item in a possessive enumeration and
   having no independent role-supporting span is capped to incidental.
5. Preserve all 36 gated targeted expectations and both diagnostic-only cases.
6. Keep scoring, rubric, facet-spec version, judge-config version, and the
   180-case consumed-development dataset unchanged.

No independent v0.28 validation or final holdout exists yet.

# Semantic Relevance v0.28.0 r1

Development-only revision after the one-time v0.27 r7 final holdout.

## Scope
- Preserve all 34 r7 targeted contracts.
- Hard-fix only high-confidence consumed-holdout judge defects: U3_Q03_T01 and U3_Q11_T02.
- Keep U3_Q10_T01 and U3_Q11_T04 diagnostic-only pending semantic adjudication.
- Fail closed when an ENTAILED component rationale explicitly admits speculation (could/might/possibly/plausibly/can be interpreted/potential for).
- Strengthen Q03 personal-growth and Q11 adventure/legendary-hero boundaries.
- Reinforce that direct lexical verification proves presence, not prominence.
- Keep deterministic scoring and rubric byte-identical to r7.
- Build 180-case consumed development evidence by adding the consumed v0.27 final holdout to the prior 150 cases.

No v0.28 unseen validation or final holdout exists yet.

# v0.27.0 r7 changelog

- Preserved the successful r6 Q05 war-as-subject repair.
- Tightened Q01 `restoration_or_atonement` so directional movement away from a negative state cannot alone produce redemption.
- Tightened Q02 `story_level_tension_or_anticipation` so future-tense dangerous/consequential plot progression cannot alone produce suspense.
- Added a deterministic constituent-item scope guard for whole-book prominence, with an explicit collection-level facet escape hatch.
- Added v0.9.7 facet spec and v1.3.0 34-case regression manifest.
- Kept deterministic scoring, rubric, model/temperature, component IDs, and consumed-development dataset version 3.1.0 unchanged.

# v0.27.0 r6 changelog

- Consumed the first 30-case v0.27 validation as development evidence after r5 validation REVIEW.
- Added localized Q01 redemption, Q02 suspense, and Q05 war semantic-boundary fixes.
- Added generic collection/anthology constituent-item prominence guard.
- Preserved deterministic scoring, rubric v0.1.0, model, temperature, and all canonical component IDs.
- Added 150-case development dataset builder and 34-case regression manifest v1.2.0.
- Final 30-case holdout remains locked and untouched.

# v0.27.0 r5 changelog

- Preserved the r4 semantic changes and the 26/26 targeted-pass evidence.
- Fixed an unattended-run robustness defect in isolated canonical-component verification.
- Added explicit retry guidance for the invariant `external_knowledge_required=true => grounding_relation=missing`.
- Added a narrow deterministic fail-closed fallback after bounded retries only for that exact contradiction.
- Added equivalent explicit repair wording to missing-component recovery; its existing conservative fallback remains unchanged.
- Added unit coverage for successful repair and stubborn repeated contradiction.
- No scoring, rubric, model, dataset-label, targeted-manifest, facet-spec, Q04, Q05, or Q09 semantic change.

# v0.27.0 r4 changelog

- Diagnosed the r3 120-case REVIEW result.
- Added Q04/F1 action-driven-conflict entailment as an alternative to travel/quest structure.
- Added a high-precision Q05/F1 deterministic cue for standardized World War I/II names.
- Expanded the targeted regression manifest from 24 to 26 cases with U2_Q04_T02 and U_Q05_T30.
- Preserved the r3 Q09 external-knowledge guard and all earlier r6/r3 protections.
- Deliberately left U_Q03_T10 as the known single severe residual rather than case-tuning it.
- No scoring/rubric/model/dataset-label/facet-spec change.

# v0.27.0 r3 changelog

## Targeted-run evidence

The r1 semantic run passed 21/24 targeted expectations. `U2_Q05_T30 La Débâcle` passed, while three cases remained:

- `U_Q03_T02 Save the Date`: 2 -> 0, a new inherited-regression failure.
- `U2_Q04_T70 Gulliver's Travels`: 3 -> 2, improved but below target.
- `U2_Q09_T30 The Darling`: 0 -> 2, unchanged severe over-promotion.

Diagnostics showed three distinct mechanisms: Q03's explicit Mode-B wording was overridden by the global r1 policy; Q04 F3 still rejected text-grounded travel/danger evidence; and Q09 imported named-entity history while falsely setting `external_knowledge_required=false`.

## Semantic changes

- Removed the r1 global text-licensed-inference examples from all components.
- Added narrowly scoped component-local inference contracts for Q03, Q04, Q05, and Q09 only.
- Added a high-precision deterministic Q03/F1 promoted-growth Mode-B cue.
- Added a Q09/F2 Python text-anchor precision guard for both isolated verification and full-context recovery so identity/membership alone cannot establish resistance action or an oppressive/authoritarian target.
- Added facet spec v0.9.5. Relative to v0.9.4, only Q04 and Q09 semantics change:
  - Q04: `shipwrecked` in the supplied travel narrative is explicitly accepted as a danger/hazard cue.
  - Q09: political radical identity or named-movement membership alone establishes neither active resistance nor its oppressive/authoritarian target.
- Q05 v0.9.4 semantics are unchanged; local prompt guidance preserves the r1 fix.

## Unchanged

- Judge config version: 0.27.0
- Development dataset: 3.0.0 / 120 consumed cases
- Regression manifest: 1.0.0 / 24 targeted cases
- Scoring module and thresholds
- Rubric 0.1.0
- Model and temperature
- Canonical component IDs

## Methodology status

All 120 cases are consumed development evidence. There is no independent v0.27 holdout. The frozen v0.26 final-holdout result remains unchanged.

# v0.27.0 r2 changelog

## Packaging / verifier fix

- Repository-wide Python source scanning/compilation in `verify_v0_27_package.py` now uses `utf-8-sig`.
- This permits pre-existing UTF-8 BOM-prefixed Python files to be verified without producing `SyntaxError: invalid non-printable character U+FEFF`.
- No judge, prompt, facet, scoring, rubric, dataset, regression expectation, model, or temperature change from r1.

## Added — text-licensed entailment / external-knowledge boundary

Component verification now explicitly distinguishes ordinary semantic entailment licensed by supplied text from facts that would have to come from named-entity or world knowledge. The new `external_knowledge_required` audit field is carried from isolated/recovery verification into the canonical component ledger; a positive result marked as requiring outside knowledge is mechanically rejected.

## Facet spec v0.9.4

Narrow local clarifications were added for the three adjudicated post-holdout judge failures:

- Q04: explicit travel framing plus a named subject's encounters across multiple stated places can entail movement/journey/adventure; location alone remains insufficient.
- Q05: regime/empire collapse or end linked by the supplied text to revolution/revolutionary violence can establish political-power stake and actual political struggle.
- Q09: membership in a named political movement/organization cannot import the movement's target or ideology from outside knowledge.

All canonical component IDs, scoring, rubric, model, and temperature remain unchanged.

## Evaluation status

v0.27 development uses 120 consumed cases. There is no independent holdout.

# v0.27.0 r1 changelog

## Added — text-licensed entailment / external-knowledge boundary

Component verification now explicitly distinguishes ordinary semantic entailment licensed by supplied text from facts that would have to come from named-entity or world knowledge. The new `external_knowledge_required` audit field is carried from isolated/recovery verification into the canonical component ledger; a positive result marked as requiring outside knowledge is mechanically rejected.

## Facet spec v0.9.4

Narrow local clarifications were added for the three adjudicated post-holdout judge failures:

- Q04: explicit travel framing plus a named subject's encounters across multiple stated places can entail movement/journey/adventure; location alone remains insufficient.
- Q05: regime/empire collapse or end linked by the supplied text to revolution/revolutionary violence can establish political-power stake and actual political struggle.
- Q09: membership in a named political movement/organization cannot import the movement's target or ideology from outside knowledge.

All canonical component IDs, scoring, rubric, model, and temperature remain unchanged.

## Evaluation status

v0.27 development uses 120 consumed cases. There is no independent holdout.

# v0.27.0 development changelog — preparation r1

## Methodological transition

The frozen v0.26.0 r6 candidate completed its one-time 30-case final holdout with
status `REVIEW`. That result is immutable.

v0.27 begins a post-holdout development phase. The 30 final-holdout cases are now
consumed evidence and may be used for diagnosis/development, but never again as
an independent holdout.

## Post-holdout human re-adjudication

Four labels were supplied by the human reviewer after the final run:

- U2_Q07_T10: 4 -> 0
- U2_Q06_T10: 3 -> 0
- U2_Q01_T10: 2 -> 0
- U2_Q01_T70: 2 -> 0

These revisions are explicitly recorded as post-holdout/non-blind and apply only
to the new v0.27 development dataset.

## Remaining severe judge targets

- U2_Q04_T70: 3 vs 0 — text-grounded entailment under-promotion
- U2_Q05_T30: 4 vs 2 — political-conflict component under-promotion
- U2_Q09_T30: 0 vs 2 — unsupported/external-knowledge over-promotion

No judge, facet, scoring, or prompt behavior is changed in this preparation package.

## v0.26.0 package r6 — local semantic-contract isolation

- Preserves facet spec `v0.9.3`, including Q03/F1's independently sufficient experienced-development and explicit-growth-promotion modes.
- Preserves Q07/F1's r4 parent-child referent-binding semantics.
- Removes the two r5 **global** prompt lines that instructed every isolated-component and missing-component verifier to interpret alternative positive grounding modes / sufficiency clauses.
- The same semantics now come only from the frozen facet/component definition that declares them; unrelated facets do not receive Q03-style global guidance.
- Adds `test_v0_26_local_contract_isolation.py`, verifying Q03 prompts contain the local Mode A/Mode B contract while Q07 prompts do not inherit it.
- Updates the Q03 personal-growth prompt contract test to prove its local definition is surfaced without the removed global instruction.
- Adds targeted regression manifest `v6.2.0`; the 21 case expectations are unchanged and now explicitly require `U_Q03_T02==2` and `U_Q07_T02>=3` together.
- Adds `analyze_v0_26_stability.py` for the post-targeted 3-run spot-check of `U_Q03_T02` and `U_Q07_T02`.
- Development dataset remains `2.2.0`; label revisions remain `1.1.0`; calibration provenance remains `0.5.0`; rubric and scoring are unchanged.
- Retains the Windows UTF-8 and verifier-syntax fixes from r5 winfix2.

## v0.26.0 package r6 — local semantic-contract isolation

- Preserves facet spec `v0.9.3`, including Q03/F1's independently sufficient experienced-development and explicit-growth-promotion modes.
- Preserves Q07/F1's r4 parent-child referent-binding semantics.
- Removes the two r5 **global** prompt lines that instructed every isolated-component and missing-component verifier to interpret alternative positive grounding modes / sufficiency clauses.
- The same semantics now come only from the frozen facet/component definition that declares them; unrelated facets do not receive Q03-style global guidance.
- Adds `test_v0_26_local_contract_isolation.py`, verifying Q03 prompts contain the local Mode A/Mode B contract while Q07 prompts do not inherit it.
- Updates the Q03 personal-growth prompt contract test to prove its local definition is surfaced without the removed global instruction.
- Adds targeted regression manifest `v6.2.0`; the 21 case expectations are unchanged and now explicitly require `U_Q03_T02==2` and `U_Q07_T02>=3` together.
- Adds `analyze_v0_26_stability.py` for the post-targeted 3-run spot-check of `U_Q03_T02` and `U_Q07_T02`.
- Development dataset remains `2.2.0`; label revisions remain `1.1.0`; calibration provenance remains `0.5.0`; rubric and scoring are unchanged.
- Retains the Windows UTF-8 and verifier-syntax fixes from r5 winfix2.

## v0.26.0 package r5 — Q03 explicit growth-promotion grounding

- Added facet spec `v0.9.3`, preserving every canonical component ID and changing semantic content only for `Q03/F1 personal growth` relative to `v0.9.2`.
- Reframed Q03/F1 as two independently sufficient positive grounding modes: experienced development (Mode A) or explicitly promoted maturity/personal growth in intended participants/readers (Mode B).
- Clarified that Mode B does not require a named individual to have already completed the development; prospective, instructional, promotional, or aspirational phrasing is not disqualifying when personal growth/maturity itself is explicitly stated as the intended effect.
- Preserved spirituality-only negative boundaries: prayer, faith, devotion, wisdom, inner peace, or similar content does not count when growth/maturity/development itself is absent.
- Added a generic component-prompt spec-fidelity rule to both isolated verification and missing-component recovery: explicit sufficiency clauses in frozen component definitions must be honored rather than silently replaced by stricter criteria.
- Strengthened the Q03 personal-growth contract test with two positive promotion-mode fixtures and two spirituality/peace negative controls, plus prompt-level assertions that the disjunctive contract is visible to the model.
- Development dataset remains `2.2.0`; label revisions remain `1.1.0`; calibration provenance remains `0.5.0`; rubric remains `0.1.0`.
- Targeted regression manifest advances `v6.0.0 -> v6.1.0` with the same 21 case expectations and r5 provenance.

## v0.26.0 package r4 — local referent/scope fixes + blind-review label adjudication

### Changed

- Added facet spec `v0.9.2` while preserving every canonical component ID from v0.9.1.
- Q07/F1 now distinguishes a person's simultaneous roles: explicit parental opposition/action toward a member of a couple may establish that person's separate parent-child relationship; a marriage by itself still cannot substitute for parent-child.
- Q03/F1 now permits explicit maturity/personal-growth statements about intended participants/readers and no longer requires the authors themselves to be the people developing.
- Preserved Q02/F1 decision-vs-suspense semantics from v0.9.1 unchanged.
- Added blind-review label revisions:
  - `U2_Q01_T02`: `2 -> 1` (weak redemption match; forgiveness absent).
  - `U2_Q05_T10`: `4 -> 2` (war matches; political-conflict aspect absent).
- Consumed development dataset version advances `2.1.0 -> 2.2.0`.
- Label revision manifest advances `1.0.0 -> 1.1.0`.
- Targeted regression manifest advances `5.0.0 -> 6.0.0` and expands from 17 to 21 protected cases.
- Added focused contracts for Q07 parent-child referent binding, Q03 personal-growth participant scope, and r4 blind-review labels.

### Unchanged

- Judge version `0.26.0`.
- Calibration dataset pin `0.5.0`.
- Rubric `0.1.0`.
- Isolated per-component verification architecture.
- Python deterministic facet assembly.
- Missing-component-only full-context recovery.
- Scoring module and 0-4 scoring semantics.
- Direct Ollama transport recovery behavior.
- Q01 forgiveness/redemption semantic boundaries and Q05 political-conflict semantic boundaries.

### Acceptance target

- Fresh 21-case targeted regression: 21/21 PASS.
- Then fresh 90-case consumed-development run and development-reference review.

## v0.26.0 package r3 — local-boundary scoping

### Evidence motivating the patch

The r2 targeted run again reached 16/17 PASS. The intended Q02 suspense false positive was fixed (`U_Q02_NEG: 2 -> 0`), but `U_Q06_T02` regressed from judge 3 to judge 2.

The r1/r2 component comparison isolated the regression to Q06/F1 grief on span S8. In both runs the span established significant loss. In r1 it also entailed `mourning_or_deep_sorrow_response`; in r2 the same component became `missing[boundary]` despite the reason acknowledging devastation and emotional impact. Q06/F2 still entailed significant personal loss from S8.

### Changed

- Reverted the r2 global `NEGATIVE-BOUNDARY PRIORITY` wording from isolated-component prompts.
- Reverted the same global precedence wording from missing-component recovery prompts.
- Kept normal component-specific negative-boundary enforcement unchanged.
- Kept facet spec v0.9.1 and the local Q02/F1 decision-vs-suspense clarification unchanged.
- Updated `test_v0_26_suspense_boundary.py` to verify local-spec ownership of the suspense boundary.
- Added `test_v0_26_grief_non_regression.py` to protect Q06/F1 from unrelated global prompt drift.

### Unchanged

- judge version 0.26.0
- facet spec v0.9.1 contents
- all canonical component IDs
- v0.26 isolated-component architecture
- deterministic facet assembly
- missing-component-only recovery
- deterministic scoring
- rubric v0.1.0
- development labels/version 2.1.0
- calibration provenance 0.5.0
- targeted regression manifest v5.0.0

### Acceptance gate

A fresh targeted run must report 17/17 PASS, including both `U_Q02_NEG <= 1` and `U_Q06_T02 >= 3`, before starting the 90-case development run.

## v0.26.0 package r2 — suspense negative-boundary precedence

### Evidence motivating the patch

The v0.26.0 r1 targeted run reached 16/17 PASS. `U_Q02_NEG` remained a false positive (`human=0`, `judge=2`) because the isolated suspense component call treated an important caregiving choice and its possible consequences as `story_level_tension_or_anticipation`.

### Changed

- Added facet spec `semantic_relevance_query_facets.v0.9.1.json`.
- Preserved every canonical component ID and all facet decomposition from v0.9.0.
- Clarified only Q02/F1 suspense semantics and component boundaries:
  - consequential or high-stakes personal decisions remain decision uncertainty by themselves;
  - illness/caregiving stakes do not become suspense merely because a choice is unresolved;
  - a plot-driving decision is not automatically story-level tension;
  - positive suspense requires an independently tension-producing unfolding event/outcome.
- Added explicit `NEGATIVE-BOUNDARY PRIORITY` to isolated-component verification.
- Added the same precedence rule to missing-component recovery.
- Explicitly forbids renaming/reframing a boundary-blocked near-concept as the target component.
- Added `test_v0_26_suspense_boundary.py` with two negative decision fixtures and one positive independent-threat fixture.
- Added a regression assertion that all canonical component IDs remain unchanged and every facet other than Q02/F1 is identical to v0.9.0.

### Unchanged

- judge version `0.26.0`
- v0.26 isolated-component architecture
- deterministic facet assembly
- missing-component-only full-context recovery
- deterministic 0–4 scoring
- rubric v0.1.0
- development labels/version 2.1.0
- calibration provenance 0.5.0
- targeted regression manifest v5.0.0

### Acceptance gate

The package-level verifier must pass, then a **fresh** 17-case targeted run must report 17/17 PASS. The new focal requirement is `U_Q02_NEG <= 1`; all 16 cases that passed under r1 must remain protected.

## v0.26.0 — isolated canonical-component verification

### Changed

- Replaced production single-span holistic facet verification with one LLM call per frozen canonical component.
- Added `IsolatedComponentVerification` with `missing | explicit | entailed` component grounding states.
- Moved full-facet `direct / entailed / adjacent / unsupported` assembly into deterministic Python.
- Replaced production holistic composite verification with missing-component-only full-context recovery.
- Added `FullContextComponentRecovery`; only components still missing after isolated audits are eligible for recovery.
- Established component outcomes are never re-opened by full-context recovery.
- Same-span component resurrection is mechanically rejected; bounded repair exhaustion leaves that component missing and records the reason.
- Preserved prior component-specific negative-boundary results when recovery does not find valid new evidence.
- Extended component audit records with `grounding_relation`.
- Direct Ollama recovery transport uses JSON mode for both the legacy composite schema and the v0.26 component-recovery schema.

### Unchanged

- deterministic 0–4 scoring
- rubric v0.1.0
- facet spec v0.9.0 and its canonical component definitions/boundaries
- Stage A candidate selection
- facet-independent book-subject analysis
- explicit subject relation / prominence resolution
- hard-exclusion precheck
- consumed development dataset version 2.1.0
- calibration provenance pinned to dataset 0.5.0
- blind-review `U2_Q03_T30` human-label revision

### Targeted acceptance gate

The 17-case regression manifest remains the preregistered gate. The main v0.26 focus cases are:

- `U_Q06_T02 >= 3`
- `U_Q07_T10 <= 1`
- `U_Q04_NEG <= 1`
- `U_Q06_T70 <= 1`

The package-level tests verify architecture and deterministic contracts only. Semantic acceptance still requires the actual Ollama targeted run.

## v0.26.0 — isolated canonical-component verification

### Changed

- Replaced production single-span holistic facet verification with one LLM call per frozen canonical component.
- Added `IsolatedComponentVerification` with `missing | explicit | entailed` component grounding states.
- Moved full-facet `direct / entailed / adjacent / unsupported` assembly into deterministic Python.
- Replaced production holistic composite verification with missing-component-only full-context recovery.
- Added `FullContextComponentRecovery`; only components still missing after isolated audits are eligible for recovery.
- Established component outcomes are never re-opened by full-context recovery.
- Same-span component resurrection is mechanically rejected; bounded repair exhaustion leaves that component missing and records the reason.
- Preserved prior component-specific negative-boundary results when recovery does not find valid new evidence.
- Extended component audit records with `grounding_relation`.
- Direct Ollama recovery transport uses JSON mode for both the legacy composite schema and the v0.26 component-recovery schema.

### Unchanged

- deterministic 0–4 scoring
- rubric v0.1.0
- facet spec v0.9.0 and its canonical component definitions/boundaries
- Stage A candidate selection
- facet-independent book-subject analysis
- explicit subject relation / prominence resolution
- hard-exclusion precheck
- consumed development dataset version 2.1.0
- calibration provenance pinned to dataset 0.5.0
- blind-review `U2_Q03_T30` human-label revision

### Targeted acceptance gate

The 17-case regression manifest remains the preregistered gate. The main v0.26 focus cases are:

- `U_Q06_T02 >= 3`
- `U_Q07_T10 <= 1`
- `U_Q04_NEG <= 1`
- `U_Q06_T70 <= 1`

The package-level tests verify architecture and deterministic contracts only. Semantic acceptance still requires the actual Ollama targeted run.

# Semantic relevance judge v0.25.0

## Summary

v0.25.0 replaces model-authored semantic component labels with canonical component IDs frozen in facet spec v0.9.0. It is intended to fix the four remaining v0.24 targeted failures while preserving the 13 targeted cases that already passed.

## Semantic architecture changes

- Added `required_components` to every facet in `semantic_relevance_query_facets.v0.9.0.json`.
- Each required component has a stable `component_id`, generic definition, and generic negative boundaries.
- `ComponentEvidenceCheck` now returns `component_id` and `negative_boundary_applied`.
- Python requires the returned ledger to match the frozen component IDs exactly and in order.
- `missing_semantic_component` must name a real missing canonical component ID for production facets.
- Full-context recovery may inspect Stage-A-unseen spans and may use 1-4 supporting spans.
- Canonical monotonicity prevents the same previously audited evidence from resurrecting a component that all cited single-span audits marked missing.
- Repeated structured-output validation failure now propagates as an evaluation failure instead of being silently converted to semantic `UNSUPPORTED`.

## Generic boundary additions

- parent-child relation type cannot be substituted with marriage/romance/friendship/sibling relations;
- location/trapping/pursuit/rescue is not movement/travel;
- metaphorical/historical `lost time/place/past` is not automatically personal significant loss;
- reconstructing history or another person's former life is not rebuilding one's own life after grief/loss;
- strong affective presentation can satisfy `moving` without literal reader-response wording.

## Frozen / unchanged behavior

- deterministic 0-4 aggregation semantics;
- rubric v0.1.0;
- book-subject extraction and subject-relation role resolution;
- candidate ranking;
- hard-exclusion precheck;
- dataset version 2.1.0 and label-revision manifest;
- direct Ollama recovery transport.

## Targeted gate

Focus assertions:

- `U_Q06_T02 >= 3`
- `U_Q07_T10 <= 1`
- `U_Q04_NEG <= 1`
- `U_Q06_T70 <= 1`

All other v0.24 targeted protections remain active.

# Validation

v0.25.0 r1 targeted result: ABORTED

Completed: 4/17

Known failures before abort:
  U_Q01_T70 = 2
  U_Q04_NEG = 2
  U_Q07_T10 = 3

Runtime/validation failure:
  repeated model inconsistency:
  positive facet relation with incomplete canonical ledger

Conclusion:
  canonical component specification alone is insufficient;
  component grounding must be decomposed.

# Semantic relevance judge v0.23.0

## Verification hardening

v0.23.0 addresses the dominant v0.22.2 development failure mode: false-positive semantic completion. The verifier was sometimes accepting evidence that established a nearby concept while inventing an indispensable missing component, such as treating a complicated marriage as a complicated parent-child relationship, danger while stationary as a dangerous journey, or generic recovery as redemption.

Both isolated and composite verification now expose an explicit component-completeness contract. `DIRECT` and `ENTAILED` require all indispensable semantic components to be established by the supplied evidence. `ADJACENT` must identify the missing component. Explicit exclusion clauses in the frozen semantic definition are binding.

No Stage-A subject extraction, Stage-B subject relation, scoring, facet-spec, rubric, candidate-selection, deterministic-cue, or hard-exclusion-precheck semantics are changed.

## Development-label revision

`U2_Q03_T30` is revised from human score 2 to 0 after blind review. The original source data is not mutated; the revision is applied by `build_v0_23_development_dataset.py` from an auditable revision manifest. The derived development dataset version becomes `2.1.0`.

## Regression gate

The targeted gate expands from 9 to 17 cases by adding all seven severe over-promotions from the v0.22.2 full run plus `U2_Q03_T30` as a relabel-preservation control.


# Semantic relevance judge v0.22.2

## Reliability fix

Stage-A facet-independent book-subject extraction now retries malformed exact-span references with validator feedback. The observed failure was a model response containing `primary_subject_span_ids=["S6-S9"]`, which correctly failed strict validation because `S6-S9` was not a supplied source span ID.

The retry prompt now includes the mechanical validation failure and explicitly demonstrates the valid form `["S6", "S7", "S8", "S9"]`. The base Stage-A instructions also prohibit range syntax such as `S6-S9`, `S6–S9`, and `S6 through S9`.

## Deliberately unchanged

No span-range normalization is performed in Python. Unknown IDs remain validation failures. v0.22.1 subject extraction semantics, subject-relation semantics, verification, exclusions, and deterministic scoring are unchanged.


# Semantic relevance judge v0.22.1

## Changed

Stage B now classifies each verified core facet with one mutually exclusive relationship to the frozen v0.22 book subject: same primary subject, defining content/narrative driver, causal/contextual background, example/meta, or other.

The deterministic resolver maps those relationships to existing context roles. A causal/background facet can still resolve substantive only when the role stage explicitly finds independent substantive development beyond the causal role.

## Why

v0.22.0 correctly separated book-subject extraction from facet-conditioned role judgment, fixing the persistent Outsiders Within false positive. Diagnostics then showed Stage B contradicting otherwise-correct frozen subjects: constituent mythology content, story-defining fantasy/journey content, and poverty itself could be mislabeled background. The explicit relationship taxonomy removes that ambiguity without changing Stage A, verification, exclusions, or scoring.

## Audit

Facet audit records now include `subject_relation` and `subject_relation_reason`, while legacy role booleans remain derived for backward inspection.

# Semantic relevance judge v0.22.0

## Why this is a minor-version architecture change

v0.21.2 still showed facet-conditioned subject drift: a background inequality factor could be promoted to `central`, while story-defining fantasy/magic/danger could collapse to `incidental`. Further prompt-only patching would preserve the same source of instability.

## Changes

- Add a facet-independent `BookSubjectAnalysis` stage that runs once per query-book case before facet role classification.
- The book-subject stage sees only the numbered description: no user query, no facet text, no semantic definition, no human label.
- Freeze `primary_subject_summary` and `primary_subject_span_ids` once and reuse them unchanged for every core-facet role call.
- Core-facet role classification is now explicitly comparative: judge the verified facet against the frozen book subject instead of constructing a new primary subject from the facet-conditioned prompt.
- Preserve the existing deterministic role resolver precedence and all deterministic scoring rules.
- Persist book-subject analysis and full decomposed role signals/reasons for audit.
- Add static contracts preventing query/facet leakage into book-subject extraction.
- Keep the same nine preregistered targeted acceptance gates before the 90-case development run.

No title-, author-, ISBN-, case-, or query-specific production behavior is added.

# Semantic relevance judge v0.21.2

v0.21.2 is a narrow follow-up to v0.21.1. The v0.21.1 targeted run passed 8/9 gates; the remaining failure exposed an over-broad interpretation of `is_substantively_examined` for causal/background factors.

Changes:

- Keep frozen facet spec v0.8.0, rubric v0.1.0, and deterministic 0-4 scoring unchanged.
- Keep the v0.21.1 decomposed-role resolver precedence unchanged.
- Tighten `is_substantively_examined`: the facet itself must receive meaningful development, analysis, exploration, narrative treatment, portrayal, or sustained attention.
- A facet is not substantive merely because it appears in a list of causes/conditions, explains some OTHER main subject, or is important contextual background.
- Add a counterfactual classifier check: remove the causal/contextual relation and ask whether the description still meaningfully discusses the facet itself.
- Preserve coexistence of `background_cause_or_factor=true` and `is_substantively_examined=true` when the facet genuinely receives independent development beyond its causal role.
- Add a static causal-background contract test and keep all nine targeted acceptance gates unchanged.

No title-, author-, ISBN-, case-, or query-specific judge behavior is added.

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