# Evaluation Framework

This directory contains the AI-evaluation and system-quality framework for the
[Semantic Book Recommender](../README.md).

The root project README explains how the recommender is built and run. This
document explains how the repository evaluates two different things:

1. **the semantic-relevance LLM judge itself**, and
2. **the recommender system that is measured with that judge**.

That distinction is intentional. A measurement problem should not be mistaken
for a recommender problem, and a recommender failure should not automatically
cause the judge to be retuned.

---

## What this part of the repository demonstrates

The `evals/` code is a practical example of testing non-deterministic AI
systems with more rigor than a few manual prompts.

It includes:

- human-labelled judge calibration;
- versioned rubrics, facet specifications, and judge configurations;
- isolated semantic component verification;
- deterministic guards around LLM reasoning;
- targeted regression suites;
- stability checks;
- unseen validation and final-holdout protocols;
- evidence-contamination controls;
- frozen system-evaluation contracts;
- SHA-256 provenance checks;
- query-level bootstrap confidence intervals;
- paired version-to-version evaluation;
- explicit release gates;
- failure investigation and regression testing;
- restart-safe long-running judge executions;
- top-K retrieval headroom diagnostics.

The goal is not merely to produce a metric. The goal is to preserve enough
evidence to answer:

```text
What was tested?
What was frozen?
Which evidence had already been consumed?
What changed?
What did not change?
How uncertain is the result?
What failure mode does the evidence support?
What is the next controlled experiment?
```

---

# 1. Evaluation layers

The repository has two related but separate evaluation lifecycles.

```text
                    HUMAN-LABELLED CASES
                            |
                            v
                LLM JUDGE CALIBRATION
                            |
                    judge candidate
                            |
                  regression / stability
                            |
              validation / holdout evidence
                            |
                            v
                 measurement instrument
                            |
                            |
                            v
                  BOOK RECOMMENDER
                            |
                    candidate retrieval
                            |
                    final recommendations
                            |
                            v
                 SYSTEM EVALUATION
                            |
                  frozen judge scores
                            |
             metrics + bootstrap uncertainty
                            |
               release gates / diagnosis
```

## Judge calibration

Judge calibration asks:

> Does the semantic-relevance judge measure query-book relevance well enough
> against human-labelled examples?

Human labels are relevant here.

## System evaluation

System evaluation asks:

> Does the recommender return sufficiently relevant books?

The system benchmark deliberately does **not** use human relevance labels.
Instead, it consumes a frozen judge as its measurement instrument.

This separation prevents recommender results from silently changing the
measurement definition.

---

# 2. Current semantic judge architecture

The main semantic judge implementation is:

```text
evals/semantic_relevance_facet_judge.py
```

with deterministic scoring in:

```text
evals/semantic_relevance_facet_scoring.py
```

The system-evaluation judge configuration currently pinned by the benchmark is:

```text
evals/judge_configs/semantic_relevance_judge.v0.29.0-r5.json
```

Key configuration values in that source are:

```text
provider     = ollama
model        = jeffnyman/ts-evaluator
temperature  = 0.0
facet spec   = 0.10.1
rubric       = 0.1.0
```

The config itself is still marked `development`; the system-evaluation
contract freezes this candidate as a measurement instrument for the recommender
benchmark. That should not be confused with claiming that the judge has passed
every possible independent qualification lifecycle.

The judge is hybrid rather than a single unconstrained prompt.

At a high level:

```text
book description
      |
      v
facet-independent book-subject analysis
      |
      v
query decomposed into frozen semantic facets/components
      |
      v
candidate evidence selection
      |
      v
isolated component verification
      |
      +--> deterministic direct-cue rules
      +--> polarity guards
      +--> hard-exclusion checks
      +--> external-knowledge restrictions
      |
      v
deterministic facet assembly
      |
      v
deterministic 0-4 scoring
```

The implementation intentionally separates LLM semantic interpretation from
final score aggregation so that known failure modes can be covered by
deterministic tests.

---

# 3. Relevance score scale

The semantic judge uses an ordinal 0-4 relevance scale.

| Score | Meaning |
|---:|---|
| 4 | Strong match |
| 3 | Good / clear match |
| 2 | Partial match |
| 1 | Weak / incidental match |
| 0 | No meaningful supported match |

The exact rubric is versioned under:

```text
evals/rubrics/semantic_relevance/
```

The current system benchmark uses:

```text
semantic_relevance_rubric.v0.1.0.json
```

A key principle throughout the judge lineage is that supplied book text is the
evidence boundary. External book knowledge must not be used to rescue a claim
that the supplied description does not establish.

---

# 4. Judge calibration metrics

Human-labelled calibration and qualification workflows use several metrics
rather than a single aggregate number.

Typical measures include:

- exact agreement;
- within-one agreement;
- mean absolute error;
- linear weighted Cohen's kappa;
- quadratic weighted Cohen's kappa;
- binary relevance precision;
- binary relevance recall;
- F1;
- severe disagreement count.

A severe disagreement is:

```text
abs(human_score - judge_score) >= 2
```

This matters because an ordinal judge can have acceptable average agreement
while still making a few materially wrong decisions.

The repository contains version-specific analysis scripts such as:

```text
evals/analyze_v0_28_r8_development.py
evals/analyze_v0_28_r8_final_holdout.py
evals/analyze_v0_29_r3_development.py
evals/analyze_v0_29_r5_targeted.py
```

along with generic earlier analysis utilities.

---

# 5. Evidence lifecycle

One of the most important ideas in this repository is that evaluation evidence
has a lifecycle.

Evidence is classified by how it has been used:

```text
calibration / development
targeted regression
stability
validation
final holdout
system benchmark
diagnostic evidence
```

Once a case has been inspected and used to modify the system, it is no longer
fresh evidence of generalization.

A consumed case may still be useful for:

- debugging;
- regression testing;
- failure reproduction;
- controlled before/after comparison.

But it must not later be presented as an unseen holdout.

The repository contains explicit support for this lifecycle:

```text
evals/fresh_unseen/
evals/NEXT_UNSEEN_EVALUATION_PROTOCOL.md
evals/build_*unseen*_pool.py
evals/split_*unseen*.py
evals/prepare_*labeling*.py
evals/check_*integrity.py
evals/preflight_*.py
evals/releases/
```

Several historical holdout files are deliberately named with markers such as:

```text
DO_NOT_RUN_YET
```

Those boundaries are part of the evaluation design, not decorative filenames.

---

# 6. Versioning

Evaluation components are versioned independently.

The repository's `EVALUATION_VERSIONING.md` distinguishes:

```text
rubric version
calibration dataset version
judge config version
```

This is important because a new prompt does not necessarily imply a new gold
dataset, and correcting metadata does not necessarily imply a new rubric.

The repository also versions:

- semantic facet specifications;
- regression manifests;
- release-candidate manifests;
- validation/final-holdout protocols;
- system-evaluation contracts;
- frozen analysis/scoring inputs.

The rule is simple:

> Do not silently overwrite an artifact whose meaning affects a recorded
> evaluation result.

Create a new version or a documented amendment.

---

# 7. Repository layout

The actual `evals/` tree is intentionally large because it preserves the
evolution of the evaluation system.

The important categories are:

```text
evals/
|
|-- CHANGELOG.md
|-- EVALUATION_VERSIONING.md
|-- JUDGE_CALIBRATION_README.md
|-- NEXT_UNSEEN_EVALUATION_PROTOCOL.md
|
|-- semantic_relevance_facet_judge.py
|-- semantic_relevance_facet_scoring.py
|
|-- run_judge_*.py
|-- analyze_*.py
|-- build_*.py
|-- verify_*.py
|-- preflight_*.py
|-- test_*.py
|
|-- datasets/
|   `-- versioned calibration, development, unseen and benchmark data
|
|-- facets/
|   `-- semantic_relevance/
|       `-- versioned query-facet specifications
|
|-- judge_configs/
|   `-- versioned semantic-relevance judge configurations
|
|-- rubrics/
|   `-- semantic_relevance/
|       `-- versioned scoring rubrics
|
|-- baselines/
|   `-- frozen historical reference outputs
|
|-- fresh_unseen/
|   |-- unseen-evaluation protocol
|   `-- human-labeling score guide
|
|-- releases/
|   `-- release-candidate and final-evaluation manifests
|
|-- reports/
|   `-- curated engineering/evaluation reports
|
`-- system_evaluation/
    `-- recommender system-quality contracts, collectors, scorers and analyzers
```

Runtime output is written under:

```text
evals/runs/
```

but `.gitignore` explicitly excludes that directory. Run artifacts are local
execution evidence unless selected outputs are deliberately promoted into a
report, baseline, release manifest, or other versioned artifact.

---

# 8. Why there are many version-specific scripts

The repository intentionally retains scripts and tests from earlier judge
versions.

For example, the tree contains version-specific development, regression,
stability, and verification code from v0.19 through v0.29.

This serves several purposes:

- preserves the engineering history of each repair;
- prevents a later implementation from silently redefining an earlier result;
- keeps localized regression contracts executable;
- documents how specific failure classes were diagnosed and repaired.

The directory is therefore closer to an **evaluation laboratory / audit trail**
than a minimal production package.

A reviewer should usually begin with the current architecture and reports rather
than reading every historical script chronologically.

Recommended entry points are:

```text
evals/semantic_relevance_facet_judge.py
evals/semantic_relevance_facet_scoring.py
evals/CHANGELOG.md
evals/reports/
evals/system_evaluation/
```

---

# 9. Regression and contract tests

The evaluation code has extensive deterministic tests around the
non-deterministic judge.

Examples include tests for:

- evidence-candidate selection;
- multi-span composition;
- deterministic direct cues;
- polarity-aware cues;
- facet scoring;
- hard-exclusion behavior;
- subject/referent binding;
- component isolation;
- missing-component recovery;
- external-knowledge restrictions;
- localized version repairs;
- frozen-artifact contracts;
- release and holdout execution contracts.

Representative files include:

```text
evals/test_evidence_candidate_selection.py
evals/test_multi_span_composition.py
evals/test_deterministic_direct_cues.py
evals/test_polarity_aware_direct_cues.py
evals/test_facet_scoring.py

evals/test_v0_26_component_isolation.py
evals/test_v0_26_parent_child_referent_binding.py
evals/test_v0_27_external_knowledge_repair.py
evals/test_v0_28_entailment_discipline.py
evals/test_v0_29_r5_localized_guards.py
```

A core philosophy is:

```text
LLM behavior may be stochastic
but the contracts around it should be as deterministic and reviewable as possible
```

---

# 10. System-level recommender evaluation

The system-quality code lives in:

```text
evals/system_evaluation/
```

The main benchmark contract is:

```text
semantic_relevance_system_eval.v1.0.0.json
```

It evaluates semantic retrieval in an isolated mode:

```text
category        = All
tone            = All
initial_top_k   = 50
final_top_k     = 10
```

This isolates semantic recommendation quality from category filtering and
emotion-based reordering.

The benchmark uses:

```text
12 frozen queries
10 recommendations/query
120 query-book pairs
```

The query file is:

```text
evals/datasets/semantic_relevance_system_eval_queries.v1.0.0.csv
```

Human relevance scores are explicitly disabled for this system benchmark.

---

# 11. System metrics

The system contract defines eight primary quality dimensions.

## Macro mean relevance@10

For each query:

```text
mean(top 10 judge scores)
```

Then average the 12 query means equally.

## Clear/strong rate@10

```text
fraction with judge_score >= 3
```

## Irrelevant rate@10

```text
fraction with judge_score == 0
```

## HitRate@5

Fraction of queries with at least one clear/strong result in ranks 1-5.

## Top-1 clear/strong rate

Fraction of queries whose first recommendation has score >= 3.

## MRR@10

Mean reciprocal rank of the first clear/strong recommendation.

## nDCG@10

Graded nDCG using:

```text
gain = 2^score - 1
```

The ideal ranking is the same returned candidate set sorted by judged
relevance.

## Catastrophic query rate

A query is catastrophic when:

```text
mean relevance@10 < 1.5
```

This keeps severe per-query failures visible even when aggregate metrics look
reasonable.

---

# 12. Statistical uncertainty

The 10 recommendations from one query are correlated, so recommendation rows
are not treated as independent statistical observations.

The system benchmark uses:

```text
resampling unit      = query
bootstrap replicates = 5,000
confidence level     = 95%
method               = percentile cluster bootstrap
```

During a bootstrap sample, the full ranked list for a selected query remains
together.

This produces uncertainty across the 12 benchmark intents without pretending
that 120 recommendation rows are 120 independent experiments.

The contract explicitly warns that 12 independent query clusters is small, so
confidence intervals can be wide.

---

# 13. Frozen release gates

The initial system contract preregisters eight gates:

| Gate | Threshold |
|---|---:|
| Macro mean relevance@10 | >= 2.50 |
| Macro-mean 95% CI lower bound | >= 2.00 |
| Clear/strong rate@10 | >= 60% |
| Irrelevant rate@10 | <= 10% |
| HitRate@5 | >= 90% |
| Top-1 clear/strong rate | >= 75% |
| nDCG@10 | >= 0.85 |
| Catastrophic query rate | 0% |

These gates are not weakened after a failed run.

That policy turns a failed benchmark into useful evidence instead of allowing
the benchmark definition to drift toward the current implementation.

---

# 14. System-evaluation workflow

The source code separates collection, judge execution, and analysis.

For v1/v2, the relevant classes of files are:

```text
freeze_semantic_relevance_system_eval.py
collect_semantic_relevance_system_recommendations*.py
score_frozen_semantic_relevance_recommendations*.py
analyze_semantic_relevance_system_eval*.py
compare_semantic_relevance_system_eval_v1_v2.py
render_semantic_relevance_system_eval_dashboard*.py
```

Frozen JSON inputs/locks in the same directory link each stage to exact
artifacts.

Conceptually:

```text
freeze contract
      |
      v
collect recommendations
      |
      v
freeze recommendation artifact
      |
      v
run calibrated judge
      |
      v
freeze judge scores
      |
      v
analyze metrics + bootstrap CIs
      |
      v
apply frozen release gates
```

Analysis scripts make zero judge calls.

---

# 15. Provenance and fail-closed execution

Important stages verify SHA-256 hashes before continuing.

Typical pinned inputs include:

- recommender source;
- benchmark queries;
- system contract;
- judge implementation;
- deterministic scoring implementation;
- facet specification;
- judge configuration;
- recommendation artifact;
- judge-input artifact;
- judge-score artifact.

If a required artifact changes after freeze, the stage fails rather than
silently evaluating a different system.

The source also records Git commit identifiers where the lifecycle requires
them.

This makes statements such as:

```text
"v2 improved relative to v1"
```

traceable to specific source and evidence artifacts.

---

# 16. v1 failure investigation

The first frozen recommender benchmark failed all eight system release gates.

The failure investigation did not immediately retune embeddings or the judge.

Instead it traced recommendation membership and rank flow through the
application.

The investigation found a deterministic implementation defect:

```python
books[books["isbn13"].isin(books_list)]
```

preserved membership but not vector-store ordering.

The subsequent top-K truncation therefore used metadata DataFrame order rather
than semantic similarity order.

The issue was proven with a deterministic regression contract:

```text
evals/system_evaluation/test_semantic_retrieval_order_regression.py
```

The test deliberately makes vector order differ from metadata order and checks
that the production function preserves the vector ranking.

The same test was intended to fail before the repair and pass unchanged after
the repair.

---

# 17. v2 controlled experiment

v2 changed one engineering boundary:

> preserve vector-store semantic order while joining candidate ISBNs back to
> metadata.

The comparison deliberately kept the benchmark constant:

```text
same 12 queries
same initial_top_k = 50
same final_top_k = 10
same judge
same rubric
same scoring logic
same bootstrap method
same release gates
```

The source pins the v2 recommender and links it back to the v1 baseline.

The paired comparison implementation is:

```text
evals/system_evaluation/
compare_semantic_relevance_system_eval_v1_v2.py
```

It uses:

```text
matched query as the paired unit
5,000 paired bootstrap replicates
win / tie / loss by query
frozen gate transitions
```

The comparison asks:

> Did v2 improve over v1?

The standalone gate result asks:

> Is v2 good enough?

Those are different questions.

v2 improved materially, but the frozen absolute release decision remained
`FAIL`.

Detailed evidence is documented in:

```text
evals/reports/
semantic_relevance_system_eval_v2_post_analysis_and_v3_reranker_rationale.md
```

---

# 18. Active top-50 diagnostic

The top-50 diagnostic extends the system-evaluation code with:

```text
evals/system_evaluation/
    collect_semantic_relevance_top50_candidates.py
    prepare_semantic_relevance_top50_score_reuse.py
    score_semantic_relevance_top50_novel_candidates.py
```

The purpose is to answer:

> Does the unchanged v2 top-50 candidate pool contain enough relevant material
> for a second-stage reranker to be a credible next experiment?

It does **not** change the embedding model, candidate depth, judge, or frozen
release gates.

---

## Checkpoint A — candidate reconstruction

Checkpoint A reconstructs:

```text
12 queries x 50 vector candidates = 600 query-book pairs
```

The hard reproducibility anchor is:

```text
current vector ranks 1-10
==
frozen v2 top 10
```

for all 12 queries.

Historical v1 deep-tail containment is recorded as a diagnostic signal rather
than a hard gate because historical ranks 11-50 were not frozen.

The successful frozen candidate pool contains:

```text
600 / 600 unique query-book pairs
12 / 12 frozen v2 top-10 reproductions
```

---

## Checkpoint B — score reuse

Checkpoint B reuses historical scores only when the source evidence is
compatible.

The join identity is:

```text
(query_id, isbn13)
```

Reuse also checks the frozen judge payload rather than trusting identity alone.

For the current candidate pool:

```text
historical shared v1/v2 pairs   = 24
historical score conflicts      = 0
payload mismatches              = 0

reused historical judgments    = 213
novel query-book pairs          = 387
```

This reduces unnecessary model calls and avoids injecting additional
stochastic rejudging into already-scored cases.

---

## Checkpoint C — novel-only judging

Checkpoint C uses:

```text
evals/system_evaluation/
score_semantic_relevance_top50_novel_candidates.py
```

It scores only the 387 novel cases with the same pinned system-evaluation judge.

The stage is restart-safe:

- every completed case is persisted immediately;
- completed IDs are validated against the frozen novel input;
- rerunning resumes without rejudging completed cases;
- aggregate/oracle metrics are not calculated during scoring.

When complete, it combines:

```text
213 reused scores
+
387 new scores
=
600 scored candidates
```

The complete 600-row score map becomes the input for the later headroom
analysis.

---

## Checkpoint D — planned headroom analysis

The diagnostic design then evaluates relevance by retrieval depth and builds a
diagnostic oracle top 10.

The oracle selects the highest-scoring candidates from the frozen 50.

It is **not production ranking logic**. It uses evaluation labels and exists
only to estimate the candidate pool's theoretical ceiling.

The intended interpretation is:

```text
oracle much better than v2
    -> ranking / selection headroom exists

oracle still weak
    -> candidate generation itself is limiting

different behavior by query
    -> mixed failure mode
```

The diagnostic conclusion is expected to distinguish:

```text
RERANKER_SUPPORTED
RETRIEVAL_LIMITED
MIXED
```

These are engineering diagnostic labels, not release decisions.

---

# 19. Benchmark contamination boundary

The original 12 system queries have now been used for:

- v1 evaluation;
- v1 failure investigation;
- v2 controlled comparison;
- paired v1/v2 analysis;
- recommendation-level diagnosis;
- top-50 candidate diagnosis.

They should therefore be considered a:

```text
development / diagnostic system benchmark
```

They remain valuable for regression testing and causal engineering analysis.

They should not be presented as a pristine final holdout for a future v3
generalization claim.

After a future v3 architecture is selected and frozen, a separate unseen
system-level holdout should be used for final external validation.

---

# 20. Baselines, releases, and reports

The repository separates several kinds of long-lived evidence.

## `baselines/`

Contains frozen historical reference outputs used for controlled comparisons.

## `releases/`

Contains release-candidate manifests, validation protocols, and final-evaluation
closeout records.

Examples in the current tree include v0.26, v0.27, and v0.28 release artifacts.

## `reports/`

Contains human-readable engineering summaries.

Current reports include:

```text
semantic_relevance_calibration_progress.md
semantic_relevance_system_eval_plan.v1.0.0.md
semantic_relevance_system_eval_v1_failure_investigation.md
semantic_relevance_system_eval_v2_post_analysis_and_v3_reranker_rationale.md
semantic_relevance_top50_diagnostic_design.md
```

Reports explain *why* a change was made; machine-readable contracts and locks
preserve *exactly what* was run.

---

# 21. Runtime artifacts

Execution outputs are written under:

```text
evals/runs/
```

and that directory is ignored by Git.

A run can contain artifacts such as:

```text
recommendations.csv
judge_input.csv
judge_scores.csv
query_metrics.csv
aggregate_metrics.json
bootstrap_confidence_intervals.json
release_gate_result.json
analysis_metadata.json
*_lock.json
system_evaluation_report.md
system_evaluation_dashboard.html
```

The top-50 diagnostic adds artifacts such as:

```text
top50_candidates.csv
top50_score_reuse_map.csv
top50_novel_judge_input.csv
top50_judge_scores_new.csv
top50_judge_scores_complete.csv
```

Do not edit frozen run artifacts in place.

A changed experiment should create a new version or run namespace.

---

# 22. Running the code

The project uses `uv`.

Install the locked environment from the repository root:

```powershell
uv sync
```

The current project requires Python:

```text
>= 3.12, < 3.14
```

Evaluation-related dependencies in `pyproject.toml` include:

```text
deepeval
pandas
scikit-learn
openpyxl
ollama
langchain-ollama
```

The broader application additionally uses ChromaDB/LangChain, Transformers,
PyTorch, Gradio, and the preprocessing notebooks described in the root README.

---

# 23. Ollama requirements

The system evaluation uses local Ollama-backed components.

The semantic recommender uses:

```text
nomic-embed-text
```

for vector embeddings.

The v0.29.0-r5 judge configuration uses:

```text
jeffnyman/ts-evaluator
```

through:

```text
http://localhost:11434
```

with:

```text
temperature = 0.0
```

Ollama must therefore be running for workflows that construct/query the vector
store or execute new LLM judge calls.

Pure analysis of already-frozen score files does not require judge execution.

---

# 24. PowerShell wrappers

Several important workflows have root-level PowerShell wrappers.

Examples on the current codebase include:

```text
prepare_semantic_relevance_system_eval.ps1
collect_semantic_relevance_system_recommendations.ps1
collect_semantic_relevance_system_recommendations_v2.ps1
score_semantic_relevance_system_eval.ps1
score_semantic_relevance_system_eval_v2.ps1
analyze_semantic_relevance_system_eval.ps1
analyze_semantic_relevance_system_eval_v2.ps1
collect_semantic_relevance_top50_checkpoint_a.ps1
score_semantic_relevance_top50_checkpoint_c.ps1
```

These wrappers are useful because they commonly add:

- explicit stage banners;
- error handling;
- output capture;
- sleep prevention for long model runs;
- stop boundaries between evaluation stages.

When a wrapper exists, prefer it over invoking internal implementation scripts
ad hoc.

Historical final holdouts should **not** be rerun merely because their runner
still exists in the repository.

---

# 25. Capturing long outputs

Long judge runs can exceed terminal scrollback.

A standard pattern in this project is:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\some_evaluation_stage.ps1 2>&1 |
  Tee-Object -FilePath .\stage_output.txt
```

For long-running judge stages, the persisted CSV/JSON artifacts are the source
of truth. Console output is useful supporting evidence, not a substitute for
the frozen artifacts.

---

# 26. How to investigate failures

The preferred failure workflow is:

```text
evaluation failure
      |
      v
classify the failure
      |
      +--> human-label problem?
      +--> rubric/facet ambiguity?
      +--> judge semantic defect?
      +--> deterministic implementation defect?
      +--> stochastic instability?
      +--> retrieval/candidate-recall problem?
      +--> ranking problem?
      +--> analysis/tooling defect?
      |
      v
form one bounded hypothesis
      |
      v
write/retain regression evidence
      |
      v
make the smallest attributable change
      |
      v
freeze a new candidate
      |
      v
rerun the controlled evaluation
```

Avoid changing several architectural layers at once when causal attribution
matters.

---

# 27. Things this framework deliberately avoids

The evaluation process is designed to avoid several common AI-testing mistakes.

## Changing the gate after seeing the score

Release thresholds are frozen before execution.

## Reusing consumed holdouts as fresh evidence

Once inspected, cases become development/diagnostic evidence.

## Treating recommendation rows as independent samples

Bootstrap resampling is query-level.

## Retuning the judge because the recommender failed

Judge calibration and system evaluation are separate lifecycles.

## Hiding failed versions

v1 and v2 failed runs remain useful evidence and are not overwritten.

## Rejudging known cases unnecessarily

The top-50 diagnostic reuses provenance-compatible frozen scores.

## Using oracle labels as production logic

The planned oracle is an analysis ceiling only.

---

# 28. Suggested reading order

For someone reviewing the evaluation portion of the project for the first time:

```text
1. README.md                         <- this file
2. EVALUATION_VERSIONING.md
3. semantic_relevance_facet_scoring.py
4. semantic_relevance_facet_judge.py
5. CHANGELOG.md
6. reports/semantic_relevance_calibration_progress.md
7. reports/semantic_relevance_system_eval_plan.v1.0.0.md
8. reports/semantic_relevance_system_eval_v1_failure_investigation.md
9. reports/semantic_relevance_system_eval_v2_post_analysis_and_v3_reranker_rationale.md
10. system_evaluation/
11. reports/semantic_relevance_top50_diagnostic_design.md
```

The many historical version-specific files can then be consulted when tracing a
particular repair or evidence decision.

---

# 29. Current engineering state

Current engineering state at this stage of the evaluation:

```text
recommender v1
    -> frozen system benchmark: FAIL

confirmed application defect
    -> vector similarity ordering lost during metadata join

recommender v2
    -> order-preservation repair
    -> materially better than v1
    -> absolute frozen release result still FAIL

post-v2 diagnosis
    -> possible ranking + candidate-pool limitations

top-50 diagnostic
    -> candidate pool frozen
    -> 213 historical scores safely reused
    -> 387 novel cases being judged
    -> headroom/oracle analysis follows after scoring
```

The next architecture is intentionally not assumed in advance.

The current diagnostic is designed to decide whether the evidence supports a
top-50 reranker, a candidate-generation change, or a mixture of both.

---

# 30. Related application documentation

For:

- dataset preparation;
- zero-shot Fiction/Nonfiction classification;
- emotion analysis;
- Ollama embeddings;
- Chroma vector search;
- Gradio UI;
- local application setup;

see the repository root:

```text
../README.md
```

This `evals/` README is intentionally focused on **evaluation engineering,
evidence discipline, statistical analysis, and AI-system testing** rather than
duplicating the application documentation.
