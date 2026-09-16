# Next unseen evaluation protocol after v0.16 development

The original 60-case unseen pool has now been fully consumed by v0.13–v0.16
development and the v0.15 final evaluation. It must not be reused for an
independent generalization claim.

## Recommended next pool

Keep the same 12 frozen query intents so the next test isolates candidate-level
generalization, but sample entirely new candidate books/descriptions.

Recommended size:

```text
12 queries × 5 candidates/query = 60 new cases
```

For each query, sample from several retrieval depths plus one random negative,
while excluding every ISBN already present in:

- original calibration/development datasets;
- the consumed v1.0.0 unseen pool;
- any targeted v0.16 synthetic/candidate tests.

Do not use retrieval rank as a label.

## Labeling

Human-label all 60 before running v0.16.

The human reviewer sees only:

- query
- title
- authors
- supplied description
- frozen rubric

The reviewer must not see:

- retrieval rank
- judge outputs
- old labels for similar books
- v0.16 audit traces

Double-label a subset if practical to estimate human-human agreement.

## Split before judge output

Use a fixed seed to make a 30/30 split, balanced by human score and sampling
position where possible:

```text
30 development-validation
30 final holdout
```

Record the split manifest and hash the source files before any judge run.

## Evaluation sequence

1. Freeze v0.16.0.
2. Run the new 30-case validation split.
3. If it exposes a reproducible failure that merits v0.17, use only validation
   cases for that development.
4. Freeze the resulting judge version.
5. Run the fresh final holdout exactly once.
6. Report the final result even if it is worse than expected.

Never relabel or repeatedly retest the fresh final holdout to make metrics pass.
