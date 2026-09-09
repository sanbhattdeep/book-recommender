# Evaluation Versioning

Track three things independently:

1. **Rubric version** — what "relevance" means and how scores 0–4 are assigned.
2. **Calibration dataset version** — which human-labelled query-book pairs define the gold set.
3. **Judge config version** — model + evaluation prompt/settings used to apply the rubric.

Recommended identifiers:

- `rubric: 0.2.0`
- `calibration_dataset: 0.1.0`
- `judge_config: 0.1.0`

## Rubric versioning

- **MAJOR**: score scale, evaluation unit, criterion meaning, or evidence rules change.
- **MINOR**: anchors/boundary examples/scoring guidance change in a way that may affect scores.
- **PATCH**: typo or editorial cleanup only.

## Calibration dataset versioning

- **MAJOR**: schema or target label definition changes.
- **MINOR**: cases are added/removed, or any gold human score changes.
- **PATCH**: metadata correction only; gold labels unchanged.

## Judge config versioning

- **MAJOR**: judge methodology or scoring scale changes.
- **MINOR**: model, prompt, rubric injection, temperature, or other behavior-affecting setting changes.
- **PATCH**: logging/non-behavioral configuration changes.

## Every evaluation run should record

`run_id`, timestamp, Git commit SHA, rubric version, calibration-dataset version,
judge-config version, judge model/provider, DeepEval version, Python version,
and the resulting agreement metrics.

This lets you reproduce statements such as:

> Judge config 0.1.0 achieved weighted kappa 0.78 against calibration dataset 0.1.0 using rubric 0.2.0 at Git commit abc1234.

Do not silently overwrite a versioned rubric or gold dataset. Create a new version instead.
