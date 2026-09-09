# Semantic relevance judge calibration v0.1.0

Add these files to the project:

```text
evals/
├── datasets/
│   └── semantic_relevance_calibration.v0.1.0.csv
├── rubrics/
│   └── semantic_relevance/
│       └── semantic_relevance_rubric.v0.1.0.json
├── judge_configs/
│   └── semantic_relevance_judge.v0.1.0.json
├── runs/
│   └── semantic_relevance/
├── run_judge_calibration.py
└── analyze_judge_calibration.py
```

## 1. Add dependencies

```powershell
uv add deepeval scikit-learn
```

Commit the resulting `pyproject.toml` and `uv.lock` changes.

## 2. Configure the judge API key

Put this in `.env.local` or `.env`:

```env
OPENAI_API_KEY=your_key_here
```

Do not commit the env file.

## 3. Run a five-case smoke test

```powershell
uv run python evals/run_judge_calibration.py --limit 5
```

The script prints the run directory, for example:

```text
evals/runs/semantic_relevance/20260908T071500Z
```

Inspect `judge_results.csv`.

It should contain only:

```text
case_id
judge_score
judge_reason
judge_cost
```

It should NOT contain `human_score` or `human_reason`.

## 4. Resume that same run

```powershell
uv run python evals/run_judge_calibration.py `
  --resume evals/runs/semantic_relevance/20260908T071500Z
```

The runner skips completed case IDs and writes each new result immediately.

## 5. Analyze agreement after all cases are scored

```powershell
uv run python evals/analyze_judge_calibration.py `
  --run evals/runs/semantic_relevance/20260908T071500Z
```

This produces:

- `calibration_summary.json`
- `human_vs_judge.csv`
- `confusion_matrix.csv`
- `agreement_by_human_score.csv`

The primary agreement statistic is **linear weighted Cohen's kappa** because
the 0–4 rubric is ordinal: adjacent disagreements should count less than
larger disagreements without assuming the scale is truly continuous.
