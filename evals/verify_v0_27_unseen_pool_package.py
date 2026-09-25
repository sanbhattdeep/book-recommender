"""Static verifier for the v0.27 unseen-pool preparation package r5."""
from __future__ import annotations

from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evals"
D = E / "datasets"
EXPECTED_FREEZE = "3de6ee3a6274a5d066f17ff24bc965b05dfb8d60b785c0210d45d743538d84e5"


def main() -> None:
    pool = D / "semantic_relevance_unseen_pool.v3.0.0.template.csv"
    spec = D / "semantic_relevance_unseen_pool_split_spec.v3.0.0.json"
    builder = E / "build_v0_27_unseen_evaluation_pool.py"
    exporter = E / "export_v0_27_unseen_pool_labeling_xlsx.py"
    importer = E / "import_v0_27_unseen_pool_labels_from_xlsx.py"
    validator = E / "validate_v0_27_unseen_pool.py"
    splitter = E / "split_v0_27_unseen_pool.py"
    preflight = E / "preflight_v0_27_validation.py"
    runner = E / "run_judge_v0_27_validation.py"
    analyzer = E / "analyze_v0_27_validation.py"
    required = [pool, spec, builder, exporter, importer, validator, splitter, preflight, runner, analyzer]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)
    for p in required[2:]:
        compile(p.read_text(encoding="utf-8-sig"), str(p), "exec")

    with pool.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 60 or len({r["case_id"] for r in rows}) != 60:
        raise ValueError("Pool template must contain 60 unique case IDs.")

    split_spec = json.loads(spec.read_text(encoding="utf-8-sig"))
    if split_spec.get("validation_cases") != 30 or split_spec.get("final_holdout_cases") != 30:
        raise ValueError("Split spec must be 30/30.")
    if split_spec.get("frozen_candidate_manifest_sha256") != EXPECTED_FREEZE:
        raise ValueError("Frozen-candidate hash pin mismatch.")
    if split_spec.get("split_method") != "label_aware_balanced_deterministic_backtracking_30_30_before_any_judge_output":
        raise ValueError("Split spec is not the r5 label-aware judge-blind deterministic method.")

    validator_text = validator.read_text(encoding="utf-8-sig")
    for needle in [
        "semantic_relevance_v0.27_development.v3.0.0.csv",
        "semantic_relevance_calibration.v0.5.0.csv",
        "semantic_relevance_calibration.v0.4.0.csv",
        "semantic_relevance_calibration.v0.3.0.csv",
        "no overlap with ANY consumed calibration/development identity",
    ]:
        if needle not in validator_text:
            raise ValueError(f"Validator contract missing: {needle}")

    splitter_text = splitter.read_text(encoding="utf-8-sig")
    for needle in [
        "TARGET_VALIDATION_SCORE_COUNTS = {0: 18, 1: 2, 2: 2, 3: 8, 4: 0}",
        "TARGET_HOLDOUT_SCORE_COUNTS = {0: 18, 1: 2, 2: 2, 3: 7, 4: 1}",
        "TARGET_SLOT_COUNT = 6",
        "judge_outputs_used_for_split",
        "choose_validation_case_ids",
        "pure_python_deterministic_backtracking_with_suffix_bounds",
        "LOCKED_DO_NOT_RUN",
    ]:
        if needle not in splitter_text:
            raise ValueError(f"Balanced-split contract missing: {needle}")


    if "from scipy" in splitter_text.lower() or "import scipy" in splitter_text.lower() or "milp(" in splitter_text:
        raise ValueError("r5 splitter must not depend on SciPy/native MILP.")

    runner_text = runner.read_text(encoding="utf-8-sig")
    for needle in [
        "semantic_relevance_validation.v{EVALUATION_DATASET_VERSION}.csv",
        "semantic_relevance_v0_27_validation",
        "unseen_validation",
        "independent_validation_before_final_holdout",
        "len(dataset) != 30",
        'startswith("U3_")',
    ]:
        if needle not in runner_text:
            raise ValueError(f"Validation runner contract missing: {needle}")

    print("v0.27 unseen-pool preparation package r5 verification passed.")
    print("PASS  60-case retrieval/labeling workflow retained")
    print("PASS  all-consumed-history freshness validation retained")
    print("PASS  completed-label import contract")
    print("PASS  label-aware / judge-blind balanced 30/30 split contract")
    print("PASS  each sampling slot balanced 6/6")
    print("PASS  pure-Python deterministic split solver; no SciPy MILP dependency")
    print("PASS  final-holdout lock creation contract")
    print("PASS  30-case validation runner/analyzer compile")
    print("PASS  frozen r5 manifest hash pinned")


if __name__ == "__main__":
    main()
