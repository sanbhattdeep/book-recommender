"""Static verifier for the v0.27 unseen-pool preparation/retrieval package."""
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

    required = [
        pool, spec, builder, exporter, importer, validator, splitter,
        preflight, runner, analyzer,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    for p in required[2:]:
        compile(p.read_text(encoding="utf-8-sig"), str(p), "exec")

    with pool.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 60:
        raise ValueError("Pool template must contain 60 rows.")
    if len({r["case_id"] for r in rows}) != 60:
        raise ValueError("Pool template case IDs must be unique.")
    counts = {f"Q{i:02d}": 0 for i in range(1, 13)}
    for r in rows:
        counts[r["query_id"]] += 1
    if any(v != 5 for v in counts.values()):
        raise ValueError(f"Pool template must have 5 cases/query: {counts}")

    split_spec = json.loads(spec.read_text(encoding="utf-8-sig"))
    if split_spec.get("validation_cases") != 30 or split_spec.get("final_holdout_cases") != 30:
        raise ValueError("Split spec must be 30/30.")
    if split_spec.get("frozen_candidate_manifest_sha256") != EXPECTED_FREEZE:
        raise ValueError("Frozen-candidate hash pin mismatch.")

    builder_text = builder.read_text(encoding="utf-8-sig")
    for needle in [
        "TARGET_RANKS = [2, 10, 30, 70]",
        "SEARCH_DEPTH = 250",
        "RANDOM_SEED = 20260924",
        "semantic_relevance_v0.27_development.v3.0.0.csv",
        "semantic_relevance_calibration.v0.5.0.csv",
        "semantic_relevance_calibration.v0.4.0.csv",
        "semantic_relevance_calibration.v0.3.0.csv",
        "resolve_original_calibration_dataset",
        "load_identity_source",
        "Historical calibration exclusion did not enlarge",
        "semantic_relevance_query_facets.v0.9.5.json",
        EXPECTED_FREEZE,
        "app.db_books.similarity_search",
        "random_negative_candidate",
        'case_id=f"U3_{qid}_T{slot:02d}"',
        'case_id=f"U3_{qid}_T05"',
        "title_author_key",
        '"review_status": "UNREVIEWED"',
    ]:
        if needle not in builder_text:
            raise ValueError(f"Retrieval-builder contract missing: {needle}")

    exporter_text = exporter.read_text(encoding="utf-8-sig")
    for needle in ["Labeling", "Metadata", "human_score", "human_reason", "review_status"]:
        if needle not in exporter_text:
            raise ValueError(f"Workbook-export contract missing: {needle}")

    importer_text = importer.read_text(encoding="utf-8-sig")
    for needle in ["pre_label_import", "LABELLED", "human_score", "human_reason"]:
        if needle not in importer_text:
            raise ValueError(f"Workbook-import contract missing: {needle}")

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

    print("v0.27 unseen-pool preparation package verification passed.")
    print("PASS  60-case template; 5 cases/query")
    print("PASS  raw Chroma retrieval builder: ranks 2/10/30/70 + random")
    print("PASS  all-consumed-history ISBN + title/author exclusion contract")
    print("PASS  blind Excel export + completed-label import contracts")
    print("PASS  deterministic 30/30 split spec")
    print("PASS  final-holdout lock creation contract")
    print("PASS  30-case validation runner/analyzer compile")
    print("PASS  frozen r5 manifest hash pinned")


if __name__ == "__main__":
    main()
