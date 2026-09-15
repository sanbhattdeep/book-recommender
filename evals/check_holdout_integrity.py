"""
Pre-flight integrity check for the frozen v0.15 final holdout.

This script does not call the judge and does not print human labels, titles,
descriptions, or per-case gold information.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"

HOLDOUT_FILE = EVALS_DIR / "datasets" / "semantic_relevance_final_holdout.v1.0.0.csv"
SPLIT_MANIFEST_FILE = EVALS_DIR / "datasets" / "semantic_relevance_unseen_split_manifest.v1.0.0.json"
EXECUTION_MANIFEST_FILE = EVALS_DIR / "datasets" / "semantic_relevance_holdout_execution_manifest.v1.0.0.json"
CONFIG_FILE = EVALS_DIR / "judge_configs" / "semantic_relevance_judge.v0.15.0.json"
FACET_FILE = EVALS_DIR / "facets" / "semantic_relevance" / "semantic_relevance_query_facets.v0.5.0.json"
SCORING_FILE = EVALS_DIR / "semantic_relevance_facet_scoring.py"
JUDGE_FILE = EVALS_DIR / "semantic_relevance_facet_judge.py"
RUBRIC_FILE = EVALS_DIR / "rubrics" / "semantic_relevance" / "semantic_relevance_rubric.v0.1.0.json"

EXPECTED_HOLDOUT_SHA256 = "3257da52c9b104be0122774bc9bead5b0f6925d1ad2cfc29931283b9408a31e0"
EXPECTED_SPLIT_MANIFEST_SHA256 = "cf1c302c4706b1898d5725abaa5bf25e16887fa0d4553b82ebf9be74123be7cb"
EXPECTED_CONFIG_SHA256 = "d12bfc34eb9356e8982e9b36935f6b6c1baf9a44f837e6d7e1fb30b99db8383e"
EXPECTED_FACET_SHA256 = "5f1f9608f2e590402ed5f778fc3aa05b85f441ebeaec0e94c7bd333c3892824d"
EXPECTED_SCORING_SHA256 = "2261a8c845c41f4a8d891508604a52dee4c8ab72314e4f0dfb57acb5c6830fca"
EXPECTED_JUDGE_SHA256 = "602ac766826b845cc57911f374dba8835fce051199076e3341a07e32a595aa25"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_holdout_contract(quiet: bool = False) -> dict:
    required = [
        HOLDOUT_FILE,
        SPLIT_MANIFEST_FILE,
        EXECUTION_MANIFEST_FILE,
        CONFIG_FILE,
        FACET_FILE,
        SCORING_FILE,
        JUDGE_FILE,
        RUBRIC_FILE,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required holdout artifact(s): " + ", ".join(missing))

    hashes = {
        "holdout": sha256(HOLDOUT_FILE),
        "split_manifest": sha256(SPLIT_MANIFEST_FILE),
        "judge_config": sha256(CONFIG_FILE),
        "facet_spec": sha256(FACET_FILE),
        "scoring_module": sha256(SCORING_FILE),
        "judge_module": sha256(JUDGE_FILE),
    }
    expected_hashes = {
        "holdout": EXPECTED_HOLDOUT_SHA256,
        "split_manifest": EXPECTED_SPLIT_MANIFEST_SHA256,
        "judge_config": EXPECTED_CONFIG_SHA256,
        "facet_spec": EXPECTED_FACET_SHA256,
        "scoring_module": EXPECTED_SCORING_SHA256,
        "judge_module": EXPECTED_JUDGE_SHA256,
    }
    mismatched = {
        key: {"actual": hashes[key], "expected": expected_hashes[key]}
        for key in hashes
        if hashes[key] != expected_hashes[key]
    }
    if mismatched:
        raise ValueError("Frozen holdout artifact hash mismatch: " + json.dumps(mismatched, indent=2))

    split = load_json(SPLIT_MANIFEST_FILE)
    execution = load_json(EXECUTION_MANIFEST_FILE)
    config = load_json(CONFIG_FILE)
    facets = load_json(FACET_FILE)
    rubric = load_json(RUBRIC_FILE)

    if split.get("split_version") != "1.0.0":
        raise ValueError("Unexpected split manifest version.")
    if split.get("judge_outputs_used_for_split") is not False:
        raise ValueError("Split manifest no longer asserts judge_outputs_used_for_split=false.")
    if execution.get("judge_behavior_frozen") is not True:
        raise ValueError("Execution manifest must assert judge_behavior_frozen=true.")
    if config.get("version") != "0.15.0" or config.get("facet_spec_version") != "0.5.0":
        raise ValueError("Judge config version pins do not match frozen holdout contract.")
    if facets.get("version") != "0.5.0":
        raise ValueError("Facet spec version is not 0.5.0.")
    if str(rubric.get("version")) != "0.1.0":
        raise ValueError("Rubric version is not 0.1.0.")

    with HOLDOUT_FILE.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 30:
        raise ValueError(f"Final holdout must contain exactly 30 rows; found {len(rows)}.")

    required_columns = {
        "case_id", "query_id", "query", "isbn13", "title", "authors", "description",
        "dataset_version", "rubric_version", "human_score", "human_reason", "review_status",
    }
    columns = set(rows[0].keys()) if rows else set()
    missing_columns = required_columns - columns
    if missing_columns:
        raise ValueError(f"Holdout is missing required columns: {sorted(missing_columns)}")

    case_ids = [str(r["case_id"]) for r in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Final holdout contains duplicate case IDs.")
    if not all(case_id.startswith("U_") for case_id in case_ids):
        raise ValueError("Final holdout case IDs must use U_ prefix.")

    isbns = [str(r["isbn13"]).strip() for r in rows]
    if len(isbns) != len(set(isbns)):
        raise ValueError("Final holdout contains duplicate ISBNs.")
    if not all(len(v) == 13 and v.isdigit() for v in isbns):
        raise ValueError("Every final-holdout ISBN must be an exact 13-digit text value.")

    if {str(r["dataset_version"]) for r in rows} != {"1.0.0"}:
        raise ValueError("Holdout dataset_version must be 1.0.0 on every row.")
    if {str(r["rubric_version"]) for r in rows} != {"0.1.0"}:
        raise ValueError("Holdout rubric_version must be 0.1.0 on every row.")
    if not all(str(r["review_status"]).strip() == "LABELLED" for r in rows):
        raise ValueError("Every holdout row must be LABELLED before the final run.")
    if not all(str(r["human_score"]).strip() in {"0", "1", "2", "3", "4"} for r in rows):
        raise ValueError("Holdout contains missing/invalid human scores.")
    if not all(str(r["human_reason"]).strip() for r in rows):
        raise ValueError("Holdout contains a missing human reason.")

    expected_holdout_ids = set(split["final_holdout"]["case_ids"])
    validation_ids = set(split["validation"]["case_ids"])
    actual_ids = set(case_ids)
    if actual_ids != expected_holdout_ids:
        raise ValueError("Final holdout case-ID set does not exactly match the frozen split manifest.")
    if actual_ids & validation_ids:
        raise ValueError("Final holdout overlaps the validation split.")
    if split["final_holdout"].get("cases") != 30:
        raise ValueError("Split manifest does not declare 30 final-holdout cases.")

    summary = {
        "status": "PASS",
        "cases": 30,
        "split_version": "1.0.0",
        "dataset_version": "1.0.0",
        "rubric_version": "0.1.0",
        "judge_config_version": "0.15.0",
        "facet_spec_version": "0.5.0",
        "no_validation_overlap": True,
        "judge_outputs_used_for_split": False,
        "frozen_hashes_match": True,
    }
    if not quiet:
        print("Final holdout integrity check")
        print("-----------------------------")
        print("PASS  frozen artifact hashes")
        print("PASS  exact 30-case holdout membership")
        print("PASS  no validation/holdout overlap")
        print("PASS  13-digit unique ISBNs")
        print("PASS  labels complete without printing gold values")
        print("PASS  v0.15.0 / facets v0.5.0 / rubric v0.1.0 pins")
        print()
        print("Holdout is ready for one final frozen evaluation.")
    return summary


def main() -> None:
    validate_holdout_contract(quiet=False)


if __name__ == "__main__":
    main()
