"""Build the 90-case v0.24 development set from already-consumed data only.

Inputs expected in the repository:
- 60-case v0.18 post-holdout development set
- 30-case fresh-unseen validation split that v0.18 already evaluated and failed

The untouched fresh holdout is NEVER read by this script.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import pandas as pd

EVALS = Path(__file__).resolve().parent
DATASETS = EVALS / "datasets"
OLD = DATASETS / "semantic_relevance_v0.18_development.v1.0.0.csv"
FRESH_VALIDATION = DATASETS / "semantic_relevance_fresh_validation.v2.0.0.csv"
OUT = DATASETS / "semantic_relevance_v0.24_development.v2.1.0.csv"
MANIFEST = DATASETS / "semantic_relevance_v0.24_development_manifest.v1.0.0.json"
LABEL_REVISIONS = DATASETS / "semantic_relevance_label_revisions.v1.0.0.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for path in [OLD, FRESH_VALIDATION]:
        if not path.exists():
            raise FileNotFoundError(f"Required consumed-development source not found: {path}")

    old = pd.read_csv(OLD, dtype={"case_id": str, "isbn13": str}, keep_default_na=False)
    fresh = pd.read_csv(FRESH_VALIDATION, dtype={"case_id": str, "isbn13": str}, keep_default_na=False)

    if len(old) != 60 or not old["case_id"].str.startswith("U_").all():
        raise ValueError("Expected exactly 60 historical U_ development cases.")
    if len(fresh) != 30 or not fresh["case_id"].str.startswith("U2_").all():
        raise ValueError("Expected exactly 30 consumed U2_ fresh-validation cases.")

    required = {
        "case_id", "query_id", "query", "title", "authors", "description",
        "isbn13", "human_score", "human_reason", "review_status", "rubric_version"
    }
    for name, frame in [("old", old), ("fresh", fresh)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} source missing required columns: {sorted(missing)}")
        if not frame["human_score"].isin([0,1,2,3,4]).all():
            raise ValueError(f"{name} source has invalid human_score values.")
        if not (frame["review_status"].astype(str) == "LABELLED").all():
            raise ValueError(f"{name} source contains non-LABELLED rows.")
        if set(frame["rubric_version"].astype(str)) != {"0.1.0"}:
            raise ValueError(f"{name} source is not pinned to rubric v0.1.0.")

    if set(old["case_id"]) & set(fresh["case_id"]):
        raise ValueError("Case ID overlap between old development and consumed validation.")
    if set(old["isbn13"]) & set(fresh["isbn13"]):
        raise ValueError("ISBN overlap between old development and consumed validation.")

    old = old.copy(); fresh = fresh.copy()
    old["development_source"] = "historical_consumed_60"
    fresh["development_source"] = "consumed_fresh_validation_30"

    # Union schemas without losing provenance fields that exist in either source.
    columns = list(dict.fromkeys(list(old.columns) + list(fresh.columns)))
    old = old.reindex(columns=columns, fill_value="")
    fresh = fresh.reindex(columns=columns, fill_value="")
    combined = pd.concat([old, fresh], ignore_index=True)

    # Apply explicit, auditable blind-review label revisions only after the
    # two consumed source sets have been assembled. The original source files
    # remain untouched. Every revision checks the previously recorded score
    # before changing it so accidental drift fails loudly.
    revisions = json.loads(LABEL_REVISIONS.read_text(encoding="utf-8"))
    combined["previous_human_score"] = ""
    combined["label_revision_method"] = ""
    combined["label_revision_reason"] = ""
    combined["label_revision_status"] = ""
    for revision in revisions.get("revisions", []):
        case_id = str(revision["case_id"])
        mask = combined["case_id"].astype(str).eq(case_id)
        if int(mask.sum()) != 1:
            raise ValueError(f"Label revision case must occur exactly once: {case_id}")
        current = int(combined.loc[mask, "human_score"].iloc[0])
        expected = int(revision["previous_human_score"])
        if current != expected:
            raise ValueError(
                f"Label revision {case_id} expected prior score {expected}, found {current}."
            )
        combined.loc[mask, "previous_human_score"] = str(current)
        combined.loc[mask, "human_score"] = int(revision["new_human_score"])
        combined.loc[mask, "human_reason"] = str(revision["reason"])
        combined.loc[mask, "label_revision_method"] = str(revision["review_method"])
        combined.loc[mask, "label_revision_reason"] = str(revision["reason"])
        combined.loc[mask, "label_revision_status"] = str(revision["review_status"])

    combined["dataset_version"] = "2.1.0"
    combined["rubric_version"] = "0.1.0"

    if len(combined) != 90 or combined["case_id"].nunique() != 90:
        raise ValueError("v0.24 development output must contain 90 unique cases.")
    if combined["isbn13"].nunique() != 90:
        raise ValueError("v0.24 development output must contain 90 unique ISBNs.")

    combined.to_csv(OUT, index=False, encoding="utf-8")
    manifest = {
        "version": "1.0.0",
        "dataset_version": "2.1.0",
        "rubric_version": "0.1.0",
        "case_count": 90,
        "sources": [
            {"path": str(OLD.relative_to(EVALS.parent)), "cases": 60, "sha256": sha256(OLD)},
            {"path": str(FRESH_VALIDATION.relative_to(EVALS.parent)), "cases": 30, "sha256": sha256(FRESH_VALIDATION)},
        ],
        "label_revisions": {
            "path": str(LABEL_REVISIONS.relative_to(EVALS.parent)),
            "sha256": sha256(LABEL_REVISIONS),
            "count": len(revisions.get("revisions", [])),
        },
        "output": str(OUT.relative_to(EVALS.parent)),
        "output_sha256": sha256(OUT),
        "holdout_read": False,
        "methodology_note": "All 90 rows are consumed development evidence. The fresh 30-case final holdout remains untouched and is not an input."
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print("v0.24 development dataset built")
    print("--------------------------------")
    print(f"Historical consumed cases: {len(old)}")
    print(f"Consumed fresh validation: {len(fresh)}")
    print(f"Total development cases:   {len(combined)}")
    print(f"Label revisions applied:   {len(revisions.get('revisions', []))}")
    print(f"Unique ISBNs:              {combined['isbn13'].nunique()}")
    print("Fresh holdout read:        NO")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()
