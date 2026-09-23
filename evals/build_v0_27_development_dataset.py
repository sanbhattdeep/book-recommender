"""Build the 120-case v0.27 post-holdout development set.

This script is intentionally POST-HOLDOUT. It combines:
- the 90-case v0.26 development set (already consumed), and
- the 30-case v0.26 final holdout (now consumed after its one-time evaluation),
then applies four explicitly recorded post-holdout human re-adjudications.

It NEVER rewrites the original final-holdout CSV or the frozen v0.26 result.
The output is development/regression evidence only and cannot support an
independent generalization claim for v0.27.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import pandas as pd

EVALS = Path(__file__).resolve().parent
DATASETS = EVALS / "datasets"
DEV90 = DATASETS / "semantic_relevance_v0.26_development.v2.2.0.csv"
CONSUMED_HOLDOUT30 = DATASETS / "semantic_relevance_fresh_holdout.v2.0.0.DO_NOT_RUN_YET.csv"
REVISIONS = DATASETS / "semantic_relevance_post_holdout_label_revisions.v1.0.0.json"
OUT = DATASETS / "semantic_relevance_v0.27_development.v3.0.0.csv"
MANIFEST = DATASETS / "semantic_relevance_v0.27_development_manifest.v1.0.0.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"case_id": str, "isbn13": str}, keep_default_na=False)


def main() -> None:
    for path in (DEV90, CONSUMED_HOLDOUT30, REVISIONS):
        if not path.exists():
            raise FileNotFoundError(f"Required source not found: {path}")

    dev = _load(DEV90)
    holdout = _load(CONSUMED_HOLDOUT30)

    required = {
        "case_id", "query_id", "query", "title", "authors", "description",
        "isbn13", "human_score", "human_reason", "review_status", "rubric_version"
    }
    for name, frame, expected_count in (("v0.26 development", dev, 90), ("consumed final holdout", holdout, 30)):
        if len(frame) != expected_count:
            raise ValueError(f"Expected {expected_count} rows in {name}, got {len(frame)}.")
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")
        if not frame["human_score"].astype(int).isin([0, 1, 2, 3, 4]).all():
            raise ValueError(f"{name} contains invalid human_score values.")
        if set(frame["rubric_version"].astype(str)) != {"0.1.0"}:
            raise ValueError(f"{name} is not pinned to rubric v0.1.0.")

    if set(dev["case_id"]) & set(holdout["case_id"]):
        raise ValueError("Case-ID overlap between v0.26 development and consumed final holdout.")

    # Ignore blank ISBNs when validating identity overlap.
    dev_isbn = {x for x in dev["isbn13"].astype(str) if x}
    holdout_isbn = {x for x in holdout["isbn13"].astype(str) if x}
    if dev_isbn & holdout_isbn:
        raise ValueError("ISBN overlap between v0.26 development and consumed final holdout.")

    dev = dev.copy()
    holdout = holdout.copy()
    dev["development_source"] = "consumed_v0_26_development_90"
    holdout["development_source"] = "consumed_v0_26_final_holdout_30"

    columns = list(dict.fromkeys(list(dev.columns) + list(holdout.columns)))
    dev = dev.reindex(columns=columns, fill_value="")
    holdout = holdout.reindex(columns=columns, fill_value="")
    combined = pd.concat([dev, holdout], ignore_index=True)

    revisions = json.loads(REVISIONS.read_text(encoding="utf-8"))
    combined["previous_human_score_post_holdout"] = ""
    combined["post_holdout_label_revision_version"] = ""
    combined["post_holdout_label_revision_method"] = ""
    combined["post_holdout_label_revision_status"] = ""

    for revision in revisions.get("revisions", []):
        case_id = str(revision["case_id"])
        mask = combined["case_id"].astype(str).eq(case_id)
        if int(mask.sum()) != 1:
            raise ValueError(f"Revision case must occur exactly once: {case_id}")
        current = int(combined.loc[mask, "human_score"].iloc[0])
        expected = int(revision["previous_human_score"])
        if current != expected:
            raise ValueError(
                f"Revision {case_id} expected previous score {expected}, found {current}."
            )
        combined.loc[mask, "previous_human_score_post_holdout"] = str(current)
        combined.loc[mask, "human_score"] = int(revision["new_human_score"])
        combined.loc[mask, "human_reason"] = str(revision["reason"])
        combined.loc[mask, "review_status"] = str(revision["review_status"])
        combined.loc[mask, "post_holdout_label_revision_version"] = str(revisions["version"])
        combined.loc[mask, "post_holdout_label_revision_method"] = str(revisions["review_method"])
        combined.loc[mask, "post_holdout_label_revision_status"] = str(revision["review_status"])

    combined["dataset_version"] = "3.0.0"
    combined["rubric_version"] = "0.1.0"

    if len(combined) != 120 or combined["case_id"].nunique() != 120:
        raise ValueError("v0.27 development output must contain 120 unique cases.")
    all_isbn = [x for x in combined["isbn13"].astype(str) if x]
    if len(set(all_isbn)) != len(all_isbn):
        raise ValueError("v0.27 development output contains duplicate nonblank ISBNs.")

    combined.to_csv(OUT, index=False, encoding="utf-8")

    manifest = {
        "version": "1.0.0",
        "dataset_version": "3.0.0",
        "rubric_version": "0.1.0",
        "case_count": 120,
        "methodological_status": "post_holdout_consumed_development_only",
        "sources": [
            {"path": str(DEV90.relative_to(EVALS.parent)), "cases": 90, "sha256": sha256(DEV90)},
            {"path": str(CONSUMED_HOLDOUT30.relative_to(EVALS.parent)), "cases": 30, "sha256": sha256(CONSUMED_HOLDOUT30)},
        ],
        "post_holdout_label_revisions": {
            "path": str(REVISIONS.relative_to(EVALS.parent)),
            "sha256": sha256(REVISIONS),
            "count": len(revisions.get("revisions", [])),
            "blind": False,
        },
        "output": str(OUT.relative_to(EVALS.parent)),
        "output_sha256": sha256(OUT),
        "final_holdout_immutability_note": (
            "The original v0.26.0 r6 final-holdout labels, outputs, metrics, and REVIEW result remain immutable. "
            "This 120-case dataset exists only for v0.27+ development after the holdout was consumed."
        ),
        "future_generalization_requirement": "Use a newly sampled and newly human-labelled unseen pool for any v0.27 generalization/release claim."
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    revised_ids = [str(x["case_id"]) for x in revisions.get("revisions", [])]
    print("v0.27 post-holdout development dataset built")
    print("--------------------------------------------")
    print(f"Consumed v0.26 development: {len(dev)}")
    print(f"Consumed final holdout:     {len(holdout)}")
    print(f"Total development cases:    {len(combined)}")
    print(f"Post-holdout revisions:     {len(revised_ids)}")
    print(f"Revision cases:             {', '.join(revised_ids)}")
    print("Independent holdout status: NONE (all 120 cases are consumed)")
    print(f"Output: {OUT}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
