"""Apply approved v0.29 human-label adjudications.

This is an adjudication-only step:
- NO judge calls.
- NO semantic/scoring changes.
- Source v6.0.0 is never overwritten.
- Exactly three approved human labels/reasons change.
- Two reviewed cases remain unchanged and are explicitly classified as
  judge defects.
- U4_Q09_T04 remains a judge over-promotion target even after its label
  revision from 1 to 2.

Outputs:
- semantic_relevance_v0.29_development.v6.1.0.csv
- semantic_relevance_v0.29_adjudications.v1.0.0.json
- semantic_relevance_v0.29_severe_review.adjudicated.v1.0.0.csv
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
E = R / "evals"
D = E / "datasets"
REL = E / "releases"

SOURCE = D / "semantic_relevance_v0.29_development.v6.0.0.csv"
SOURCE_REVIEW = D / "semantic_relevance_v0.29_severe_review.v1.0.0.csv"
CLOSEOUT = REL / "semantic_relevance_v0.28.0_r8_final_evaluation_closeout.json"

OUTPUT = D / "semantic_relevance_v0.29_development.v6.1.0.csv"
ADJUDICATIONS = D / "semantic_relevance_v0.29_adjudications.v1.0.0.json"
REVIEW_OUTPUT = D / "semantic_relevance_v0.29_severe_review.adjudicated.v1.0.0.csv"

EXPECTED_SOURCE_SHA = "c8087f1ed80fce92e2b666f61732f727e0bfe6342ad5960227cd8217b92122c6"
EXPECTED_REVIEW_SHA = "2155f3defb582b7830a7b9c04d79619725909515ab4d8e6faf1eb33f2a25eced"
EXPECTED_CLOSEOUT_SHA = "24f90865503f4572d54e168d7f2cca33647a7c9a3c294bfb1ab727fd544cb2de"

# Approved decisions from the human review of the five consumed final-holdout
# severe disagreements. Keep these explicit so the transformation is
# transparent and code-reviewable.
DECISIONS = {
    "U4_Q03_T01": {
        "classification": "label_spec_tension",
        "old_score": 0,
        "new_score": 2,
        "old_reason": "No match with personal growth and finding purpose",
        "new_reason": (
            "Meaningful match with personal growth through explicit self-improvement; "
            "finding purpose is not established."
        ),
        "judge_score_r8": 2,
        "semantic_repair_target": False,
        "rationale": (
            "The supplied description explicitly calls Bridget's year a perpetual "
            "quest for self-improvement. Under the frozen rubric, one central query "
            "component is meaningfully satisfied, so score 2 is more consistent than 0."
        ),
    },
    "U4_Q03_T02": {
        "classification": "judge_defect",
        "old_score": 2,
        "new_score": 2,
        "old_reason": "no match with personal growth but good match with finding purpose",
        "new_reason": "no match with personal growth but good match with finding purpose",
        "judge_score_r8": 0,
        "semantic_repair_target": True,
        "rationale": (
            "Human score remains 2. The description supplies substantive self-awakening/"
            "greatest-life/inner-peace evidence; r8's complete zero is the defect."
        ),
    },
    "U4_Q04_T02": {
        "classification": "judge_defect",
        "old_score": 0,
        "new_score": 0,
        "old_reason": "No match with A fantasy adventure involving magic and dangerous journeys",
        "new_reason": "No match with A fantasy adventure involving magic and dangerous journeys",
        "judge_score_r8": 2,
        "semantic_repair_target": True,
        "rationale": (
            "The description is realistic racing/banking suspense. Violent action and "
            "threat of death do not establish fantasy, magic, or a dangerous journey."
        ),
    },
    "U4_Q06_T01": {
        "classification": "label_spec_tension",
        "old_score": 2,
        "new_score": 0,
        "old_reason": (
            "Good match with A moving story about loss, and learning to live again "
            "but grief aspect is missing"
        ),
        "new_reason": (
            "No match under the frozen Q06 definition: despair and renewed appreciation "
            "of life are present, but no prior significant loss or grief is established."
        ),
        "judge_score_r8": 0,
        "semantic_repair_target": False,
        "rationale": (
            "Frozen Q06 requires prior grief/major loss for learning-to-live-again. "
            "The description establishes despair and renewed appreciation, not a prior "
            "significant loss or grief."
        ),
    },
    "U4_Q09_T04": {
        "classification": "label_spec_tension_plus_judge_overpromotion",
        "old_score": 1,
        "new_score": 2,
        "old_reason": "weak match with only authoritarian control",
        "new_reason": (
            "Meaningful match with authoritarian control through tyrannical rule, "
            "but resistance is not established."
        ),
        "judge_score_r8": 3,
        "semantic_repair_target": True,
        "rationale": (
            "Authoritarian control is central, so score 1 is too low; resistance is not "
            "established, so r8 score 3 is also too high. Score 2 is the adjudicated label."
        ),
    },
}

EXPECTED_REVIEW_IDS = set(DECISIONS)
EXPECTED_CHANGED_IDS = {"U4_Q03_T01", "U4_Q06_T01", "U4_Q09_T04"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)
    print("PASS ", msg)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    print("v0.29.0 human-label adjudication")
    print("--------------------------------")
    print("JUDGE EXECUTION: DISABLED")
    print("SEMANTIC CHANGES: NONE")

    for path in [SOURCE, SOURCE_REVIEW, CLOSEOUT]:
        require(path.exists(), f"required input exists: {path.relative_to(R)}")

    require(sha(SOURCE) == EXPECTED_SOURCE_SHA,
            "source v6.0.0 development dataset hash pinned")
    require(sha(SOURCE_REVIEW) == EXPECTED_REVIEW_SHA,
            "five-case severe-review artifact hash pinned")
    require(sha(CLOSEOUT) == EXPECTED_CLOSEOUT_SHA,
            "v0.28 r8 closeout hash pinned")

    closeout = load_json(CLOSEOUT)
    require(closeout.get("closeout_status") == "EVALUATED_NOT_FINAL_QUALIFIED",
            "r8 closeout status preserved")
    require(closeout.get("final_holdout_decision") == "FINAL_HOLDOUT_REVIEW",
            "r8 final-holdout decision preserved")

    source = pd.read_csv(SOURCE, dtype={"case_id": str, "isbn13": str}, encoding="utf-8-sig")
    review = pd.read_csv(SOURCE_REVIEW, dtype={"case_id": str}, encoding="utf-8-sig")

    require(len(source) == 240 and source["case_id"].is_unique,
            "source v6.0.0 contains 240 unique cases")
    require(len(review) == 5 and review["case_id"].is_unique,
            "review artifact contains five unique cases")
    require(set(review["case_id"]) == EXPECTED_REVIEW_IDS,
            "review artifact contains exactly the adjudicated five cases")

    # Verify the pre-adjudication values before touching any row.
    by_id = source.set_index("case_id", drop=False)
    review_by_id = review.set_index("case_id", drop=False)
    for case_id, spec in DECISIONS.items():
        require(case_id in by_id.index, f"source contains {case_id}")
        require(case_id in review_by_id.index, f"review contains {case_id}")
        require(int(by_id.loc[case_id, "human_score"]) == spec["old_score"],
                f"{case_id} source human_score matches expected pre-adjudication value")
        require(str(by_id.loc[case_id, "human_reason"]).strip() == spec["old_reason"],
                f"{case_id} source human_reason matches expected pre-adjudication text")
        require(int(review_by_id.loc[case_id, "judge_score"]) == spec["judge_score_r8"],
                f"{case_id} frozen r8 judge score pinned")

    output = source.copy()
    output["dataset_version"] = "6.1.0"

    for case_id, spec in DECISIONS.items():
        mask = output["case_id"] == case_id
        output.loc[mask, "human_score"] = spec["new_score"]
        output.loc[mask, "human_reason"] = spec["new_reason"]

    # Prove that only the three approved rows changed in score/reason.
    comparison = source[["case_id", "human_score", "human_reason"]].merge(
        output[["case_id", "human_score", "human_reason"]],
        on="case_id",
        suffixes=("_before", "_after"),
        validate="one_to_one",
    )
    comparison["score_changed"] = (
        comparison["human_score_before"].astype(int)
        != comparison["human_score_after"].astype(int)
    )
    comparison["reason_changed"] = (
        comparison["human_reason_before"].astype(str)
        != comparison["human_reason_after"].astype(str)
    )
    changed = set(
        comparison.loc[
            comparison["score_changed"] | comparison["reason_changed"],
            "case_id",
        ].astype(str)
    )
    require(changed == EXPECTED_CHANGED_IDS,
            "only the three approved adjudication rows changed score/reason")

    # Prove that all non-label fields remain unchanged for all 240 rows.
    # The prior r1 check incorrectly compared the approved edited label cells
    # against their pre-adjudication values, which necessarily failed.
    protected_columns = [
        c for c in source.columns
        if c not in {"human_score", "human_reason", "dataset_version"}
    ]
    require(
        source[protected_columns].fillna("").astype(str).equals(
            output[protected_columns].fillna("").astype(str)
        ),
        "all non-label fields remain unchanged across all 240 rows"
    )

    # For the other 237 cases, even the human label/reason must be untouched.
    unaffected_mask = ~source["case_id"].isin(EXPECTED_CHANGED_IDS)
    unaffected_columns = ["case_id", "human_score", "human_reason"]
    require(
        source.loc[unaffected_mask, unaffected_columns]
        .reset_index(drop=True)
        .fillna("")
        .astype(str)
        .equals(
            output.loc[unaffected_mask, unaffected_columns]
            .reset_index(drop=True)
            .fillna("")
            .astype(str)
        ),
        "human score/reason unchanged for all 237 non-adjudicated rows"
    )

    require(
        output["dataset_version"].astype(str).eq("6.1.0").all(),
        "all output rows use dataset_version 6.1.0"
    )

    # Confirm the exact approved transitions.
    expected_transitions = {
        "U4_Q03_T01": (0, 2),
        "U4_Q06_T01": (2, 0),
        "U4_Q09_T04": (1, 2),
    }
    for case_id, (before, after) in expected_transitions.items():
        require(
            int(source.loc[source["case_id"] == case_id, "human_score"].iloc[0]) == before
            and int(output.loc[output["case_id"] == case_id, "human_score"].iloc[0]) == after,
            f"{case_id} approved score transition {before} -> {after}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8")

    # Produce an adjudicated five-case review sheet carrying both the frozen r8
    # score and the human decision. This is the bridge to semantic repair design.
    adjudicated_review = review.copy()
    adjudicated_review["pre_adjudication_human_score"] = adjudicated_review["human_score"].astype(int)
    adjudicated_review["pre_adjudication_human_reason"] = adjudicated_review["human_reason"].astype(str)
    adjudicated_review["adjudicated_human_score"] = adjudicated_review["case_id"].map(
        {cid: spec["new_score"] for cid, spec in DECISIONS.items()}
    )
    adjudicated_review["adjudicated_human_reason"] = adjudicated_review["case_id"].map(
        {cid: spec["new_reason"] for cid, spec in DECISIONS.items()}
    )
    adjudicated_review["adjudication_classification"] = adjudicated_review["case_id"].map(
        {cid: spec["classification"] for cid, spec in DECISIONS.items()}
    )
    adjudicated_review["semantic_repair_target"] = adjudicated_review["case_id"].map(
        {cid: spec["semantic_repair_target"] for cid, spec in DECISIONS.items()}
    )
    adjudicated_review["adjudication_rationale"] = adjudicated_review["case_id"].map(
        {cid: spec["rationale"] for cid, spec in DECISIONS.items()}
    )
    adjudicated_review["post_adjudication_signed_difference"] = (
        adjudicated_review["judge_score"].astype(int)
        - adjudicated_review["adjudicated_human_score"].astype(int)
    )
    adjudicated_review["post_adjudication_absolute_difference"] = (
        adjudicated_review["post_adjudication_signed_difference"].abs()
    )
    adjudicated_review.sort_values("case_id").to_csv(
        REVIEW_OUTPUT, index=False, encoding="utf-8"
    )

    adjudication_manifest = {
        "schema_version": "1.0.0",
        "lineage": "semantic_relevance_v0.29.0",
        "source_dataset": str(SOURCE.relative_to(R)),
        "source_dataset_sha256": sha(SOURCE),
        "output_dataset": str(OUTPUT.relative_to(R)),
        "output_dataset_sha256": sha(OUTPUT),
        "source_review": str(SOURCE_REVIEW.relative_to(R)),
        "source_review_sha256": sha(SOURCE_REVIEW),
        "adjudicated_review": str(REVIEW_OUTPUT.relative_to(R)),
        "adjudicated_review_sha256": sha(REVIEW_OUTPUT),
        "judge_calls_made": False,
        "semantic_changes_made": False,
        "approved_label_revisions": [
            {
                "case_id": cid,
                "old_score": spec["old_score"],
                "new_score": spec["new_score"],
                "old_reason": spec["old_reason"],
                "new_reason": spec["new_reason"],
                "classification": spec["classification"],
                "semantic_repair_target": spec["semantic_repair_target"],
                "rationale": spec["rationale"],
            }
            for cid, spec in DECISIONS.items()
            if spec["old_score"] != spec["new_score"]
        ],
        "reviewed_without_label_change": [
            {
                "case_id": cid,
                "score": spec["new_score"],
                "reason": spec["new_reason"],
                "classification": spec["classification"],
                "semantic_repair_target": spec["semantic_repair_target"],
                "rationale": spec["rationale"],
            }
            for cid, spec in DECISIONS.items()
            if spec["old_score"] == spec["new_score"]
        ],
        "semantic_repair_targets": [
            cid for cid, spec in DECISIONS.items()
            if spec["semantic_repair_target"]
        ],
        "methodology_note": (
            "These are consumed v0.28 final-holdout cases now used only as v0.29 "
            "development evidence. The three revisions are human adjudications, not "
            "judge tuning. No v0.29 semantic change is introduced by this step."
        ),
    }
    ADJUDICATIONS.write_text(
        json.dumps(adjudication_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("v0.29.0 ADJUDICATION: PASS")
    print("Output dataset:", OUTPUT)
    print("Output SHA-256:", sha(OUTPUT))
    print("Adjudications:", ADJUDICATIONS)
    print("Adjudicated review:", REVIEW_OUTPUT)
    print("Approved score revisions:")
    print("  U4_Q03_T01: 0 -> 2")
    print("  U4_Q06_T01: 2 -> 0")
    print("  U4_Q09_T04: 1 -> 2")
    print("Semantic repair targets retained:")
    print("  U4_Q03_T02")
    print("  U4_Q04_T02")
    print("  U4_Q09_T04")
    print("NO JUDGE CALLS WERE MADE.")
    print("NO SEMANTIC CHANGES WERE MADE.")


if __name__ == "__main__":
    main()
