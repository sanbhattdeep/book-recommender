from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

V4A_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
V3_HOLDOUT_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
V3_POSTHOC_RUN = REPO_ROOT / "evals/runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"

DISPLAY_UNION = V4A_RUN / "v3b_v4a_dev_display_union.csv"
DISPLAY_UNION_LOCK = V4A_RUN / "v3b_v4a_dev_display_union.lock.json"
HOLDOUT_SCORES = V3_HOLDOUT_RUN / "holdout_judge_scores.csv"
HOLDOUT_SCORING_LOCK = V3_HOLDOUT_RUN / "holdout_judge_scoring_lock.json"
POSTHOC_COMPLETE_MAP = V3_POSTHOC_RUN / "target_top50_complete_score_map.csv"
POSTHOC_SCORING_LOCK = V3_POSTHOC_RUN / "deep_candidate_scoring_lock.json"

EXPECTED = {
    "display_union": "caa34848a25b4f6d245a398a8860325f9d2ed3898951b665fa387ef6f4c7b85a",
    "display_union_lock": "607fe469087ce78dd04dc4574c7f0f889f2f04939b6f73abac003243a9be6028",
    "holdout_scores": "d1c9ba1848fb4875c1e925b6c5bdafc3483db37a613bad49286a634d96803fa3",
    "holdout_scoring_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "posthoc_complete_map": "723fa27ecb8947334d25b3b7423a6597da39d594eb7ec2f979056b1c3cc516e5",
    "posthoc_scoring_lock": "af67cc8090d0613973da4c927b573d29bb956be34d4c44255594cb12bde6fd1f",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bool_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def verify_frozen_inputs() -> None:
    checks = [
        ("v4a displayed union", DISPLAY_UNION, EXPECTED["display_union"]),
        ("v4a displayed-union lock", DISPLAY_UNION_LOCK, EXPECTED["display_union_lock"]),
        ("v3 holdout judge scores", HOLDOUT_SCORES, EXPECTED["holdout_scores"]),
        ("v3 holdout scoring lock", HOLDOUT_SCORING_LOCK, EXPECTED["holdout_scoring_lock"]),
        ("v3 posthoc complete score map", POSTHOC_COMPLETE_MAP, EXPECTED["posthoc_complete_map"]),
        ("v3 posthoc scoring lock", POSTHOC_SCORING_LOCK, EXPECTED["posthoc_scoring_lock"]),
    ]
    for label, path, expected in checks:
        require(path.exists(), f"Missing frozen input: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    display_lock = load_json(DISPLAY_UNION_LOCK)
    require(display_lock.get("status") == "V4A_DEV_DISPLAY_UNION_FROZEN_BEFORE_JUDGING", "Unexpected v4a display-union lock status")
    require(display_lock.get("judge_execution_performed") is False, "Display-union lock says judge execution already occurred")
    require(int(display_lock.get("union_rows", -1)) == 347, "Display-union lock row count changed")

    holdout_lock = load_json(HOLDOUT_SCORING_LOCK)
    require(holdout_lock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS", "Unexpected v3 holdout scoring-lock status")

    posthoc_lock = load_json(POSTHOC_SCORING_LOCK)
    require(posthoc_lock.get("status") == "V3_POSTHOC_DEEP_JUDGE_SCORES_FROZEN_BEFORE_CLASSIFICATION", "Unexpected v3 posthoc scoring-lock status")
    require(posthoc_lock.get("human_relevance_labels_used") is False, "Posthoc score source used human labels")


def read_display_union() -> pd.DataFrame:
    frame = pd.read_csv(DISPLAY_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "pair_source", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank",
    }
    missing = required - set(frame.columns)
    require(not missing, f"Display union missing columns: {sorted(missing)}")
    require(len(frame) == 347, f"Expected 347 display-union rows, got {len(frame)}")
    require(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate query-book pair in display union")
    require(frame["query_id"].nunique() == 24, "Display union is not H01-H24")
    return frame


def normalize_holdout_scores() -> pd.DataFrame:
    frame = pd.read_csv(HOLDOUT_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "isbn13", "judge_score", "match_level", "judge_reason", "scoring_explanation",
        "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json",
    }
    missing = required - set(frame.columns)
    require(not missing, f"Holdout score artifact missing columns: {sorted(missing)}")
    require(len(frame) == 322, f"Expected 322 v3 holdout scores, got {len(frame)}")
    require(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in holdout scores")
    out = frame[list(required)].copy()
    out["judge_score"] = pd.to_numeric(out["judge_score"], errors="raise").astype(int)
    require(out["judge_score"].between(0, 4).all(), "Holdout judge score outside 0..4")
    out["score_source"] = "v3_holdout_frozen"
    return out


def normalize_posthoc_scores() -> pd.DataFrame:
    frame = pd.read_csv(POSTHOC_COMPLETE_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "isbn13", "judge_score", "final_match_level", "final_judge_reason",
        "final_scoring_explanation", "final_facet_assessments_json", "final_facet_evidence_json",
        "final_component_evidence_ledger_json",
    }
    missing = required - set(frame.columns)
    require(not missing, f"Posthoc complete score map missing columns: {sorted(missing)}")
    require(len(frame) == 850, f"Expected 850 v3 posthoc complete-map rows, got {len(frame)}")
    require(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in posthoc complete score map")
    out = frame[list(required)].copy().rename(columns={
        "final_match_level": "match_level",
        "final_judge_reason": "judge_reason",
        "final_scoring_explanation": "scoring_explanation",
        "final_facet_assessments_json": "facet_assessments_json",
        "final_facet_evidence_json": "facet_evidence_json",
        "final_component_evidence_ledger_json": "component_evidence_ledger_json",
    })
    out["judge_score"] = pd.to_numeric(out["judge_score"], errors="raise").astype(int)
    require(out["judge_score"].between(0, 4).all(), "Posthoc judge score outside 0..4")
    out["score_source"] = "v3_posthoc_deep_frozen"
    return out


def build_frozen_score_map() -> tuple[pd.DataFrame, int]:
    holdout = normalize_holdout_scores()
    posthoc = normalize_posthoc_scores()
    combined = pd.concat([holdout, posthoc], ignore_index=True)

    payload_cols = [
        "judge_score", "match_level", "judge_reason", "scoring_explanation",
        "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json",
    ]
    conflict_count = 0
    rows: list[dict[str, Any]] = []
    for (qid, isbn), group in combined.groupby(["query_id", "isbn13"], sort=False):
        if len(group) > 1:
            first = group.iloc[0]
            for _, other in group.iloc[1:].iterrows():
                if any(str(first[c]) != str(other[c]) for c in payload_cols):
                    conflict_count += 1
                    raise ValueError(f"Frozen score conflict for {qid}/{isbn}")
        # Prefer original unseen-holdout score when both sources carry the same pair.
        selected = group[group["score_source"] == "v3_holdout_frozen"]
        row = (selected.iloc[0] if not selected.empty else group.iloc[0]).to_dict()
        rows.append(row)

    out = pd.DataFrame(rows)
    require(not out.duplicated(["query_id", "isbn13"]).any(), "Deduplicated score map still contains duplicate pair")
    return out, conflict_count


def main() -> None:
    print("Semantic relevance v4a - prepare frozen score reuse / novel judging input")
    print("-----------------------------------------------------------------------")
    print("New judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics: NOT CALCULATED")
    print()

    verify_frozen_inputs()
    display = read_display_union()
    frozen_scores, conflicts = build_frozen_score_map()
    print(f"PASS  frozen prior-score sources normalized with conflicts={conflicts}")

    merged = display.merge(
        frozen_scores,
        on=["query_id", "isbn13"],
        how="left",
        validate="one_to_one",
    )
    merged["score_status"] = merged["judge_score"].apply(lambda x: "reused" if pd.notna(x) else "novel")

    reused_mask = merged["score_status"] == "reused"
    novel_mask = ~reused_mask
    reused = int(reused_mask.sum())
    novel = int(novel_mask.sum())
    require(reused + novel == 347, "Reuse/novel counts do not cover display union")

    in_v3b = merged["in_v3b"].map(bool_value)
    require(merged.loc[in_v3b, "score_status"].eq("reused").all(), "A frozen v3b displayed pair lacks a reusable judge score")
    require(merged.loc[novel_mask, "pair_source"].eq("v4a_only").all(), "Novel judging set contains a pair that is not v4a-only")

    # Canonical integer scores for reused rows; blank for novel rows.
    merged["judge_score"] = merged["judge_score"].astype("object")
    merged.loc[reused_mask, "judge_score"] = pd.to_numeric(merged.loc[reused_mask, "judge_score"], errors="raise").astype(int)
    merged.loc[novel_mask, "judge_score"] = ""
    merged.loc[novel_mask, "score_source"] = ""
    for col in ["match_level", "judge_reason", "scoring_explanation", "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json"]:
        merged.loc[novel_mask, col] = ""

    inventory_cols = [
        "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "pair_source", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank",
        "score_status", "score_source", "judge_score", "match_level", "judge_reason",
        "scoring_explanation", "facet_assessments_json", "facet_evidence_json",
        "component_evidence_ledger_json",
    ]
    inventory = merged[inventory_cols].sort_values(["query_id", "isbn13"], kind="stable").reset_index(drop=True)

    novel_pairs = inventory[inventory["score_status"] == "novel"].copy()
    novel_pairs.insert(0, "case_id", novel_pairs.apply(lambda r: f"V4A_DEV_{r['query_id']}_{r['isbn13']}", axis=1))
    novel_cols = [
        "case_id", "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "pair_source", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank",
    ]
    novel_pairs = novel_pairs[novel_cols].sort_values(["query_id", "v4a_rank", "isbn13"], kind="stable").reset_index(drop=True)
    require(not novel_pairs["case_id"].duplicated().any(), "Novel case IDs are not unique")

    inventory_file = V4A_RUN / "v4a_dev_display_score_inventory.csv"
    novel_file = V4A_RUN / "v4a_dev_novel_display_pairs_to_judge.csv"
    metadata_file = V4A_RUN / "v4a_dev_score_reuse_metadata.json"
    lock_file = V4A_RUN / "v4a_dev_judging_input.lock.json"
    for path in [inventory_file, novel_file, metadata_file, lock_file]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing score-reuse artifact: {path}")

    inventory.to_csv(inventory_file, index=False, encoding="utf-8")
    novel_pairs.to_csv(novel_file, index=False, encoding="utf-8")

    source_counts = inventory.loc[reused_mask, "score_source"].value_counts().to_dict()
    pair_source_novel_counts = inventory.loc[novel_mask, "pair_source"].value_counts().to_dict()

    metadata = {
        "status": "V4A_DEV_SCORE_REUSE_PREPARATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "display_union_sha256": EXPECTED["display_union"],
        "display_union_lock_sha256": EXPECTED["display_union_lock"],
        "prior_frozen_score_sources": {
            "v3_holdout_judge_scores_sha256": EXPECTED["holdout_scores"],
            "v3_holdout_scoring_lock_sha256": EXPECTED["holdout_scoring_lock"],
            "v3_posthoc_complete_score_map_sha256": EXPECTED["posthoc_complete_map"],
            "v3_posthoc_scoring_lock_sha256": EXPECTED["posthoc_scoring_lock"],
        },
        "display_union_rows": 347,
        "reused_frozen_scores": reused,
        "novel_pairs_to_judge": novel,
        "reused_score_source_counts": {k: int(v) for k, v in source_counts.items()},
        "novel_pair_source_counts": {k: int(v) for k, v in pair_source_novel_counts.items()},
        "frozen_score_conflicts": conflicts,
        "new_judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "outputs": {
            inventory_file.name: sha256(inventory_file),
            novel_file.name: sha256(novel_file),
        },
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4A_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "display_union_sha256": EXPECTED["display_union"],
        "display_union_lock_sha256": EXPECTED["display_union_lock"],
        "display_union_rows": 347,
        "reused_frozen_scores": reused,
        "novel_pairs_to_judge": novel,
        "score_inventory_sha256": sha256(inventory_file),
        "novel_judging_input_sha256": sha256(novel_file),
        "score_reuse_metadata_sha256": sha256(metadata_file),
        "prior_score_sources": {
            "v3_holdout_judge_scores_sha256": EXPECTED["holdout_scores"],
            "v3_holdout_scoring_lock_sha256": EXPECTED["holdout_scoring_lock"],
            "v3_posthoc_complete_score_map_sha256": EXPECTED["posthoc_complete_map"],
            "v3_posthoc_scoring_lock_sha256": EXPECTED["posthoc_scoring_lock"],
        },
        "frozen_score_conflicts": conflicts,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "next_stage": "prepare_v4a_development_novel_pair_judge_execution_package",
    }
    lock_file.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  display score inventory frozen: {len(inventory)} rows")
    print(f"INFO  reused frozen scores: {reused}")
    print(f"INFO  novel displayed pairs to judge: {novel}")
    print(f"INFO  reused sources: {source_counts}")
    print(f"INFO  novel pair sources: {pair_source_novel_counts}")
    print("PASS  novel judging input frozen before any new judge execution")
    print()
    print("V4A DEVELOPMENT SCORE REUSE PREPARATION: PASS")
    print(f"Score inventory SHA-256: {sha256(inventory_file)}")
    print(f"Novel judging input SHA-256: {sha256(novel_file)}")
    print(f"Judging-input lock SHA-256: {sha256(lock_file)}")
    print("New judge calls performed: 0")
    print("Human relevance labels used: NO")
    print("Release metrics calculated: NO")
    print("Next: prepare judge execution package for only the frozen novel displayed pairs.")


if __name__ == "__main__":
    main()
