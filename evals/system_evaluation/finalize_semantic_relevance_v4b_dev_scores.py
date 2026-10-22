from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

SCORE_INVENTORY = RUN / "v4b_dev_display_score_inventory.csv"
NOVEL_INPUT = RUN / "v4b_dev_novel_display_pairs_to_judge.csv"
JUDGING_INPUT_LOCK = RUN / "v4b_dev_judging_input.lock.json"
NOVEL_SCORES = RUN / "v4b_dev_novel_judge_scores.csv"
NOVEL_SCORING_LOCK = RUN / "v4b_dev_novel_judge_scoring.lock.json"

COMPLETE_MAP = RUN / "v4b_dev_complete_display_score_map.csv"
FINALIZATION_METADATA = RUN / "v4b_dev_score_finalization_metadata.json"
FINALIZATION_LOCK = RUN / "v4b_dev_complete_score_map.lock.json"

EXPECTED = {
    "score_inventory": "f2c3d8905e582cc9df6b3a9417f3647d71496eed45ea5240d95c1cf1ece85cc1",
    "novel_input": "5eeb07e860ba70248a3c8f6df238f347535d7fe0a6d3c76314815b493ef753d0",
    "judging_input_lock": "9099bae135a7a97459633243a0659543b631fcc9bb3987aee0a86b6ee5fedf20",
    "novel_scores": "42dcbe7f8fb39427775c9a5a4d9251c677dc7de844aecaf91f1738a2e79a68c8",
    "novel_scoring_lock": "87555c2703582f0d8fd9c3ccdee1b7a3e8e6df850ceea48ee85275e0e888194f"
}
EXPECTED_TOTAL = 367
EXPECTED_REUSED = 345
EXPECTED_NOVEL = 22

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def bool_value(value: Any) -> bool:
    t = str(value).strip().lower()
    if t in {"true","1","yes"}:
        return True
    if t in {"false","0","no",""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")

def verify_inputs():
    for label, path, expected_sha in [
        ("score inventory", SCORE_INVENTORY, EXPECTED["score_inventory"]),
        ("novel judging input", NOVEL_INPUT, EXPECTED["novel_input"]),
        ("judging-input lock", JUDGING_INPUT_LOCK, EXPECTED["judging_input_lock"]),
        ("novel judge scores", NOVEL_SCORES, EXPECTED["novel_scores"]),
        ("novel scoring lock", NOVEL_SCORING_LOCK, EXPECTED["novel_scoring_lock"]),
    ]:
        req(path.exists(), f"Missing frozen input: {path}")
        actual = sha256(path)
        req(actual == expected_sha, f"{label} hash mismatch: expected={expected_sha} actual={actual}")
        print(f"PASS  frozen {label} SHA-256 verified")

    ilock = load(JUDGING_INPUT_LOCK)
    req(ilock.get("status") == "V4B_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "Unexpected judging-input status")
    req(int(ilock.get("display_union_rows",-1)) == EXPECTED_TOTAL, "Display-union total changed")
    req(int(ilock.get("reused_frozen_scores",-1)) == EXPECTED_REUSED, "Reused count changed")
    req(int(ilock.get("novel_pairs_to_judge",-1)) == EXPECTED_NOVEL, "Novel count changed")

    slock = load(NOVEL_SCORING_LOCK)
    req(slock.get("status") == "V4B_DEV_NOVEL_JUDGE_SCORES_FROZEN_BEFORE_FINALIZATION",
        "Unexpected novel scoring status")
    req(int(slock.get("novel_pairs",-1)) == EXPECTED_NOVEL, "Novel scoring count changed")
    req(slock.get("novel_judge_scores_sha256") == EXPECTED["novel_scores"],
        "Novel score hash differs inside scoring lock")
    req(slock.get("human_relevance_labels_used") is False,
        "Novel scoring lock says human labels were used")
    req(slock.get("release_metrics_calculated") is False,
        "Novel scoring stage calculated release metrics")
    req(slock.get("development_selection_decision_produced") is False,
        "Novel scoring stage produced a development decision")

    inventory = pd.read_csv(SCORE_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(inventory) == EXPECTED_TOTAL, "Inventory row count changed")
    req(not inventory.duplicated(["query_id","isbn13"]).any(), "Duplicate inventory pair")
    req(int((inventory["score_status"]=="reused").sum()) == EXPECTED_REUSED, "Reused inventory count changed")
    req(int((inventory["score_status"]=="novel").sum()) == EXPECTED_NOVEL, "Novel inventory count changed")
    req(int(inventory["in_v3b"].map(bool_value).sum()) == 240, "V3B membership is not 240")
    req(int(inventory["in_v4b"].map(bool_value).sum()) == 240, "V4B membership is not 240")
    req(inventory.loc[inventory["in_v3b"].map(bool_value), "score_status"].eq("reused").all(),
        "A V3B pair lacks a reused frozen score")
    req(inventory.loc[inventory["score_status"].eq("novel"), "pair_source"].eq("v4b_only").all(),
        "Novel inventory includes non-V4B-only pair")

    reused_sources = inventory.loc[inventory["score_status"].eq("reused"), "score_source"].value_counts().to_dict()
    req(reused_sources == {
        "v4a_complete_display_frozen": 296,
        "v4a_retrdiag_novel_frozen": 39,
        "v3_posthoc_deep_frozen": 10,
    }, f"Unexpected reused score sources: {reused_sources}")

    novel = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    scores = pd.read_csv(NOVEL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(novel) == EXPECTED_NOVEL and len(scores) == EXPECTED_NOVEL,
        "Novel input/score row counts changed")
    req(not novel["case_id"].duplicated().any(), "Duplicate novel input case IDs")
    req(not scores["case_id"].duplicated().any(), "Duplicate novel score case IDs")
    req(set(novel["case_id"]) == set(scores["case_id"]),
        "Novel judged case set differs from frozen input")

    expected_cases = novel.set_index("case_id", drop=False)
    actual_cases = scores.set_index("case_id", drop=False)
    for field in ["query_id","query","stratum","isbn13","title","pair_source","in_v3b","v3b_rank","in_v4b","v4b_rank"]:
        left = expected_cases.loc[actual_cases.index, field].astype(str)
        right = actual_cases[field].astype(str)
        req((left == right).all(), f"Novel judged payload differs from frozen input: {field}")

    req(not ((novel["query_id"]=="H14") & (novel["isbn13"]=="9781582972947")).any(),
        "Unresolved H14 quarantine pair entered finalization")
    vals = pd.to_numeric(scores["judge_score"], errors="raise").astype(int)
    req(vals.between(0,4).all(), "Novel score outside 0..4")

    print("PASS  frozen reuse/novel boundary verified: 345 reused + 22 judged = 367")
    print(f"INFO  reused score sources: {reused_sources}")
    print("PASS  unresolved H14 quarantine pair absent")
    return inventory, scores

def build_complete_map(inventory: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    payload = scores[[
        "query_id","isbn13","judge_score","match_level","judge_reason",
        "scoring_explanation","facet_assessments_json","facet_evidence_json",
        "component_evidence_ledger_json",
    ]].copy().rename(columns={
        "judge_score":"new_judge_score",
        "match_level":"new_match_level",
        "judge_reason":"new_judge_reason",
        "scoring_explanation":"new_scoring_explanation",
        "facet_assessments_json":"new_facet_assessments_json",
        "facet_evidence_json":"new_facet_evidence_json",
        "component_evidence_ledger_json":"new_component_evidence_ledger_json",
    })

    merged = inventory.merge(payload, on=["query_id","isbn13"], how="left", validate="one_to_one")
    reused = merged["score_status"].eq("reused")
    novel = merged["score_status"].eq("novel")
    req(merged.loc[reused, "new_judge_score"].isna().all(),
        "A reused pair unexpectedly has a new judge score")
    req(merged.loc[novel, "new_judge_score"].notna().all(),
        "A novel pair is missing its frozen new judgment")

    for col in [
        "final_score_source","final_judge_score","final_match_level","final_judge_reason",
        "final_scoring_explanation","final_facet_assessments_json","final_facet_evidence_json",
        "final_component_evidence_ledger_json",
    ]:
        merged[col] = pd.Series([None]*len(merged), dtype="object")

    for src,dst in [
        ("score_source","final_score_source"),("judge_score","final_judge_score"),
        ("match_level","final_match_level"),("judge_reason","final_judge_reason"),
        ("scoring_explanation","final_scoring_explanation"),
        ("facet_assessments_json","final_facet_assessments_json"),
        ("facet_evidence_json","final_facet_evidence_json"),
        ("component_evidence_ledger_json","final_component_evidence_ledger_json"),
    ]:
        merged.loc[reused,dst] = merged.loc[reused,src].astype("object")

    merged.loc[novel,"final_score_source"] = "v4b_dev_novel_judged"
    for src,dst in [
        ("new_judge_score","final_judge_score"),("new_match_level","final_match_level"),
        ("new_judge_reason","final_judge_reason"),
        ("new_scoring_explanation","final_scoring_explanation"),
        ("new_facet_assessments_json","final_facet_assessments_json"),
        ("new_facet_evidence_json","final_facet_evidence_json"),
        ("new_component_evidence_ledger_json","final_component_evidence_ledger_json"),
    ]:
        merged.loc[novel,dst] = merged.loc[novel,src].astype("object")

    merged["final_judge_score"] = pd.to_numeric(
        merged["final_judge_score"], errors="raise"
    ).astype(int)
    req(merged["final_judge_score"].between(0,4).all(),
        "Complete-map judge score outside 0..4")

    source_counts = merged["final_score_source"].value_counts().to_dict()
    expected_counts = {
        "v4a_complete_display_frozen": 296,
        "v4a_retrdiag_novel_frozen": 39,
        "v3_posthoc_deep_frozen": 10,
        "v4b_dev_novel_judged": 22,
    }
    req(source_counts == expected_counts, f"Unexpected final score sources: {source_counts}")

    cols = [
        "query_id","query","stratum","isbn13","title","description","authors",
        "pair_source","in_v3b","v3b_rank","in_v4b","v4b_rank",
        "final_score_source","final_judge_score","final_match_level","final_judge_reason",
        "final_scoring_explanation","final_facet_assessments_json","final_facet_evidence_json",
        "final_component_evidence_ledger_json",
    ]
    complete = merged[cols].copy()
    complete["_v3"] = complete["v3b_rank"].replace("", "9999").astype(int)
    complete["_v4"] = complete["v4b_rank"].replace("", "9999").astype(int)
    complete = complete.sort_values(
        ["query_id","_v3","_v4","isbn13"], kind="stable"
    ).drop(columns=["_v3","_v4"]).reset_index(drop=True)

    req(len(complete) == EXPECTED_TOTAL, "Complete score map row count changed")
    req(not complete.duplicated(["query_id","isbn13"]).any(), "Duplicate pair in complete score map")
    req(complete["final_judge_score"].notna().all(), "Complete score map has an unscored row")
    req(not ((complete["query_id"]=="H14") & (complete["isbn13"]=="9781582972947")).any(),
        "Unresolved H14 quarantine pair entered complete score map")
    return complete

def main() -> None:
    print("Semantic relevance V4B - deterministic development score finalization")
    print("-------------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Development comparison: NOT CALCULATED")
    print("Development selection: NOT PRODUCED")
    print("Release metrics/gates: NOT CALCULATED")
    print()

    for out in [COMPLETE_MAP, FINALIZATION_METADATA, FINALIZATION_LOCK]:
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite existing finalization artifact: {out}")

    inventory, scores = verify_inputs()
    complete = build_complete_map(inventory, scores)
    complete.to_csv(COMPLETE_MAP, index=False, encoding="utf-8")
    counts = {str(k):int(v) for k,v in complete["final_score_source"].value_counts().to_dict().items()}

    metadata = {
        "status":"V4B_DEV_SCORE_FINALIZATION_COMPLETE",
        "completed_at_utc":datetime.now(timezone.utc).isoformat(),
        "development_query_set":"H01-H24",
        "score_inventory_sha256":EXPECTED["score_inventory"],
        "novel_judging_input_sha256":EXPECTED["novel_input"],
        "judging_input_lock_sha256":EXPECTED["judging_input_lock"],
        "novel_judge_scores_sha256":EXPECTED["novel_scores"],
        "novel_judge_scoring_lock_sha256":EXPECTED["novel_scoring_lock"],
        "displayed_pair_count":EXPECTED_TOTAL,
        "reused_frozen_scores":EXPECTED_REUSED,
        "new_novel_scores":EXPECTED_NOVEL,
        "score_source_counts":counts,
        "known_h14_quarantine_pair_present":False,
        "complete_score_map_sha256":sha256(COMPLETE_MAP),
        "judge_calls_performed_during_finalization":0,
        "model_calls_performed_during_finalization":0,
        "human_relevance_labels_used":False,
        "development_comparison_calculated":False,
        "development_selection_decision_produced":False,
        "release_metrics_calculated":False,
        "release_gates_calculated":False,
        "next_stage":"V3B_VS_V4B_DEVELOPMENT_COMPARISON_AND_CANDIDATE_SELECTION",
    }
    FINALIZATION_METADATA.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")

    lock = dict(metadata)
    lock["status"] = "V4B_DEV_COMPLETE_DISPLAY_SCORE_MAP_FROZEN_BEFORE_COMPARISON"
    lock["frozen_at_utc"] = lock.pop("completed_at_utc")
    lock["finalization_metadata_sha256"] = sha256(FINALIZATION_METADATA)
    FINALIZATION_LOCK.write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")

    print("PASS  complete displayed-pair score map finalized: 367/367")
    print(f"INFO  score sources: {counts}")
    print("PASS  unresolved H14 quarantine pair absent")
    print("PASS  complete score map frozen before development comparison")
    print()
    print("V4B DEVELOPMENT SCORE FINALIZATION: PASS")
    print(f"Complete score map SHA-256: {sha256(COMPLETE_MAP)}")
    print(f"Finalization lock SHA-256: {sha256(FINALIZATION_LOCK)}")
    print("Judge calls during finalization: 0")
    print("Model calls during finalization: 0")
    print("Human relevance labels used: NO")
    print("Development comparison calculated: NO")
    print("Development selection produced: NO")
    print("Release metrics calculated: NO")
    print("Next: V3B-vs-V4B development comparison and candidate selection.")

if __name__ == "__main__":
    main()
