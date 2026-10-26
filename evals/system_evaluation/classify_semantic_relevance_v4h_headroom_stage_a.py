from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
SYS = ROOT / "evals/system_evaluation"

CANDIDATES = RUN / "v4h_no_clear_headroom_stage_a_candidates.csv"
CANDIDATE_LOCK = RUN / "v4h_no_clear_headroom_stage_a.lock.json"
SCORE_INVENTORY = RUN / "v4h_headroom_stage_a_score_inventory.csv"
JUDGING_INPUT = RUN / "v4h_headroom_stage_a_judging_input.csv"
JUDGING_INPUT_LOCK = RUN / "v4h_headroom_stage_a_judging_input.lock.json"
EXISTING_SCORES = RUN / "v4h_judge_scores.csv"
EXISTING_SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"
FRESH_SCORES = RUN / "v4h_headroom_stage_a_fresh_judge_scores.csv"
FRESH_SCORING_LOCK = RUN / "v4h_headroom_stage_a_fresh_judge_scoring.lock.json"
FAILURE_LOCK = RUN / "v4h_failure_analysis.lock.json"
FINAL_RELEASE_LOCK = RUN / "v4h_final_release_analysis.lock.json"
STAGE_A_CONTRACT = SYS / "semantic_relevance_v4h_no_clear_headroom_stage_a.v0.1.0.json"
JUDGING_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_a_judging_contract.v0.1.0.json"
EXECUTION_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_a_execution_contract.v0.1.0.json"
CLASSIFICATION_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_a_classification_contract.v0.1.0.json"

COMPLETE_SCORES = RUN / "v4h_headroom_stage_a_complete_scores.csv"
QUERY_CLASSIFICATION = RUN / "v4h_headroom_stage_a_query_classification.csv"
SUMMARY = RUN / "v4h_headroom_stage_a_summary.json"
REPORT = RUN / "v4h_headroom_stage_a_report.md"
METADATA = RUN / "v4h_headroom_stage_a_classification_metadata.json"
CLASSIFICATION_LOCK = RUN / "v4h_headroom_stage_a_classification.lock.json"

TARGET_QUERY_IDS = ["V4H07","V4H09","V4H10","V4H11","V4H13","V4H16","V4H19","V4H21","V4H22","V4H23"]
EXPECTED = {
    "candidates": "2f2bfe05e1c21bb6a7b8c6dbe6c37fca8c42b9985ebeb74fcda710f5b629a68d",
    "candidate_lock": "cb0ee919fd4a830d6d7e30f7acc91854470354684768ab9cf4762ec289f1fdfa",
    "score_inventory": "fbbc796bc885842e5f9e63ce9ea7d2c7540bdeea09aa34b51c44a80988807cda",
    "judging_input": "eb57cfead0f79cedb9f350b3be683a5b063ab62355cc9074e7df264fc4fe14c8",
    "judging_input_lock": "67d1dfbc2852adf96c97de0a34603010f06bab37aa77854b21c6dd5325653173",
    "existing_scores": "8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906",
    "existing_scoring_lock": "ac7a96c89eb2d4a51eda1b56756ed3b943311e7f300b661659cff51285423a8f",
    "fresh_scores": "5a1c988e0efb6898e721f700261e834ed1994de107338dc49e68549aa9e1ca02",
    "fresh_scoring_lock": "b64ff7d8828fcd4d8938c487d48335c0837aaca0d6439f42b91579467a5c57f9",
    "failure_lock": "4e0f10f3d70fae6154c20d32622c21f437749d30835e892736cafa7d6ab3fc90",
    "final_release_lock": "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0",
    "stage_a_contract": "e47636abed2ac834c69ad9ce17c8c32fa4803b9f0cbca86dfd44f8a83ab7639d",
    "judging_contract": "75810d2451809943a1e5d586029aa27ab42f87113a028273d5ee582496cf9422",
    "execution_contract": "5385d221e2c80968dd976b4bf18abdbda6bbc3c8b30db264167f572c78b47dbb",
    "classification_contract": "16228e03a1148d298aa0e2529265d3ddeee021062882b6c043e6cf9bdd33a5a9",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    req(bool(text), "ISBN is empty")
    return text

def verify_boundary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks = [
        (CANDIDATES, "candidates", "Stage-A candidate inventory"),
        (CANDIDATE_LOCK, "candidate_lock", "Stage-A candidate lock"),
        (SCORE_INVENTORY, "score_inventory", "Stage-A score inventory"),
        (JUDGING_INPUT, "judging_input", "Stage-A fresh judging input"),
        (JUDGING_INPUT_LOCK, "judging_input_lock", "Stage-A judging-input lock"),
        (EXISTING_SCORES, "existing_scores", "frozen V4H judge scores"),
        (EXISTING_SCORING_LOCK, "existing_scoring_lock", "frozen V4H judge scoring lock"),
        (FRESH_SCORES, "fresh_scores", "Stage-A fresh judge scores"),
        (FRESH_SCORING_LOCK, "fresh_scoring_lock", "Stage-A fresh scoring lock"),
        (FAILURE_LOCK, "failure_lock", "V4H failure-analysis lock"),
        (FINAL_RELEASE_LOCK, "final_release_lock", "V4H final release-analysis lock"),
        (STAGE_A_CONTRACT, "stage_a_contract", "Stage-A definition contract"),
        (JUDGING_CONTRACT, "judging_contract", "Stage-A judging-preparation contract"),
        (EXECUTION_CONTRACT, "execution_contract", "Stage-A execution contract"),
        (CLASSIFICATION_CONTRACT, "classification_contract", "Stage-A classification contract"),
    ]
    for path, key, label in checks:
        verify_raw(path, EXPECTED[key], label)

    failure = load_json(FAILURE_LOCK)
    req(failure.get("status") == "V4H_POST_RELEASE_FAILURE_ANALYSIS_FROZEN", "Unexpected failure-analysis lock status")
    req(failure.get("frozen_release_decision") == "FAIL", "Frozen V4B release decision changed")
    req(failure.get("release_decision_modified") is False, "Failure analysis modified release decision")
    req(failure.get("v4h_consumed") is True, "V4H is not marked consumed")

    final = load_json(FINAL_RELEASE_LOCK)
    req(final.get("status") == "V4H_FINAL_RELEASE_ANALYSIS_FROZEN", "Unexpected final release-analysis lock status")
    req(final.get("v4b_final_release_decision") == "FAIL", "V4B final release decision changed")

    fresh_lock = load_json(FRESH_SCORING_LOCK)
    req(fresh_lock.get("status") == "V4H_HEADROOM_STAGE_A_FRESH_JUDGE_SCORES_FROZEN", "Unexpected Stage-A fresh-score lock status")
    req(int(fresh_lock.get("fresh_pair_count", -1)) == 340, "Fresh pair count changed")
    req(int(fresh_lock.get("reused_exact_pair_count_external", -1)) == 60, "Reused pair count changed")
    req(int(fresh_lock.get("total_stage_a_pair_count", -1)) == 400, "Stage-A pair count changed")
    req(fresh_lock.get("fresh_judge_scores_sha256") == EXPECTED["fresh_scores"], "Fresh-score lock does not bind expected score artifact")
    req(fresh_lock.get("human_relevance_labels_used") is False, "Fresh-score lock says human labels were used")
    req(fresh_lock.get("release_metrics_calculated") is False, "Fresh-score lock says release metrics were calculated")
    req(fresh_lock.get("release_decision_modified") is False, "Fresh-score lock says release decision was modified")

    inventory = pd.read_csv(SCORE_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(inventory) == 400, "Expected exactly 400 Stage-A inventory rows")
    required = {"case_id","query_id","stratum","v4b_rank","isbn13","score_source","reused_judge_score"}
    req(required <= set(inventory.columns), f"Stage-A score inventory missing columns: {sorted(required-set(inventory.columns))}")
    inventory["isbn13"] = inventory["isbn13"].map(canonical_isbn)
    inventory["v4b_rank"] = pd.to_numeric(inventory["v4b_rank"], errors="raise").astype(int)
    req(not inventory["case_id"].duplicated().any(), "Duplicate Stage-A case_id")
    req(not inventory.duplicated(["query_id","isbn13"]).any(), "Duplicate Stage-A query-book pair")
    req(sorted(inventory["query_id"].unique()) == TARGET_QUERY_IDS, "Stage-A target query set changed")
    req(inventory["v4b_rank"].between(11,50).all(), "Stage-A inventory contains rank outside 11-50")
    req((inventory.groupby("query_id").size() == 40).all(), "Expected 40 Stage-A candidates per query")
    counts = inventory["score_source"].value_counts().to_dict()
    req(counts == {"fresh_judge_required": 340, "reused_frozen_v4h_display_score": 60}, f"Stage-A score-source partition changed: {counts}")

    fresh = pd.read_csv(FRESH_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    fresh_required = {"case_id","query_id","isbn13","v4b_rank","judge_score","match_level"}
    req(fresh_required <= set(fresh.columns), f"Fresh score artifact missing columns: {sorted(fresh_required-set(fresh.columns))}")
    req(len(fresh) == 340, "Expected exactly 340 fresh Stage-A scores")
    fresh["isbn13"] = fresh["isbn13"].map(canonical_isbn)
    fresh["v4b_rank"] = pd.to_numeric(fresh["v4b_rank"], errors="raise").astype(int)
    fresh["judge_score"] = pd.to_numeric(fresh["judge_score"], errors="raise").astype(int)
    req(fresh["judge_score"].between(0,4).all(), "Fresh judge score outside 0..4")
    req(not fresh["case_id"].duplicated().any(), "Duplicate fresh case_id")
    req(not fresh.duplicated(["query_id","isbn13"]).any(), "Duplicate fresh query-book pair")

    reused = inventory[inventory["score_source"] == "reused_frozen_v4h_display_score"].copy()
    fresh_inventory = inventory[inventory["score_source"] == "fresh_judge_required"].copy()
    req(set(fresh["case_id"]) == set(fresh_inventory["case_id"]), "Fresh score case set does not equal frozen fresh inventory")
    for row in fresh.itertuples(index=False):
        src = fresh_inventory.set_index("case_id").loc[str(row.case_id)]
        req(str(row.query_id) == str(src["query_id"]), f"{row.case_id} query mismatch")
        req(canonical_isbn(row.isbn13) == canonical_isbn(src["isbn13"]), f"{row.case_id} ISBN mismatch")
        req(int(row.v4b_rank) == int(src["v4b_rank"]), f"{row.case_id} rank mismatch")

    reused_scores = pd.to_numeric(reused["reused_judge_score"], errors="raise").astype(int)
    req(reused_scores.between(0,4).all(), "Reused Stage-A score outside 0..4")

    existing = pd.read_csv(EXISTING_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req({"query_id","isbn13","judge_score"} <= set(existing.columns), "Frozen V4H score artifact missing required columns")
    existing["isbn13"] = existing["isbn13"].map(canonical_isbn)
    existing["judge_score"] = pd.to_numeric(existing["judge_score"], errors="raise").astype(int)
    req(not existing.duplicated(["query_id","isbn13"]).any(), "Frozen V4H score artifact has duplicate pair")
    lookup = existing.set_index(["query_id","isbn13"])["judge_score"].to_dict()
    for row in reused.itertuples(index=False):
        key = (str(row.query_id), canonical_isbn(row.isbn13))
        req(key in lookup, f"Reused pair missing from frozen V4H scores: {key}")
        req(int(row.reused_judge_score) == int(lookup[key]), f"Reused score mismatch for {key}")

    return inventory, fresh, existing

from datetime import datetime, timezone

def label_from_score(score: int) -> str:
    return {0:"none",1:"incidental",2:"partial",3:"clear",4:"strong"}[int(score)]

def main() -> None:
    print("Semantic relevance V4H - headroom Stage-A deterministic classification")
    print("-----------------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Judge/model/retrieval/reranker calls: 0")
    print("Human relevance labels: NOT USED")
    print("Release metrics/gates: NOT CALCULATED")
    print("Frozen V4B release decision: UNCHANGED")
    print()

    inventory, fresh, _ = verify_boundary()
    for path in [COMPLETE_SCORES, QUERY_CLASSIFICATION, SUMMARY, REPORT, METADATA, CLASSIFICATION_LOCK]:
        req(not path.exists(), f"Refusing to overwrite existing Stage-A classification artifact: {path}")

    fresh_lookup = fresh.set_index("case_id")
    rows = []
    for _, row in inventory.sort_values(["query_id","v4b_rank"], kind="stable").iterrows():
        out = row.to_dict()
        if row["score_source"] == "fresh_judge_required":
            scored = fresh_lookup.loc[str(row["case_id"])]
            score = int(scored["judge_score"])
            match_level = str(scored["match_level"])
            score_provenance = "fresh_stage_a_frozen_judge_score"
        else:
            score = int(row["reused_judge_score"])
            match_level = label_from_score(score)
            score_provenance = "exact_reused_frozen_v4h_score"
        out["judge_score"] = score
        out["match_level"] = match_level
        out["score_provenance"] = score_provenance
        rows.append(out)

    complete = pd.DataFrame(rows)
    req(len(complete) == 400, "Combined Stage-A score map is not 400 rows")
    req(not complete["case_id"].duplicated().any(), "Combined Stage-A score map has duplicate case_id")
    complete["judge_score"] = pd.to_numeric(complete["judge_score"], errors="raise").astype(int)
    complete["v4b_rank"] = pd.to_numeric(complete["v4b_rank"], errors="raise").astype(int)
    req(complete["judge_score"].between(0,4).all(), "Combined Stage-A score outside 0..4")
    req((complete.groupby("query_id").size() == 40).all(), "Combined map does not have 40 rows/query")
    complete.to_csv(COMPLETE_SCORES, index=False, encoding="utf-8")

    qrows = []
    for qid in TARGET_QUERY_IDS:
        g = complete[complete["query_id"] == qid].sort_values("v4b_rank", kind="stable")
        req(len(g) == 40, f"{qid} expected 40 Stage-A rows")
        clear = g[g["judge_score"] >= 3]
        clear_ranks = [int(v) for v in clear["v4b_rank"].tolist()]
        classification = "HEADROOM_FOUND_11_50" if clear_ranks else "NO_CLEAR_THROUGH_50"
        qrows.append({
            "query_id": qid,
            "query": str(g.iloc[0]["query"]) if "query" in g.columns else "",
            "stratum": str(g.iloc[0]["stratum"]),
            "stage_a_candidate_count": len(g),
            "reused_exact_count": int((g["score_source"] == "reused_frozen_v4h_display_score").sum()),
            "fresh_score_count": int((g["score_source"] == "fresh_judge_required").sum()),
            "best_score_11_50": int(g["judge_score"].max()),
            "clear_or_strong_count_11_50": int((g["judge_score"] >= 3).sum()),
            "partial_count_11_50": int((g["judge_score"] == 2).sum()),
            "incidental_count_11_50": int((g["judge_score"] == 1).sum()),
            "irrelevant_count_11_50": int((g["judge_score"] == 0).sum()),
            "first_clear_or_strong_rank": "" if not clear_ranks else min(clear_ranks),
            "clear_or_strong_ranks_json": json.dumps(clear_ranks),
            "classification": classification,
        })

    q = pd.DataFrame(qrows)
    req(len(q) == 10, "Expected 10 Stage-A query classifications")
    q.to_csv(QUERY_CLASSIFICATION, index=False, encoding="utf-8")

    counts = q["classification"].value_counts().to_dict()
    found = int(counts.get("HEADROOM_FOUND_11_50", 0))
    unresolved = int(counts.get("NO_CLEAR_THROUGH_50", 0))
    req(found + unresolved == 10, "Stage-A classification count mismatch")
    unresolved_ids = q.loc[q["classification"] == "NO_CLEAR_THROUGH_50", "query_id"].tolist()
    found_ids = q.loc[q["classification"] == "HEADROOM_FOUND_11_50", "query_id"].tolist()

    summary = {
        "status": "COMPLETE",
        "role": "post_release_consumed_holdout_diagnostic",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "target_query_count": 10,
        "rank_window": [11,50],
        "combined_pair_count": 400,
        "reused_exact_pair_count": 60,
        "fresh_pair_count": 340,
        "clear_or_strong_threshold": "judge_score >= 3",
        "classification_counts": {
            "HEADROOM_FOUND_11_50": found,
            "NO_CLEAR_THROUGH_50": unresolved,
        },
        "headroom_found_query_ids": found_ids,
        "no_clear_through_50_query_ids": unresolved_ids,
        "interpretation": {
            "HEADROOM_FOUND_11_50": "Displayed-ranking headroom is proven for this query because a clear/strong candidate already exists below displayed rank 10 but within frozen V4B ranks 11-50.",
            "NO_CLEAR_THROUGH_50": "Still unresolved. This does not prove full-union candidate limitation; optional Stage B must inspect ranks 51-end before such a conclusion.",
        },
        "new_judge_calls": 0,
        "new_model_calls": 0,
        "new_retrieval_calls": 0,
        "new_reranker_calls": 0,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "next_stage": "OPTIONAL_STAGE_B_FOR_NO_CLEAR_THROUGH_50_ONLY" if unresolved else "STAGE_A_COMPLETE_ALL_TARGET_QUERIES_HAVE_RANKING_HEADROOM",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V4H Headroom Diagnostic — Stage A Classification",
        "",
        "**Consumed-holdout diagnostic only. V4B release decision remains frozen FAIL.**",
        "",
        "Stage A combines 60 exact same-query/same-ISBN frozen V4H scores with 340 newly frozen Stage-A scores. No judge, model, retrieval, or reranker calls occur in this classification stage.",
        "",
        "## Frozen rule",
        "",
        "- `HEADROOM_FOUND_11_50`: at least one candidate at V4B rank 11-50 has `judge_score >= 3`.",
        "- `NO_CLEAR_THROUGH_50`: no candidate at ranks 11-50 has `judge_score >= 3`; this remains unresolved and is **not** proof of full-union retrieval limitation.",
        "",
        "## Query classification",
        "",
        "| Query | Stratum | Best score 11-50 | Clear/strong count | First clear rank | Classification |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in q.itertuples(index=False):
        first = "—" if str(row.first_clear_or_strong_rank) == "" else str(row.first_clear_or_strong_rank)
        lines.append(f"| {row.query_id} | {row.stratum} | {row.best_score_11_50} | {row.clear_or_strong_count_11_50} | {first} | `{row.classification}` |")
    lines += [
        "",
        "## Summary",
        "",
        f"- `HEADROOM_FOUND_11_50`: **{found}/10**",
        f"- `NO_CLEAR_THROUGH_50`: **{unresolved}/10**",
        "",
        "## Interpretation boundary",
        "",
        "A positive Stage-A result proves displayed-ranking headroom for that query. A negative Stage-A result only establishes no clear/strong candidate through rank 50; it does not establish full-union retrieval limitation. Optional Stage B is restricted to the unresolved queries and ranks 51-end.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "status": "V4H_HEADROOM_STAGE_A_CLASSIFICATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "classification_contract_sha256": EXPECTED["classification_contract"],
        "complete_scores_sha256": sha256(COMPLETE_SCORES),
        "query_classification_sha256": sha256(QUERY_CLASSIFICATION),
        "summary_sha256": sha256(SUMMARY),
        "report_sha256": sha256(REPORT),
        "new_judge_calls": 0,
        "new_model_calls": 0,
        "new_retrieval_calls": 0,
        "new_reranker_calls": 0,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4H_HEADROOM_STAGE_A_CLASSIFICATION_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "classification_contract_sha256": EXPECTED["classification_contract"],
        "fresh_stage_a_scoring_lock_sha256": EXPECTED["fresh_scoring_lock"],
        "fresh_stage_a_judge_scores_sha256": EXPECTED["fresh_scores"],
        "score_inventory_sha256": EXPECTED["score_inventory"],
        "combined_pair_count": 400,
        "reused_exact_pair_count": 60,
        "fresh_pair_count": 340,
        "classification_counts": summary["classification_counts"],
        "outputs": {
            COMPLETE_SCORES.name: sha256(COMPLETE_SCORES),
            QUERY_CLASSIFICATION.name: sha256(QUERY_CLASSIFICATION),
            SUMMARY.name: sha256(SUMMARY),
            REPORT.name: sha256(REPORT),
            METADATA.name: sha256(METADATA),
        },
        "new_judge_calls": 0,
        "new_model_calls": 0,
        "new_retrieval_calls": 0,
        "new_reranker_calls": 0,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "next_stage": summary["next_stage"],
    }
    CLASSIFICATION_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("PASS  60 exact reused + 340 fresh scores combined into frozen 400-pair Stage-A map")
    print("PASS  all 10 target queries classified using the pre-frozen score>=3 rule")
    print()
    print("V4H HEADROOM STAGE-A CLASSIFICATION: PASS")
    print(f"HEADROOM_FOUND_11_50: {found}/10")
    print(f"NO_CLEAR_THROUGH_50: {unresolved}/10")
    print(f"Classification lock SHA-256: {sha256(CLASSIFICATION_LOCK)}")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Human relevance labels used: NO")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    if unresolved:
        print("Next: optional Stage B on ranks 51-end for NO_CLEAR_THROUGH_50 queries only.")
    else:
        print("Next: close Stage-A; every target query has displayed-ranking headroom within ranks 11-50.")

if __name__ == "__main__":
    main()
