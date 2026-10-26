from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from verify_semantic_relevance_v4h_headroom_stage_b_classification import (
    CLASSIFICATION_LOCK,
    CLASSIFICATION_CONTRACT,
    COMPLETE_SCORES,
    EXPECTED,
    METADATA,
    OVERALL_DISPOSITION,
    QUERY_CLASSIFICATION,
    REPORT,
    SUMMARY,
    TARGET_QUERY_IDS,
    load_json,
    req,
    sha256,
    verify_boundary,
)


def label_from_score(score: int) -> str:
    return {0: "none", 1: "incidental", 2: "partial", 3: "clear", 4: "strong"}[int(score)]


def main() -> None:
    print("Semantic relevance V4H - headroom Stage-B deterministic classification")
    print("-----------------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Judge/model/retrieval/reranker calls: 0")
    print("Human relevance labels: NOT USED")
    print("Release metrics/gates: NOT CALCULATED")
    print("Frozen V4B release decision: UNCHANGED")
    print()

    boundary = verify_boundary()
    inventory = boundary["inventory"]
    fresh = boundary["fresh"]
    qinv = boundary["query_inventory"]
    stage_a_q = boundary["stage_a_query_classification"]

    for path in [COMPLETE_SCORES, QUERY_CLASSIFICATION, OVERALL_DISPOSITION, SUMMARY, REPORT, METADATA, CLASSIFICATION_LOCK]:
        req(not path.exists(), f"Refusing to overwrite existing Stage-B classification artifact: {path}")

    fresh_lookup = fresh.set_index("case_id")
    rows: list[dict[str, object]] = []
    for _, row in inventory.sort_values(["query_id", "v4b_rank"], kind="stable").iterrows():
        out = row.to_dict()
        if row["score_source"] == "fresh_judge_required":
            scored = fresh_lookup.loc[str(row["case_id"])]
            score = int(scored["judge_score"])
            match_level = str(scored["match_level"])
            provenance = "fresh_stage_b_frozen_judge_score"
        else:
            score = int(row["reused_judge_score"])
            match_level = label_from_score(score)
            provenance = "exact_reused_frozen_score"
        out["judge_score"] = score
        out["match_level"] = match_level
        out["score_provenance"] = provenance
        rows.append(out)

    complete = pd.DataFrame(rows)
    req(len(complete) == 252, "Combined Stage-B score map is not 252 rows")
    req(not complete["case_id"].duplicated().any(), "Combined Stage-B map has duplicate case_id")
    complete["judge_score"] = pd.to_numeric(complete["judge_score"], errors="raise").astype(int)
    complete["v4b_rank"] = pd.to_numeric(complete["v4b_rank"], errors="raise").astype(int)
    req(complete["judge_score"].between(0, 4).all(), "Combined Stage-B score outside 0..4")
    req(complete["v4b_rank"].ge(51).all(), "Combined Stage-B map contains rank below 51")
    COMPLETE_SCORES.parent.mkdir(parents=True, exist_ok=True)
    complete.to_csv(COMPLETE_SCORES, index=False, encoding="utf-8")

    qinv_idx = qinv.set_index("query_id")
    qrows: list[dict[str, object]] = []
    for qid in TARGET_QUERY_IDS:
        g = complete[complete["query_id"] == qid].sort_values("v4b_rank", kind="stable")
        expected_count = int(qinv_idx.loc[qid, "stage_b_candidate_count"])
        req(len(g) == expected_count,
            f"{qid} Stage-B row count mismatch: expected={expected_count} actual={len(g)}")
        clear = g[g["judge_score"] >= 3]
        clear_ranks = [int(v) for v in clear["v4b_rank"].tolist()]
        classification = "HEADROOM_FOUND_51_END" if clear_ranks else "NO_CLEAR_THROUGH_FULL_UNION"
        qrows.append({
            "query_id": qid,
            "query": str(g.iloc[0]["query"]) if len(g) and "query" in g.columns else "",
            "stratum": str(g.iloc[0]["stratum"]) if len(g) else str(stage_a_q.loc[stage_a_q["query_id"] == qid, "stratum"].iloc[0]),
            "stage_b_candidate_count": len(g),
            "stage_b_rank_start": int(g["v4b_rank"].min()) if len(g) else "",
            "stage_b_rank_end": int(g["v4b_rank"].max()) if len(g) else "",
            "reused_exact_count": int((g["score_source"] != "fresh_judge_required").sum()),
            "fresh_score_count": int((g["score_source"] == "fresh_judge_required").sum()),
            "best_score_51_end": int(g["judge_score"].max()) if len(g) else 0,
            "clear_or_strong_count_51_end": int((g["judge_score"] >= 3).sum()),
            "partial_count_51_end": int((g["judge_score"] == 2).sum()),
            "incidental_count_51_end": int((g["judge_score"] == 1).sum()),
            "irrelevant_count_51_end": int((g["judge_score"] == 0).sum()),
            "first_clear_or_strong_rank": "" if not clear_ranks else min(clear_ranks),
            "clear_or_strong_ranks_json": json.dumps(clear_ranks),
            "classification": classification,
        })

    q = pd.DataFrame(qrows)
    req(len(q) == 6, "Expected six Stage-B query classifications")
    q.to_csv(QUERY_CLASSIFICATION, index=False, encoding="utf-8")

    stage_b_map = q.set_index("query_id")["classification"].to_dict()
    overall_rows: list[dict[str, object]] = []
    for row in stage_a_q.itertuples(index=False):
        stage_a_class = str(row.classification)
        if stage_a_class == "HEADROOM_FOUND_11_50":
            final_disposition = "HEADROOM_FOUND_11_50"
            stage_b_class = "NOT_RUN_STAGE_A_ALREADY_RESOLVED"
        else:
            req(stage_a_class == "NO_CLEAR_THROUGH_50",
                f"Unexpected Stage-A classification for {row.query_id}: {stage_a_class}")
            req(str(row.query_id) in stage_b_map, f"Missing Stage-B classification for {row.query_id}")
            stage_b_class = stage_b_map[str(row.query_id)]
            final_disposition = stage_b_class
        overall_rows.append({
            "query_id": str(row.query_id),
            "stratum": str(row.stratum),
            "stage_a_classification": stage_a_class,
            "stage_b_classification": stage_b_class,
            "overall_headroom_disposition": final_disposition,
        })
    overall = pd.DataFrame(overall_rows)
    req(len(overall) == 10, "Expected 10-query overall disposition")
    OVERALL_DISPOSITION.write_text(overall.to_csv(index=False), encoding="utf-8")

    stage_b_counts = q["classification"].value_counts().to_dict()
    found_deep = int(stage_b_counts.get("HEADROOM_FOUND_51_END", 0))
    no_clear_full = int(stage_b_counts.get("NO_CLEAR_THROUGH_FULL_UNION", 0))
    req(found_deep + no_clear_full == 6, "Stage-B classification count mismatch")

    overall_counts = overall["overall_headroom_disposition"].value_counts().to_dict()
    found_11_50 = int(overall_counts.get("HEADROOM_FOUND_11_50", 0))
    found_51_end = int(overall_counts.get("HEADROOM_FOUND_51_END", 0))
    no_clear_union = int(overall_counts.get("NO_CLEAR_THROUGH_FULL_UNION", 0))
    req(found_11_50 + found_51_end + no_clear_union == 10,
        "Overall headroom disposition count mismatch")

    summary = {
        "status": "COMPLETE",
        "role": "post_release_consumed_holdout_diagnostic",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "stage_b_target_query_count": 6,
        "stage_b_rank_window": "51-end",
        "stage_b_combined_pair_count": 252,
        "stage_b_reused_exact_pair_count": 1,
        "stage_b_fresh_pair_count": 251,
        "clear_or_strong_threshold": "judge_score >= 3",
        "stage_b_classification_counts": {
            "HEADROOM_FOUND_51_END": found_deep,
            "NO_CLEAR_THROUGH_FULL_UNION": no_clear_full,
        },
        "overall_10_query_disposition_counts": {
            "HEADROOM_FOUND_11_50": found_11_50,
            "HEADROOM_FOUND_51_END": found_51_end,
            "NO_CLEAR_THROUGH_FULL_UNION": no_clear_union,
        },
        "headroom_found_51_end_query_ids": q.loc[
            q["classification"] == "HEADROOM_FOUND_51_END", "query_id"
        ].tolist(),
        "no_clear_through_full_union_query_ids": q.loc[
            q["classification"] == "NO_CLEAR_THROUGH_FULL_UNION", "query_id"
        ].tolist(),
        "interpretation": {
            "HEADROOM_FOUND_11_50": "Clear/strong material exists within frozen V4B ranks 11-50; displayed ranking failed to surface it.",
            "HEADROOM_FOUND_51_END": "Clear/strong material exists only deeper in the frozen V4B full union; deeper ranking headroom exists.",
            "NO_CLEAR_THROUGH_FULL_UNION": "No clear/strong result exists anywhere in the frozen V4B full union for this query. This is candidate-set limitation evidence for the frozen V4B retrieval/union boundary.",
        },
        "new_judge_calls": 0,
        "new_model_calls": 0,
        "new_retrieval_calls": 0,
        "new_reranker_calls": 0,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "next_stage": "CLOSE_V4H_HEADROOM_DIAGNOSTIC_AND_SYNTHESIZE_V5_DESIGN_EVIDENCE",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V4H Headroom Diagnostic — Stage B Classification",
        "",
        "**Consumed-holdout diagnostic only. V4B release decision remains frozen FAIL.**",
        "",
        "Stage B combines 1 exact same-query/same-ISBN frozen score with 251 frozen fresh Stage-B judge scores for the six queries that remained `NO_CLEAR_THROUGH_50` after Stage A.",
        "",
        "No judge, model, retrieval, or reranker calls occur in this classification stage.",
        "",
        "## Frozen Stage-B rule",
        "",
        "- `HEADROOM_FOUND_51_END`: at least one frozen V4B rank 51-end candidate has `judge_score >= 3`.",
        "- `NO_CLEAR_THROUGH_FULL_UNION`: no rank 51-end candidate has `judge_score >= 3`; combined with the frozen Stage-A result, no clear/strong result exists anywhere in the frozen V4B full union for that query.",
        "",
        "## Stage-B query classification",
        "",
        "| Query | Stratum | Candidates 51-end | Best score | Clear/strong count | First clear rank | Classification |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in q.itertuples(index=False):
        first = "—" if str(row.first_clear_or_strong_rank) == "" else str(row.first_clear_or_strong_rank)
        lines.append(
            f"| {row.query_id} | {row.stratum} | {row.stage_b_candidate_count} | "
            f"{row.best_score_51_end} | {row.clear_or_strong_count_51_end} | {first} | `{row.classification}` |"
        )
    lines += [
        "",
        "## Overall 10-query headroom disposition",
        "",
        f"- `HEADROOM_FOUND_11_50`: **{found_11_50}/10**",
        f"- `HEADROOM_FOUND_51_END`: **{found_51_end}/10**",
        f"- `NO_CLEAR_THROUGH_FULL_UNION`: **{no_clear_union}/10**",
        "",
        "## Interpretation boundary",
        "",
        "`NO_CLEAR_THROUGH_FULL_UNION` is evidence about the frozen V4B candidate set for the consumed V4H query. It does not reopen the failed release decision and does not by itself prescribe a new retrieval architecture. It is suitable only as post-release design evidence for the next development cycle.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "status": "V4H_HEADROOM_STAGE_B_CLASSIFICATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "classification_contract_sha256": EXPECTED["classification_contract"],
        "complete_scores_sha256": sha256(COMPLETE_SCORES),
        "query_classification_sha256": sha256(QUERY_CLASSIFICATION),
        "overall_disposition_sha256": sha256(OVERALL_DISPOSITION),
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
        "status": "V4H_HEADROOM_STAGE_B_CLASSIFICATION_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "classification_contract_sha256": EXPECTED["classification_contract"],
        "stage_a_classification_lock_sha256": EXPECTED["stage_a_classification_lock"],
        "fresh_stage_b_scoring_lock_sha256": EXPECTED["fresh_scoring_lock"],
        "fresh_stage_b_judge_scores_sha256": EXPECTED["fresh_scores"],
        "score_inventory_sha256": EXPECTED["score_inventory"],
        "combined_pair_count": 252,
        "reused_exact_pair_count": 1,
        "fresh_pair_count": 251,
        "stage_b_classification_counts": summary["stage_b_classification_counts"],
        "overall_10_query_disposition_counts": summary["overall_10_query_disposition_counts"],
        "outputs": {
            COMPLETE_SCORES.name: sha256(COMPLETE_SCORES),
            QUERY_CLASSIFICATION.name: sha256(QUERY_CLASSIFICATION),
            OVERALL_DISPOSITION.name: sha256(OVERALL_DISPOSITION),
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

    print("PASS  1 exact reused + 251 fresh scores combined into frozen 252-pair Stage-B map")
    print("PASS  all 6 Stage-B target queries classified using the pre-frozen score>=3 rule")
    print("PASS  frozen Stage-A classifications preserved in the 10-query overall disposition")
    print()
    print("V4H HEADROOM STAGE-B CLASSIFICATION: PASS")
    print(f"HEADROOM_FOUND_51_END: {found_deep}/6")
    print(f"NO_CLEAR_THROUGH_FULL_UNION: {no_clear_full}/6")
    print(f"OVERALL HEADROOM_FOUND_11_50: {found_11_50}/10")
    print(f"OVERALL HEADROOM_FOUND_51_END: {found_51_end}/10")
    print(f"OVERALL NO_CLEAR_THROUGH_FULL_UNION: {no_clear_union}/10")
    print(f"Classification lock SHA-256: {sha256(CLASSIFICATION_LOCK)}")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Human relevance labels used: NO")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: close the V4H headroom diagnostic and use the frozen evidence only for the next development-cycle design decision.")


if __name__ == "__main__":
    main()
