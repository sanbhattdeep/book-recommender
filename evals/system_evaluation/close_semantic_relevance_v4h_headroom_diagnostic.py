from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from verify_semantic_relevance_v4h_headroom_closeout import (
    CLOSEOUT_LOCK,
    CONTRACT,
    EXPECTED,
    HEADROOM_11_50,
    METADATA,
    NO_CLEAR_FULL_UNION,
    QUERY_COHORTS,
    REPORT,
    SUMMARY,
    req,
    sha256,
    verify_boundary,
)


def main() -> None:
    print("Semantic relevance V4H - headroom diagnostic closeout")
    print("-----------------------------------------------------")
    print("Role: consumed-holdout diagnostic closeout only")
    print("Judge/model/retrieval/reranker calls: 0")
    print("Human relevance labels: NOT USED")
    print("Release metrics/gates: NOT CALCULATED")
    print("Frozen V4B release decision: UNCHANGED")
    print()

    boundary = verify_boundary()
    failure_q = boundary["failure_q"].copy()
    overall = boundary["overall"].copy()

    for path in [QUERY_COHORTS, SUMMARY, REPORT, METADATA, CLOSEOUT_LOCK]:
        req(not path.exists(), f"Refusing to overwrite existing closeout artifact: {path}")

    fq = failure_q.set_index("query_id")
    rows: list[dict[str, object]] = []
    for row in overall.itertuples(index=False):
        qid = str(row.query_id)
        src = fq.loc[qid]
        disposition = str(row.overall_headroom_disposition)
        if disposition == "HEADROOM_FOUND_11_50":
            evidence_class = "RANKING_DISPLAY_HEADROOM_WITHIN_FROZEN_UNION"
        elif disposition == "NO_CLEAR_THROUGH_FULL_UNION":
            evidence_class = "CANDIDATE_SET_LIMITATION_IN_FROZEN_V4B_UNION"
        else:
            raise ValueError(f"Unexpected final disposition for {qid}: {disposition}")
        rows.append({
            "query_id": qid,
            "query": str(src["query"]),
            "stratum": str(src["stratum"]),
            "original_v4_failure_signature": str(src["v4_failure_signature"]),
            "stage_a_classification": str(row.stage_a_classification),
            "stage_b_classification": str(row.stage_b_classification),
            "overall_headroom_disposition": disposition,
            "design_evidence_class": evidence_class,
        })

    cohorts = pd.DataFrame(rows)
    req(len(cohorts) == 10, "Expected 10 closeout cohort rows")
    QUERY_COHORTS.parent.mkdir(parents=True, exist_ok=True)
    cohorts.to_csv(QUERY_COHORTS, index=False, encoding="utf-8")

    full_union_rows = cohorts[cohorts["overall_headroom_disposition"] == "NO_CLEAR_THROUGH_FULL_UNION"]
    headroom_rows = cohorts[cohorts["overall_headroom_disposition"] == "HEADROOM_FOUND_11_50"]
    stratum_counts = full_union_rows["stratum"].value_counts().to_dict()

    summary = {
        "status": "COMPLETE",
        "role": "post_release_consumed_holdout_diagnostic_closeout",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "original_no_clear_top10_query_count": 10,
        "final_headroom_disposition_counts": {
            "HEADROOM_FOUND_11_50": 4,
            "HEADROOM_FOUND_51_END": 0,
            "NO_CLEAR_THROUGH_FULL_UNION": 6,
        },
        "headroom_found_11_50_query_ids": HEADROOM_11_50,
        "headroom_found_51_end_query_ids": [],
        "no_clear_through_full_union_query_ids": NO_CLEAR_FULL_UNION,
        "no_clear_through_full_union_stratum_counts": stratum_counts,
        "diagnostic_conclusions": {
            "ranking_display_headroom": "Four of the ten original no-clear-top10 queries contain at least one clear/strong candidate at frozen V4B ranks 11-50.",
            "deep_rank_headroom": "None of the six Stage-A-unresolved queries gains a clear/strong candidate at ranks 51-end.",
            "candidate_set_limitation": "Six of ten original no-clear-top10 queries contain no clear/strong candidate anywhere in the frozen V4B full union.",
            "architecture_boundary": "The six full-union cases justify candidate-generation/retrieval work in the next development cycle, but the consumed-holdout evidence does not select a specific architecture or permit tuning on V4H.",
        },
        "non_target_query_boundary": "The other 14 V4H queries were not part of this no-clear headroom diagnostic because their displayed V4B top10 already contained at least one clear/strong result.",
        "new_judge_calls": 0,
        "new_model_calls": 0,
        "new_retrieval_calls": 0,
        "new_reranker_calls": 0,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "next_stage": "SYNTHESIZE_V5_DEVELOPMENT_DESIGN_FROM_FROZEN_V4H_EVIDENCE",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# V4H Headroom Diagnostic Closeout",
        "",
        "**Consumed-holdout diagnostic only. The frozen V4B release decision remains FAIL.**",
        "",
        "## Final disposition of the 10 original `NO_CLEAR_TOP10` queries",
        "",
        "- `HEADROOM_FOUND_11_50`: **4/10**",
        "- `HEADROOM_FOUND_51_END`: **0/10**",
        "- `NO_CLEAR_THROUGH_FULL_UNION`: **6/10**",
        "",
        "The Stage-B result is especially constraining: none of the six queries unresolved through rank 50 acquired a clear/strong candidate deeper in the frozen V4B full union.",
        "",
        "## Query cohorts",
        "",
        "| Query | Stratum | Final disposition | Design evidence |",
        "|---|---|---|---|",
    ]
    for row in cohorts.itertuples(index=False):
        lines.append(f"| {row.query_id} | {row.stratum} | `{row.overall_headroom_disposition}` | `{row.design_evidence_class}` |")
    lines += [
        "",
        "## What this establishes",
        "",
        "1. **Ranking/display headroom exists for four queries.** Clear/strong material was present in the frozen candidate union at ranks 11-50 but did not reach the displayed top 10.",
        "2. **There is no ranks-51-to-end-only rescue cohort.** Extending the same frozen union deeper did not reveal a new clear/strong match for any of the six Stage-A-unresolved queries.",
        "3. **Six queries show candidate-set limitation evidence.** For those consumed-holdout queries, the frozen V4B union contains no clear/strong candidate at any rank.",
        "4. **This does not prescribe a retrieval architecture.** It supports retrieval/candidate-generation work in the next development cycle, while architecture selection and tuning must use development evidence rather than V4H.",
        "",
        "## Next development-cycle implication",
        "",
        "The evidence supports a two-track V5 design investigation: improve candidate generation for the six full-union-limited queries, and improve ranking/display selection for cases where relevant material already exists in the candidate pool. The remaining 14 V4H queries require their existing failure signatures to be considered separately; this headroom diagnostic does not reclassify them.",
        "",
        "No release metrics are recalculated and V4H remains consumed.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    metadata = {
        "status": "V4H_HEADROOM_DIAGNOSTIC_CLOSEOUT_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic_closeout",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "closeout_contract_sha256": EXPECTED["contract"],
        "failure_analysis_lock_sha256": EXPECTED["failure_lock"],
        "stage_a_classification_lock_sha256": EXPECTED["stage_a_lock"],
        "stage_b_classification_lock_sha256": EXPECTED["stage_b_lock"],
        "query_cohorts_sha256": sha256(QUERY_COHORTS),
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
        "status": "V4H_HEADROOM_DIAGNOSTIC_CLOSED_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic_closeout",
        "v4h_consumed": True,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "closeout_contract_sha256": EXPECTED["contract"],
        "failure_analysis_lock_sha256": EXPECTED["failure_lock"],
        "final_release_analysis_lock_sha256": EXPECTED["final_release_lock"],
        "stage_a_classification_lock_sha256": EXPECTED["stage_a_lock"],
        "stage_b_classification_lock_sha256": EXPECTED["stage_b_lock"],
        "final_headroom_disposition_counts": summary["final_headroom_disposition_counts"],
        "headroom_found_11_50_query_ids": HEADROOM_11_50,
        "headroom_found_51_end_query_ids": [],
        "no_clear_through_full_union_query_ids": NO_CLEAR_FULL_UNION,
        "outputs": {
            QUERY_COHORTS.name: sha256(QUERY_COHORTS),
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
    CLOSEOUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("PASS  original V4H failure signatures joined to frozen 10-query headroom disposition")
    print("PASS  four ranks-11-50 ranking/display-headroom queries frozen")
    print("PASS  zero ranks-51-end-only rescue queries frozen")
    print("PASS  six full-union candidate-set-limitation queries frozen")
    print()
    print("V4H HEADROOM DIAGNOSTIC CLOSEOUT: PASS")
    print("HEADROOM_FOUND_11_50: 4/10")
    print("HEADROOM_FOUND_51_END: 0/10")
    print("NO_CLEAR_THROUGH_FULL_UNION: 6/10")
    print(f"Closeout lock SHA-256: {sha256(CLOSEOUT_LOCK)}")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Human relevance labels used: NO")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: synthesize V5 development design from the frozen V4H evidence without tuning on V4H.")


if __name__ == "__main__":
    main()
