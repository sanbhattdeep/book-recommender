from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import verify_semantic_relevance_v5_development_judging_input as boundary

RUN = boundary.RUN
INVENTORY = RUN / "v5d_display_score_inventory.csv"
NOVEL = RUN / "v5d_novel_display_pairs_to_judge.csv"
METADATA = RUN / "v5d_judging_input_metadata.json"
LOCK = RUN / "v5d_judging_input.lock.json"

EXPECTED_VERIFIER_SHA256 = "f6e252442e0768a7d7647faddb8e769df0ed07bc195d53821cc6764db4ebffbd"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_verifier_unchanged() -> None:
    path = Path(boundary.__file__).resolve()
    raw = sha256(path)
    normalized = lf_sha256(path)
    req(raw == EXPECTED_VERIFIER_SHA256 or normalized == EXPECTED_VERIFIER_SHA256,
        "Judging-input preflight verifier changed after package freeze")
    print("PASS  judging-input preflight verifier content verified")


def main() -> None:
    print("Semantic relevance V5 - freeze score-reuse decision and novel judge input")
    print("-----------------------------------------------------------------------")
    print("Retrieval/reranking: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Development metrics/selection: NOT CALCULATED")
    print()

    frame = boundary.validate()
    verify_verifier_unchanged()
    for out in [INVENTORY, NOVEL, METADATA, LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing judging-input artifact: {out}")

    inventory = frame.copy()
    inventory["score_status"] = "novel"
    inventory["score_source"] = ""
    inventory["judge_score"] = ""
    inventory["match_level"] = ""
    inventory["judge_reason"] = ""
    inventory["scoring_explanation"] = ""
    inventory["facet_assessments_json"] = ""
    inventory["facet_evidence_json"] = ""
    inventory["component_evidence_ledger_json"] = ""
    inventory = inventory.sort_values(["query_id", "isbn13"], kind="stable").reset_index(drop=True)
    inventory.to_csv(INVENTORY, index=False, encoding="utf-8")

    novel = frame.copy()
    novel.insert(
        0,
        "case_id",
        novel.apply(lambda r: f"V5D_JUDGE_{r['query_id']}_{boundary.canonical_isbn(r['isbn13'])}", axis=1),
    )
    req(not novel["case_id"].duplicated().any(), "Novel V5D case IDs are not unique")
    novel["_min_rank"] = novel[["B_rank", "R_rank", "K_rank", "RK_rank"]].apply(
        lambda row: min([int(x) for x in row if str(x).strip() != ""]),
        axis=1,
    )
    novel = novel.sort_values(["query_id", "_min_rank", "isbn13"], kind="stable").drop(
        columns=["_min_rank"]
    ).reset_index(drop=True)
    novel.to_csv(NOVEL, index=False, encoding="utf-8")

    req(len(inventory) == boundary.EXPECTED_PAIRS and len(novel) == boundary.EXPECTED_PAIRS,
        "Judging-input outputs do not contain exactly 452 pairs")
    req(set(zip(inventory["query_id"], inventory["isbn13"])) ==
        set(zip(novel["query_id"], novel["isbn13"])),
        "Inventory and novel judging input pair sets differ")

    per_query = novel.groupby("query_id").size()
    metadata = {
        "status": "V5D_JUDGING_INPUT_PREPARATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_set": "V5D01-V5D24",
        "query_count": 24,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "source_display_union_sha256": boundary.EXPECTED["display_union"],
        "source_display_union_lock_sha256": boundary.EXPECTED["display_lock"],
        "historical_reuse_rule": "exact same query text + same canonical ISBN only",
        "historical_compatible_query_universe": "Q01-Q12 + H01-H24 + V4H01-V4H24",
        "frozen_prior_query_manifest_sha256": boundary.EXPECTED["prior_manifest"],
        "reused_frozen_scores": 0,
        "novel_pairs_to_judge": int(len(novel)),
        "novel_pairs_per_query_min": int(per_query.min()),
        "novel_pairs_per_query_max": int(per_query.max()),
        "new_judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_metrics_calculated": False,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "outputs": {
            INVENTORY.name: sha256(INVENTORY),
            NOVEL.name: sha256(NOVEL),
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V5D_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_set": "V5D01-V5D24",
        "query_count": 24,
        "case_count": int(len(novel)),
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "judging_input_contract_sha256": boundary.EXPECTED["contract"],
        "source_display_union_sha256": boundary.EXPECTED["display_union"],
        "source_display_union_lock_sha256": boundary.EXPECTED["display_lock"],
        "v5d_facet_spec_sha256": boundary.EXPECTED["facet"],
        "v5d_authoring_lock_sha256": boundary.EXPECTED["authoring_lock"],
        "frozen_prior_query_manifest_sha256": boundary.EXPECTED["prior_manifest"],
        "historical_measurement_instrument_lock_sha256": boundary.EXPECTED["historical_scoring_lock"],
        "historical_score_reuse_rule": "exact same query text + same canonical ISBN only",
        "historical_score_reuse_performed": False,
        "reused_frozen_scores": 0,
        "fresh_pairs_to_judge": int(len(novel)),
        "score_inventory_sha256": sha256(INVENTORY),
        "novel_judging_input_sha256": sha256(NOVEL),
        "judging_input_metadata_sha256": sha256(METADATA),
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "development_metrics_calculated": False,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "PREPARE_V5D_V029_R5_JUDGE_EXECUTION_PACKAGE",
    }
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("PASS  exact-query historical score-reuse decision frozen: 0 reused")
    print("PASS  all 452 frozen displayed pairs materialized as novel judge cases")
    print("PASS  V5D judge input frozen before any relevance judge call")
    print()
    print("V5D JUDGING-INPUT FREEZE: PASS")
    print(f"Score inventory SHA-256: {sha256(INVENTORY)}")
    print(f"Novel judging input SHA-256: {sha256(NOVEL)}")
    print(f"Judging-input lock SHA-256: {sha256(LOCK)}")
    print("Reused frozen scores: 0")
    print(f"Fresh pairs to judge: {len(novel)}")
    print("Judge calls performed: 0")
    print("Development selection calculated: NO")
    print("Next: prepare the frozen v0.29.0-r5 execution package for these 452 cases.")


if __name__ == "__main__":
    main()
