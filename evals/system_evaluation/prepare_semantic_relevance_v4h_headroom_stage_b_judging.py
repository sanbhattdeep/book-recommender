from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))
SYS = EVALS / "system_evaluation"
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"
HIST_RUN = ROOT / "evals/runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"

# Frozen Stage-B candidate boundary.
CANDIDATES = RUN / "v4h_no_clear_headroom_stage_b_candidates.csv"
QUERY_INVENTORY = RUN / "v4h_no_clear_headroom_stage_b_query_inventory.csv"
CANDIDATE_LOCK = RUN / "v4h_no_clear_headroom_stage_b.lock.json"
STAGE_B_DEFINITION_CONTRACT = SYS / "semantic_relevance_v4h_no_clear_headroom_stage_b.v0.1.0.json"
STAGE_A_CLASSIFICATION_LOCK = RUN / "v4h_headroom_stage_a_classification.lock.json"
FINAL_RELEASE_LOCK = RUN / "v4h_final_release_analysis.lock.json"

# Frozen score sources eligible for exact same-query/same-book reuse.
V4H_DISPLAY_SCORES = RUN / "v4h_judge_scores.csv"
V4H_DISPLAY_SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"
STAGE_A_FRESH_SCORES = RUN / "v4h_headroom_stage_a_fresh_judge_scores.csv"
STAGE_A_FRESH_SCORING_LOCK = RUN / "v4h_headroom_stage_a_fresh_judge_scoring.lock.json"

# Frozen measurement instrument provenance.
V4H_FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v4_holdout_query_facets.v0.1.0.json"
V4H_AUTHORING_LOCK = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"
HISTORICAL_SCORING_LOCK = HIST_RUN / "holdout_judge_scoring_lock.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = SYS / "semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
JUDGING_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_b_judging_contract.v0.1.0.json"

# Outputs. These are frozen before any Stage-B judge execution.
INVENTORY = RUN / "v4h_headroom_stage_b_score_inventory.csv"
JUDGING_INPUT = RUN / "v4h_headroom_stage_b_judging_input.csv"
JUDGING_INPUT_LOCK = RUN / "v4h_headroom_stage_b_judging_input.lock.json"

from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, validate_query_facet_spec

TARGET_QUERY_IDS = ["V4H10", "V4H11", "V4H13", "V4H16", "V4H21", "V4H23"]
EXPECTED_CANDIDATE_ROWS = 252
EXPECTED_CANDIDATE_SHA256 = "910f1a3f05487ab95f83cf3a7c6415750957a35ff9cf0e95a697b209a58c255f"
EXPECTED_QUERY_INVENTORY_SHA256 = "bca3146d181d52ee6c3e8f43db3795ccf79ce764799fc0089d34067b89ad302b"
EXPECTED_CANDIDATE_LOCK_SHA256 = "eb598bec298d8dd617d7b291a3274b9e1c4ae3dcf5f45c8bf98708a6a179023f"
EXPECTED_STAGE_B_DEFINITION_CONTRACT_SHA256 = "c26af8da897995b817d9eeffef56f94a2fd4cab99dca16777d85f569759a4476"
EXPECTED_STAGE_A_CLASSIFICATION_LOCK_SHA256 = "906b5975345c9d9eed1273ecc4ccd0611880b14d6326bf9f0e74b46f7d63d948"
EXPECTED_FINAL_RELEASE_LOCK_SHA256 = "7e7d73920452ec39ea72947d2b3762a02aaa1b2bb0bcd7470c7e21dbb3a25ab0"
EXPECTED_V4H_DISPLAY_SCORES_SHA256 = "8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906"
EXPECTED_V4H_DISPLAY_SCORING_LOCK_SHA256 = "ac7a96c89eb2d4a51eda1b56756ed3b943311e7f300b661659cff51285423a8f"
EXPECTED_STAGE_A_FRESH_SCORES_SHA256 = "5a1c988e0efb6898e721f700261e834ed1994de107338dc49e68549aa9e1ca02"
EXPECTED_STAGE_A_FRESH_SCORING_LOCK_SHA256 = "b64ff7d8828fcd4d8938c487d48335c0837aaca0d6439f42b91579467a5c57f9"
EXPECTED_V4H_FACET_SHA256 = "dbdb7c5d5603e17164cf6ef880df912f6196663387885e1f5773e54b97d2b106"
EXPECTED_V4H_AUTHORING_LOCK_SHA256 = "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7"
EXPECTED_V3_SCORING_LOCK_SHA256 = "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
EXPECTED_SPAN_ECHO_PATCH_SHA256 = "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"
EXPECTED_CONTRACT_SHA256 = "45ceaac688a3e6f6c733c5343a38590652cbfa9fa0236db68e1b4e6835322aef"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")


def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw, norm = sha256(path), lf_sha256(path)
    req(raw == expected or norm == expected,
        f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
    print(f"PASS  {label} {'raw' if raw == expected else 'canonical-LF'} SHA-256 verified")


def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    req(bool(text), "ISBN is empty")
    return text


def load_facets() -> dict[str, QueryFacetSpec]:
    payload = load_json(V4H_FACET_FILE)
    raw_queries = payload.get("queries")
    req(isinstance(raw_queries, list), "Facet spec must contain queries list")
    specs: dict[str, QueryFacetSpec] = {}
    for raw in raw_queries:
        qid = str(raw["query_id"])
        spec = QueryFacetSpec(
            query_id=qid,
            query=str(raw["query"]),
            facets=[QueryFacet(**f) for f in raw["facets"]],
        )
        validate_query_facet_spec(spec)
        specs[qid] = spec
    req(all(q in specs for q in TARGET_QUERY_IDS), "Target query missing from frozen V4H facet spec")
    return specs


def verify_measurement_instrument() -> dict[str, QueryFacetSpec]:
    verify_raw(HISTORICAL_SCORING_LOCK, EXPECTED_V3_SCORING_LOCK_SHA256,
               "V3 frozen holdout scoring lock")
    hlock = load_json(HISTORICAL_SCORING_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "Unexpected historical scoring-lock status")
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5",
        "Judge candidate changed")
    req(hlock.get("human_relevance_labels_used") is False,
        "Historical measurement instrument used human labels")

    for path, key, label in [
        (JUDGE_FILE, "judge_source_sha256", "frozen judge source"),
        (SCORING_FILE, "scoring_source_sha256", "frozen deterministic scoring"),
        (JUDGE_CONFIG_FILE, "judge_config_sha256", "frozen judge config"),
        (RUBRIC_FILE, "rubric_sha256", "frozen rubric"),
    ]:
        expected = str(hlock.get(key, ""))
        req(bool(expected), f"Historical lock missing {key}")
        verify_text(path, expected, label)

    verify_raw(V4H_FACET_FILE, EXPECTED_V4H_FACET_SHA256, "V4H facet specification")
    verify_raw(V4H_AUTHORING_LOCK, EXPECTED_V4H_AUTHORING_LOCK_SHA256, "V4H authoring lock")
    verify_text(SPAN_ECHO_PATCH_FILE, EXPECTED_SPAN_ECHO_PATCH_SHA256,
                "accepted span-echo execution patch")

    config = load_json(JUDGE_CONFIG_FILE)
    req(config.get("version") == "0.29.0-r5", "Judge config version changed")
    req(config.get("provider") == "ollama", "Judge provider changed")
    req(float(config.get("temperature")) == 0.0, "Judge temperature changed")
    rubric = load_json(RUBRIC_FILE)
    req(rubric.get("status") == "frozen", "Rubric is not frozen")
    req(rubric.get("version") == "0.1.0", "Rubric version changed")

    specs = load_facets()
    print("PASS  frozen v0.29.0-r5 measurement instrument verified")
    print("PASS  target V4H facets validated against frozen judge schema")
    return specs


def load_score_map(path: Path, source_name: str) -> dict[tuple[str, str], str]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    req({"query_id", "isbn13", "judge_score"}.issubset(frame.columns),
        f"{source_name} score artifact missing required columns")
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    req(not frame.duplicated(["query_id", "isbn13"]).any(),
        f"{source_name} score artifact has duplicate query-book pair")
    scores = pd.to_numeric(frame["judge_score"], errors="raise").astype(int)
    req(scores.between(0, 4).all(), f"{source_name} contains judge score outside 0..4")
    frame["judge_score"] = scores.astype(str)
    return frame.set_index(["query_id", "isbn13"])["judge_score"].to_dict()


def deterministic_case_id(query_id: str, isbn13: str) -> str:
    return f"V4H_HEADROOM_B_{query_id}_{canonical_isbn(isbn13)}"


def main() -> None:
    print("Semantic relevance V4H - headroom Stage-B judging preparation")
    print("--------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Judge execution: DISABLED")
    print("Exact frozen-score reuse: ENABLED for same query + same ISBN only")
    print("Cross-query/approximate reuse: DISABLED")
    print("Release decision modification: DISABLED")
    print()

    # Preparation is one-shot: never overwrite an already-frozen partition.
    for path in [INVENTORY, JUDGING_INPUT, JUDGING_INPUT_LOCK]:
        req(not path.exists(), f"Refusing to overwrite existing Stage-B judging-preparation artifact: {path}")

    verify_raw(CANDIDATES, EXPECTED_CANDIDATE_SHA256, "Stage-B frozen candidate inventory")
    verify_raw(QUERY_INVENTORY, EXPECTED_QUERY_INVENTORY_SHA256, "Stage-B query inventory")
    verify_raw(CANDIDATE_LOCK, EXPECTED_CANDIDATE_LOCK_SHA256, "Stage-B candidate lock")
    verify_raw(STAGE_B_DEFINITION_CONTRACT, EXPECTED_STAGE_B_DEFINITION_CONTRACT_SHA256,
               "Stage-B definition contract")
    verify_raw(STAGE_A_CLASSIFICATION_LOCK, EXPECTED_STAGE_A_CLASSIFICATION_LOCK_SHA256,
               "Stage-A classification lock")
    verify_raw(FINAL_RELEASE_LOCK, EXPECTED_FINAL_RELEASE_LOCK_SHA256,
               "V4H final release-analysis lock")
    verify_raw(V4H_DISPLAY_SCORES, EXPECTED_V4H_DISPLAY_SCORES_SHA256,
               "frozen V4H displayed-union judge scores")
    verify_raw(V4H_DISPLAY_SCORING_LOCK, EXPECTED_V4H_DISPLAY_SCORING_LOCK_SHA256,
               "frozen V4H displayed-union scoring lock")
    verify_raw(STAGE_A_FRESH_SCORES, EXPECTED_STAGE_A_FRESH_SCORES_SHA256,
               "frozen Stage-A fresh judge scores")
    verify_raw(STAGE_A_FRESH_SCORING_LOCK, EXPECTED_STAGE_A_FRESH_SCORING_LOCK_SHA256,
               "frozen Stage-A fresh scoring lock")
    verify_raw(JUDGING_CONTRACT, EXPECTED_CONTRACT_SHA256,
               "Stage-B judging-preparation contract")

    clock = load_json(CANDIDATE_LOCK)
    req(clock.get("status") == "V4H_NO_CLEAR_HEADROOM_STAGE_B_CANDIDATES_FROZEN_BEFORE_JUDGING",
        "Unexpected Stage-B candidate-lock status")
    req(int(clock.get("candidate_count", -1)) == EXPECTED_CANDIDATE_ROWS,
        "Stage-B candidate count changed")
    req(clock.get("candidate_sha256") == EXPECTED_CANDIDATE_SHA256,
        "Stage-B candidate lock does not bind expected candidate artifact")
    req(clock.get("query_inventory_sha256") == EXPECTED_QUERY_INVENTORY_SHA256,
        "Stage-B candidate lock does not bind expected query inventory")
    req(clock.get("target_query_ids") == TARGET_QUERY_IDS,
        f"Stage-B target query set changed: {clock.get('target_query_ids')}")
    req(clock.get("rank_window") == "51-end", "Stage-B rank window changed")
    req(clock.get("release_decision_modified") is False,
        "Stage-B candidate lock says release decision was modified")

    display_lock = load_json(V4H_DISPLAY_SCORING_LOCK)
    req(display_lock.get("status") == "V4H_JUDGE_SCORES_FROZEN_BEFORE_RELEASE_ANALYSIS",
        "Unexpected V4H displayed-union scoring-lock status")
    req(display_lock.get("judge_scores_sha256") == EXPECTED_V4H_DISPLAY_SCORES_SHA256,
        "Displayed-union score lock does not bind expected score artifact")
    req(display_lock.get("human_relevance_labels_used") is False,
        "Displayed-union score lock says human labels were used")

    stage_a_lock = load_json(STAGE_A_FRESH_SCORING_LOCK)
    req(stage_a_lock.get("status") == "V4H_HEADROOM_STAGE_A_FRESH_JUDGE_SCORES_FROZEN",
        "Unexpected Stage-A fresh scoring-lock status")
    req(stage_a_lock.get("fresh_judge_scores_sha256") == EXPECTED_STAGE_A_FRESH_SCORES_SHA256,
        "Stage-A fresh scoring lock does not bind expected score artifact")
    req(stage_a_lock.get("human_relevance_labels_used") is False,
        "Stage-A fresh scoring lock says human labels were used")
    req(stage_a_lock.get("release_decision_modified") is False,
        "Stage-A fresh scoring lock says release decision was modified")

    specs = verify_measurement_instrument()

    candidates = pd.read_csv(CANDIDATES, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id", "query", "stratum", "v4b_rank", "isbn13", "title", "description"}
    missing = required - set(candidates.columns)
    req(not missing, f"Stage-B candidate inventory missing columns: {sorted(missing)}")
    req(len(candidates) == EXPECTED_CANDIDATE_ROWS,
        f"Expected {EXPECTED_CANDIDATE_ROWS} Stage-B candidates, found {len(candidates)}")
    candidates["isbn13"] = candidates["isbn13"].map(canonical_isbn)
    candidates["v4b_rank_num"] = pd.to_numeric(candidates["v4b_rank"], errors="raise").astype(int)
    req(not candidates.duplicated(["query_id", "isbn13"]).any(),
        "Duplicate Stage-B query-book pair")
    req(candidates["query_id"].drop_duplicates().tolist() == TARGET_QUERY_IDS,
        "Stage-B candidate query order/set changed")
    req((candidates["v4b_rank_num"] >= 51).all(), "Stage-B candidate outside ranks 51-end")
    req(candidates["description"].str.strip().ne("").all(), "Stage-B candidate has empty description")

    expected_counts = {str(k): int(v) for k, v in clock.get("candidate_counts_by_query", {}).items()}
    actual_counts = candidates.groupby("query_id", sort=False).size().astype(int).to_dict()
    req(actual_counts == expected_counts,
        f"Stage-B per-query candidate counts changed: expected={expected_counts} actual={actual_counts}")

    for qid, g in candidates.groupby("query_id", sort=False):
        req(g["query"].nunique() == 1, f"{qid} has multiple query texts")
        req(str(g["query"].iloc[0]) == specs[qid].query,
            f"{qid} query text disagrees with frozen facet specification")

    display_scores = load_score_map(V4H_DISPLAY_SCORES, "V4H displayed-union")
    stage_a_scores = load_score_map(STAGE_A_FRESH_SCORES, "Stage-A fresh")
    overlap = set(display_scores).intersection(stage_a_scores)
    req(not overlap,
        f"Frozen displayed-union and Stage-A fresh score sources unexpectedly overlap: {sorted(overlap)[:5]}")

    # Build one provenance-preserving exact-key lookup across all frozen sources.
    score_lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for key, score in display_scores.items():
        score_lookup[key] = (score, "reused_frozen_v4h_display_score")
    for key, score in stage_a_scores.items():
        score_lookup[key] = (score, "reused_frozen_stage_a_fresh_score")

    inventory = candidates.copy()
    inventory.insert(0, "case_id", [
        deterministic_case_id(q, i)
        for q, i in zip(inventory["query_id"], inventory["isbn13"])
    ])
    inventory["score_source"] = [
        score_lookup.get((q, i), ("", "fresh_judge_required"))[1]
        for q, i in zip(inventory["query_id"], inventory["isbn13"])
    ]
    inventory["reused_judge_score"] = [
        score_lookup.get((q, i), ("", "fresh_judge_required"))[0]
        for q, i in zip(inventory["query_id"], inventory["isbn13"])
    ]
    req(not inventory["case_id"].duplicated().any(), "Duplicate Stage-B deterministic case ID")
    inventory = inventory.sort_values(["query_id", "v4b_rank_num"], kind="stable").reset_index(drop=True)
    inventory.to_csv(INVENTORY, index=False, encoding="utf-8")

    fresh = inventory[inventory["score_source"] == "fresh_judge_required"].copy()
    reused_display = inventory[inventory["score_source"] == "reused_frozen_v4h_display_score"].copy()
    reused_stage_a = inventory[inventory["score_source"] == "reused_frozen_stage_a_fresh_score"].copy()
    reused = pd.concat([reused_display, reused_stage_a], ignore_index=True)
    req(len(fresh) + len(reused) == EXPECTED_CANDIDATE_ROWS, "Stage-B inventory partition mismatch")

    drop_cols = ["score_source", "reused_judge_score", "v4b_rank_num"]
    fresh.drop(columns=[c for c in drop_cols if c in fresh.columns]).to_csv(
        JUDGING_INPUT, index=False, encoding="utf-8"
    )

    lock = {
        "status": "V4H_HEADROOM_STAGE_B_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "target_query_ids": TARGET_QUERY_IDS,
        "candidate_count": EXPECTED_CANDIDATE_ROWS,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "candidate_lock_sha256": EXPECTED_CANDIDATE_LOCK_SHA256,
        "stage_b_definition_contract_sha256": EXPECTED_STAGE_B_DEFINITION_CONTRACT_SHA256,
        "judging_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "score_inventory_sha256": sha256(INVENTORY),
        "fresh_judging_input_sha256": sha256(JUDGING_INPUT),
        "exact_score_reuse_key": ["query_id", "isbn13"],
        "reuse_source_sha256": {
            "v4h_displayed_union_scores": EXPECTED_V4H_DISPLAY_SCORES_SHA256,
            "v4h_displayed_union_scoring_lock": EXPECTED_V4H_DISPLAY_SCORING_LOCK_SHA256,
            "stage_a_fresh_scores": EXPECTED_STAGE_A_FRESH_SCORES_SHA256,
            "stage_a_fresh_scoring_lock": EXPECTED_STAGE_A_FRESH_SCORING_LOCK_SHA256,
        },
        "reused_exact_pair_count": len(reused),
        "reused_displayed_union_pair_count": len(reused_display),
        "reused_stage_a_fresh_pair_count": len(reused_stage_a),
        "fresh_pair_count": len(fresh),
        "total_stage_b_pair_count": EXPECTED_CANDIDATE_ROWS,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "v4h_facet_spec_sha256": EXPECTED_V4H_FACET_SHA256,
        "cross_query_score_reuse_allowed": False,
        "approximate_score_reuse_allowed": False,
        "judge_execution_performed": False,
        "model_execution_performed": False,
        "retrieval_execution_performed": False,
        "reranker_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "frozen_release_decision": "FAIL",
        "release_decision_modified": False,
        "case_id_rule": "V4H_HEADROOM_B_<query_id>_<canonical_isbn13>",
        "next_stage": "PREPARE_FROZEN_JUDGE_EXECUTION_FOR_FRESH_STAGE_B_PAIRS_ONLY",
    }
    JUDGING_INPUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  Stage-B score inventory frozen: {len(inventory)} pairs")
    print(f"PASS  exact frozen-score reuse total: {len(reused)} pairs")
    print(f"INFO  reused from frozen V4H displayed-union scores: {len(reused_display)} pairs")
    print(f"INFO  reused from frozen Stage-A fresh scores: {len(reused_stage_a)} pairs")
    print(f"PASS  fresh Stage-B judge input frozen: {len(fresh)} pairs")
    print("PASS  cross-query and approximate score reuse disabled")
    print("PASS  no human relevance labels present")
    print()
    print("V4H HEADROOM STAGE-B JUDGING PREPARATION: PASS")
    print(f"Score inventory SHA-256: {sha256(INVENTORY)}")
    print(f"Fresh judging input SHA-256: {sha256(JUDGING_INPUT)}")
    print(f"Judging-input lock SHA-256: {sha256(JUDGING_INPUT_LOCK)}")
    print("Judge/model/retrieval/reranker calls performed: 0")
    print("Release metrics calculated: NO")
    print("Frozen V4B release decision modified: NO")
    print("Next: prepare frozen judge execution for fresh Stage-B pairs only.")


if __name__ == "__main__":
    main()
