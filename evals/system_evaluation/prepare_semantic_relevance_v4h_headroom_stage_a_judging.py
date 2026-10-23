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

CANDIDATES = RUN / "v4h_no_clear_headroom_stage_a_candidates.csv"
CANDIDATE_LOCK = RUN / "v4h_no_clear_headroom_stage_a.lock.json"
EXISTING_SCORES = RUN / "v4h_judge_scores.csv"
EXISTING_SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"

V4H_FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v4_holdout_query_facets.v0.1.0.json"
V4H_AUTHORING_LOCK = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"
HISTORICAL_SCORING_LOCK = HIST_RUN / "holdout_judge_scoring_lock.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = SYS / "semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
JUDGING_CONTRACT = SYS / "semantic_relevance_v4h_headroom_stage_a_judging_contract.v0.1.0.json"

INVENTORY = RUN / "v4h_headroom_stage_a_score_inventory.csv"
JUDGING_INPUT = RUN / "v4h_headroom_stage_a_judging_input.csv"
JUDGING_INPUT_LOCK = RUN / "v4h_headroom_stage_a_judging_input.lock.json"

from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, validate_query_facet_spec

TARGET_QUERY_IDS = ['V4H07', 'V4H09', 'V4H10', 'V4H11', 'V4H13', 'V4H16', 'V4H19', 'V4H21', 'V4H22', 'V4H23']
EXPECTED_CANDIDATE_ROWS = 400
EXPECTED_CANDIDATE_SHA256 = "2f2bfe05e1c21bb6a7b8c6dbe6c37fca8c42b9985ebeb74fcda710f5b629a68d"
EXPECTED_CANDIDATE_LOCK_SHA256 = "cb0ee919fd4a830d6d7e30f7acc91854470354684768ab9cf4762ec289f1fdfa"
EXPECTED_EXISTING_SCORES_SHA256 = "8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906"
EXPECTED_EXISTING_SCORING_LOCK_SHA256 = "ac7a96c89eb2d4a51eda1b56756ed3b943311e7f300b661659cff51285423a8f"
EXPECTED_V4H_FACET_SHA256 = "dbdb7c5d5603e17164cf6ef880df912f6196663387885e1f5773e54b97d2b106"
EXPECTED_V4H_AUTHORING_LOCK_SHA256 = "14cd3a8a9f6bf22666dd4c74a150e85f82a53789ccaa5bd2870d606b66ef83e7"
EXPECTED_V3_SCORING_LOCK_SHA256 = "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777"
EXPECTED_SPAN_ECHO_PATCH_SHA256 = "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266"
EXPECTED_CONTRACT_SHA256 = "75810d2451809943a1e5d586029aa27ab42f87113a028273d5ee582496cf9422"

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
    specs = {}
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

def deterministic_case_id(query_id: str, isbn13: str) -> str:
    return f"V4H_HEADROOM_A_{query_id}_{canonical_isbn(isbn13)}"

def main() -> None:
    print("Semantic relevance V4H - headroom Stage-A judging preparation")
    print("--------------------------------------------------------------")
    print("Role: consumed-holdout diagnostic only")
    print("Judge execution: DISABLED")
    print("Exact V4H score reuse: ENABLED for same query + same ISBN only")
    print("Cross-query/approximate reuse: DISABLED")
    print("Release decision modification: DISABLED")
    print()

    for path in [INVENTORY, JUDGING_INPUT, JUDGING_INPUT_LOCK]:
        req(not path.exists(), f"Refusing to overwrite existing preparation artifact: {path}")

    verify_raw(CANDIDATES, EXPECTED_CANDIDATE_SHA256, "Stage-A frozen candidate inventory")
    verify_raw(CANDIDATE_LOCK, EXPECTED_CANDIDATE_LOCK_SHA256, "Stage-A candidate lock")
    verify_raw(EXISTING_SCORES, EXPECTED_EXISTING_SCORES_SHA256, "existing frozen V4H judge scores")
    verify_raw(EXISTING_SCORING_LOCK, EXPECTED_EXISTING_SCORING_LOCK_SHA256,
               "existing V4H judge scoring lock")
    verify_raw(JUDGING_CONTRACT, EXPECTED_CONTRACT_SHA256, "Stage-A judging-preparation contract")

    clock = load_json(CANDIDATE_LOCK)
    req(clock.get("status") == "V4H_NO_CLEAR_HEADROOM_STAGE_A_CANDIDATES_FROZEN_BEFORE_JUDGING",
        "Unexpected Stage-A candidate-lock status")
    req(int(clock.get("candidate_count", -1)) == EXPECTED_CANDIDATE_ROWS,
        "Stage-A candidate count changed")
    req(clock.get("candidate_sha256") == EXPECTED_CANDIDATE_SHA256,
        "Candidate lock does not bind expected candidate artifact")
    req(clock.get("release_decision_modified") is False,
        "Candidate lock says release decision was modified")

    slock = load_json(EXISTING_SCORING_LOCK)
    req(slock.get("status") == "V4H_JUDGE_SCORES_FROZEN_BEFORE_RELEASE_ANALYSIS",
        "Unexpected V4H score-lock status")
    req(slock.get("judge_scores_sha256") == EXPECTED_EXISTING_SCORES_SHA256,
        "Existing score lock does not bind expected score artifact")
    req(slock.get("human_relevance_labels_used") is False,
        "Existing score lock says human labels were used")

    specs = verify_measurement_instrument()

    candidates = pd.read_csv(CANDIDATES, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id","query","stratum","v4b_rank","isbn13","title","description"
    }
    missing = required - set(candidates.columns)
    req(not missing, f"Candidate inventory missing columns: {sorted(missing)}")
    req(len(candidates) == EXPECTED_CANDIDATE_ROWS, "Candidate row count changed")
    candidates["isbn13"] = candidates["isbn13"].map(canonical_isbn)
    candidates["v4b_rank_num"] = pd.to_numeric(candidates["v4b_rank"], errors="raise").astype(int)
    req(not candidates.duplicated(["query_id","isbn13"]).any(), "Duplicate Stage-A query-book pair")
    req(sorted(candidates["query_id"].unique()) == TARGET_QUERY_IDS, "Target query set changed")
    req(candidates["v4b_rank_num"].between(11,50).all(), "Candidate outside ranks 11-50")
    counts = candidates.groupby("query_id").size()
    req((counts == 40).all(), f"Expected 40 candidates/query: {counts.to_dict()}")
    req(candidates["description"].str.strip().ne("").all(), "Candidate inventory has empty description")

    for qid, g in candidates.groupby("query_id", sort=True):
        req(g["query"].nunique() == 1, f"{qid} has multiple query texts")
        req(str(g["query"].iloc[0]) == specs[qid].query,
            f"{qid} query text disagrees with frozen facet specification")

    scores = pd.read_csv(EXISTING_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    req("judge_score" in scores.columns, "Existing score artifact missing judge_score")
    scores["isbn13"] = scores["isbn13"].map(canonical_isbn)
    req(not scores.duplicated(["query_id","isbn13"]).any(), "Existing V4H score artifact has duplicate pair")
    score_lookup = scores.set_index(["query_id","isbn13"])["judge_score"].to_dict()

    inventory = candidates.copy()
    inventory.insert(
        0, "case_id",
        [deterministic_case_id(q,i) for q,i in zip(inventory["query_id"], inventory["isbn13"])]
    )
    inventory["score_source"] = [
        "reused_frozen_v4h_display_score" if (q,i) in score_lookup else "fresh_judge_required"
        for q,i in zip(inventory["query_id"], inventory["isbn13"])
    ]
    inventory["reused_judge_score"] = [
        score_lookup.get((q,i), "")
        for q,i in zip(inventory["query_id"], inventory["isbn13"])
    ]
    req(not inventory["case_id"].duplicated().any(), "Duplicate Stage-A deterministic case ID")
    inventory = inventory.sort_values(["query_id","v4b_rank_num"], kind="stable").reset_index(drop=True)
    inventory.to_csv(INVENTORY, index=False, encoding="utf-8")

    fresh = inventory[inventory["score_source"] == "fresh_judge_required"].copy()
    reused = inventory[inventory["score_source"] == "reused_frozen_v4h_display_score"].copy()
    req(len(fresh) + len(reused) == EXPECTED_CANDIDATE_ROWS, "Inventory partition mismatch")

    drop_cols = ["score_source","reused_judge_score","v4b_rank_num"]
    fresh.drop(columns=[c for c in drop_cols if c in fresh.columns]).to_csv(
        JUDGING_INPUT, index=False, encoding="utf-8"
    )

    lock = {
        "status": "V4H_HEADROOM_STAGE_A_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": "post_release_consumed_holdout_diagnostic",
        "target_query_ids": TARGET_QUERY_IDS,
        "candidate_count": EXPECTED_CANDIDATE_ROWS,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "candidate_lock_sha256": EXPECTED_CANDIDATE_LOCK_SHA256,
        "existing_v4h_judge_scores_sha256": EXPECTED_EXISTING_SCORES_SHA256,
        "existing_v4h_judge_scoring_lock_sha256": EXPECTED_EXISTING_SCORING_LOCK_SHA256,
        "judging_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "score_inventory_sha256": sha256(INVENTORY),
        "fresh_judging_input_sha256": sha256(JUDGING_INPUT),
        "exact_score_reuse_key": ["query_id","isbn13"],
        "reused_exact_pair_count": len(reused),
        "fresh_pair_count": len(fresh),
        "total_stage_a_pair_count": EXPECTED_CANDIDATE_ROWS,
        "judge_candidate": "semantic_relevance_v0.29.0-r5",
        "v4h_facet_spec_sha256": EXPECTED_V4H_FACET_SHA256,
        "historical_score_reuse_cross_query_allowed": False,
        "approximate_score_reuse_allowed": False,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "release_metrics_calculated": False,
        "release_decision_modified": False,
        "case_id_rule": "V4H_HEADROOM_A_<query_id>_<canonical_isbn13>",
        "next_stage": "EXECUTE_FROZEN_JUDGE_ON_FRESH_STAGE_A_PAIRS_ONLY",
    }
    JUDGING_INPUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  Stage-A score inventory frozen: {len(inventory)} pairs")
    print(f"PASS  exact frozen V4H score reuse: {len(reused)} pairs")
    print(f"PASS  fresh judge input frozen: {len(fresh)} pairs")
    print("PASS  cross-query and approximate score reuse disabled")
    print("PASS  no human relevance labels present")
    print()
    print("V4H HEADROOM STAGE-A JUDGING PREPARATION: PASS")
    print(f"Score inventory SHA-256: {sha256(INVENTORY)}")
    print(f"Fresh judging input SHA-256: {sha256(JUDGING_INPUT)}")
    print(f"Judging-input lock SHA-256: {sha256(JUDGING_INPUT_LOCK)}")
    print("Judge calls performed: 0")
    print("Frozen V4B release decision modified: NO")
    print("Next: smoke-test the frozen judge on a small number of fresh Stage-A pairs.")

if __name__ == "__main__":
    main()
