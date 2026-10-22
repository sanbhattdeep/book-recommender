from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYSTEM = EVALS / "system_evaluation"

V4A_RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
V3_POSTHOC_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"
V3_CLASS_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v3_classification"

SPEC = SYSTEM / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json"
HYBRID_UNION = V4A_RUN / "v4a_dev_hybrid_union.csv"
RETRIEVAL_LOCK = V4A_RUN / "v4a_dev_retrieval.lock.json"
V4A_COMPLETE_DISPLAY_MAP = V4A_RUN / "v4a_dev_complete_display_score_map.csv"
V4A_FINALIZATION_LOCK = V4A_RUN / "v4a_dev_complete_score_map.lock.json"
V4A_COMPARISON_LOCK = V4A_RUN / "v4a_dev_comparison_selection.lock.json"

TARGETS = V3_POSTHOC_RUN / "target_queries.csv"
V3_POSTHOC_INPUT_LOCK = V3_POSTHOC_RUN / "deep_candidate_input_lock.json"
V3_POSTHOC_COMPLETE_MAP = V3_POSTHOC_RUN / "target_top50_complete_score_map.csv"
V3_POSTHOC_SCORING_LOCK = V3_POSTHOC_RUN / "deep_candidate_scoring_lock.json"
V3_CLASSIFICATION_LOCK = V3_CLASS_RUN / "classification_lock.json"

OUT_INVENTORY = V4A_RUN / "v4a_retrieval_diagnostic_score_inventory.csv"
OUT_NOVEL = V4A_RUN / "v4a_retrieval_diagnostic_novel_bm25_pairs_to_judge.csv"
OUT_METADATA = V4A_RUN / "v4a_retrieval_diagnostic_input_metadata.json"
OUT_LOCK = V4A_RUN / "v4a_retrieval_diagnostic_input.lock.json"

EXPECTED = {
    "spec": "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9",
    "hybrid_union": "71be13c297f6449e6a53c54c7b9db590f6f033d69a11f16b21377797d9390155",
    "retrieval_lock": "7587aa140c1cb26f8ac5addaf6873f3bc7ec05b6ec34775a6cbfe98d7ffa40d1",
    "v3_posthoc_input_lock": "ca7340bfc21db93a8b70cdafcf591d6b13a8a31ade55f2de1c954358195af832",
    "v3_posthoc_complete_map": "723fa27ecb8947334d25b3b7423a6597da39d594eb7ec2f979056b1c3cc516e5",
    "v3_posthoc_scoring_lock": "af67cc8090d0613973da4c927b573d29bb956be34d4c44255594cb12bde6fd1f",
    "v3_classification_lock": "cbaaaa1795773e46867ee5c135822b4b559c7805eee8e4264609dd4c9a913385",
    "v4a_complete_display_map": "60fb8b44f6eb13d486eaa6ffae483dca374e1415af8708d833912a18ec4d2000",
    "v4a_finalization_lock": "fe5b4865df05c4086247360fc7693e71d3cfe63f8877e944eaea61016d418f23",
    "v4a_comparison_lock": "c9a33feabf061e9bec373cbf5a5bdfb8b83ac03fd700f9fa55ffdfaa77e5056f",
}

PAYLOAD_COLS = [
    "judge_score", "match_level", "judge_reason", "scoring_explanation",
    "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_lf_sha256(path: Path) -> str:
    # Git stores tracked text with LF, while Windows may materialize CRLF in the
    # working tree. Normalize only line endings; do not change any other bytes.
    raw = path.read_bytes()
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def verify_frozen_text(path: Path, expected: str, label: str) -> str:
    raw = sha256(path)
    if raw == expected:
        mode = "raw_bytes"
    else:
        canonical = canonical_lf_sha256(path)
        require(
            canonical == expected,
            f"{label} hash mismatch: expected={expected} raw={raw} canonical_lf={canonical}",
        )
        mode = "canonical_lf_line_endings"
    print(f"PASS  {label} {mode} SHA-256 verified")
    return mode


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    require(bool(text), "ISBN is empty")
    return text


def verify_frozen_inputs() -> list[str]:
    require(SPEC.exists(), f"Missing frozen input: {SPEC}")
    verify_frozen_text(SPEC, EXPECTED["spec"], "v4a spec")

    checks = [
        ("v4a hybrid union", HYBRID_UNION, EXPECTED["hybrid_union"]),
        ("v4a retrieval lock", RETRIEVAL_LOCK, EXPECTED["retrieval_lock"]),
        ("v3 posthoc input lock", V3_POSTHOC_INPUT_LOCK, EXPECTED["v3_posthoc_input_lock"]),
        ("v3 posthoc complete top50 map", V3_POSTHOC_COMPLETE_MAP, EXPECTED["v3_posthoc_complete_map"]),
        ("v3 posthoc scoring lock", V3_POSTHOC_SCORING_LOCK, EXPECTED["v3_posthoc_scoring_lock"]),
        ("v3 posthoc classification lock", V3_CLASSIFICATION_LOCK, EXPECTED["v3_classification_lock"]),
        ("v4a complete displayed score map", V4A_COMPLETE_DISPLAY_MAP, EXPECTED["v4a_complete_display_map"]),
        ("v4a finalization lock", V4A_FINALIZATION_LOCK, EXPECTED["v4a_finalization_lock"]),
        ("v4a comparison/selection lock", V4A_COMPARISON_LOCK, EXPECTED["v4a_comparison_lock"]),
    ]
    for label, path, expected in checks:
        require(path.exists(), f"Missing frozen input: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  {label} SHA-256 verified")

    spec = load_json(SPEC)
    require(spec.get("status") == "FROZEN_BEFORE_V4A_IMPLEMENTATION", "Unexpected v4a spec status")
    protocol = spec.get("development_protocol", {})
    require(protocol.get("retrieval_diagnostic_queries") == "17 Stage-B target queries", "Retrieval diagnostic query contract changed")
    require("judge only previously-unjudged BM25-only candidates" in str(protocol.get("retrieval_diagnostic_scoring", "")), "Retrieval diagnostic scoring contract changed")
    selection = spec.get("development_selection_rule", {})
    require(selection.get("retrieval_diagnostics_can_explain_but_not_override_selection") is True, "Diagnostic authority boundary changed")

    retrieval_lock = load_json(RETRIEVAL_LOCK)
    require(retrieval_lock.get("status") == "V4A_DEVELOPMENT_RETRIEVAL_FROZEN_BEFORE_RERANKING_OR_JUDGING", "Unexpected v4a retrieval lock status")
    require(retrieval_lock.get("hybrid_union_sha256") == EXPECTED["hybrid_union"], "Retrieval lock disagrees with hybrid union hash")
    require(retrieval_lock.get("hybrid_union_truncated_before_reranker") is False, "Hybrid union was unexpectedly truncated")

    input_lock = load_json(V3_POSTHOC_INPUT_LOCK)
    require(input_lock.get("status") == "V3_POSTHOC_DEEP_DIAGNOSTIC_INPUT_FROZEN_BEFORE_JUDGING", "Unexpected v3 posthoc input-lock status")
    target_sha = input_lock.get("outputs", {}).get("target_queries.csv")
    require(isinstance(target_sha, str) and len(target_sha) == 64, "v3 posthoc input lock lacks target_queries.csv hash")
    require(TARGETS.exists(), f"Missing frozen target-query file: {TARGETS}")
    require(sha256(TARGETS) == target_sha, "Frozen target_queries.csv hash mismatch")
    print("PASS  frozen 17-query target set SHA-256 verified")

    posthoc_scoring = load_json(V3_POSTHOC_SCORING_LOCK)
    require(posthoc_scoring.get("status") == "V3_POSTHOC_DEEP_JUDGE_SCORES_FROZEN_BEFORE_CLASSIFICATION", "Unexpected v3 posthoc scoring-lock status")
    require(posthoc_scoring.get("target_top50_complete_score_map_sha256") == EXPECTED["v3_posthoc_complete_map"], "Posthoc scoring lock disagrees with complete-map hash")

    class_lock = load_json(V3_CLASSIFICATION_LOCK)
    require(class_lock.get("status") == "V3_POSTHOC_RETRIEVAL_VS_RERANKER_CLASSIFICATION_FROZEN", "Unexpected posthoc classification-lock status")
    require(int(class_lock.get("target_query_count", -1)) == 17, "Posthoc classification target count changed")

    final_lock = load_json(V4A_FINALIZATION_LOCK)
    require(final_lock.get("status") == "V4A_DEV_COMPLETE_DISPLAY_SCORE_MAP_FROZEN_BEFORE_COMPARISON", "Unexpected v4a finalization-lock status")
    require(final_lock.get("complete_score_map_sha256") == EXPECTED["v4a_complete_display_map"], "Finalization lock disagrees with displayed score map")

    comparison_lock = load_json(V4A_COMPARISON_LOCK)
    require(comparison_lock.get("status") == "V4A_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN", "Unexpected v4a comparison lock status")
    require(comparison_lock.get("selected_candidate") == "v3b", "Frozen v4a development selection is no longer v3b")
    require(comparison_lock.get("requires_new_unseen_v4_holdout") is False, "Unexpected V4 holdout requirement")
    require(comparison_lock.get("release_decision_produced") is False, "V4A development comparison produced a release decision")

    targets = pd.read_csv(TARGETS, dtype=str, keep_default_na=False, encoding="utf-8")
    require(len(targets) == 17 and targets["query_id"].nunique() == 17, "Frozen target set is not exactly 17 unique queries")
    return sorted(targets["query_id"].astype(str).tolist())


def normalize_v3_posthoc_scores() -> pd.DataFrame:
    frame = pd.read_csv(V3_POSTHOC_COMPLETE_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "isbn13", "judge_score", "final_match_level", "final_judge_reason",
        "final_scoring_explanation", "final_facet_assessments_json", "final_facet_evidence_json",
        "final_component_evidence_ledger_json",
    }
    missing = required - set(frame.columns)
    require(not missing, f"Posthoc complete map missing columns: {sorted(missing)}")
    require(len(frame) == 850, f"Expected 850 posthoc rows, got {len(frame)}")
    out = frame[list(required)].copy().rename(columns={
        "final_match_level": "match_level",
        "final_judge_reason": "judge_reason",
        "final_scoring_explanation": "scoring_explanation",
        "final_facet_assessments_json": "facet_assessments_json",
        "final_facet_evidence_json": "facet_evidence_json",
        "final_component_evidence_ledger_json": "component_evidence_ledger_json",
    })
    out["query_id"] = out["query_id"].astype(str)
    out["isbn13"] = out["isbn13"].map(canonical_isbn)
    out["judge_score"] = pd.to_numeric(out["judge_score"], errors="raise").astype(int)
    require(out["judge_score"].between(0, 4).all(), "Posthoc score outside 0..4")
    require(not out.duplicated(["query_id", "isbn13"]).any(), "Duplicate posthoc score pair")
    out["score_source"] = "v3_posthoc_deep_frozen"
    return out


def normalize_v4a_display_scores() -> pd.DataFrame:
    frame = pd.read_csv(V4A_COMPLETE_DISPLAY_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "isbn13", "final_judge_score", "final_match_level", "final_judge_reason",
        "final_scoring_explanation", "final_facet_assessments_json", "final_facet_evidence_json",
        "final_component_evidence_ledger_json", "final_score_source",
    }
    missing = required - set(frame.columns)
    require(not missing, f"V4A displayed complete map missing columns: {sorted(missing)}")
    require(len(frame) == 347, f"Expected 347 v4a displayed-map rows, got {len(frame)}")
    out = frame[list(required)].copy().rename(columns={
        "final_judge_score": "judge_score",
        "final_match_level": "match_level",
        "final_judge_reason": "judge_reason",
        "final_scoring_explanation": "scoring_explanation",
        "final_facet_assessments_json": "facet_assessments_json",
        "final_facet_evidence_json": "facet_evidence_json",
        "final_component_evidence_ledger_json": "component_evidence_ledger_json",
        "final_score_source": "score_source",
    })
    out["query_id"] = out["query_id"].astype(str)
    out["isbn13"] = out["isbn13"].map(canonical_isbn)
    out["judge_score"] = pd.to_numeric(out["judge_score"], errors="raise").astype(int)
    require(out["judge_score"].between(0, 4).all(), "V4A displayed score outside 0..4")
    require(not out.duplicated(["query_id", "isbn13"]).any(), "Duplicate v4a displayed score pair")
    return out


def build_prior_score_map() -> tuple[pd.DataFrame, int]:
    posthoc = normalize_v3_posthoc_scores()
    display = normalize_v4a_display_scores()
    combined = pd.concat([posthoc, display], ignore_index=True)
    rows: list[dict[str, Any]] = []
    conflicts = 0
    for (qid, isbn), group in combined.groupby(["query_id", "isbn13"], sort=False):
        first = group.iloc[0]
        for _, other in group.iloc[1:].iterrows():
            if any(str(first[c]) != str(other[c]) for c in PAYLOAD_COLS):
                conflicts += 1
                raise ValueError(f"Frozen score conflict for {qid}/{isbn}")
        # Prefer V4A map for source provenance when it contains a newly judged BM25-only pair.
        novel = group[group["score_source"] == "v4a_dev_novel_judged"]
        selected = novel.iloc[0] if not novel.empty else first
        rows.append(selected.to_dict())
    out = pd.DataFrame(rows)
    require(not out.duplicated(["query_id", "isbn13"]).any(), "Deduplicated prior-score map still has duplicates")
    return out, conflicts


def build_inventory(target_qids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    union = pd.read_csv(HYBRID_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "query", "stratum", "isbn13", "dense_rank", "bm25_rank", "bm25_score",
        "candidate_source", "hybrid_retrieval_rrf_score", "hybrid_retrieval_rank", "title", "description", "authors",
    }
    missing = required - set(union.columns)
    require(not missing, f"Hybrid union missing columns: {sorted(missing)}")
    require(len(union) == 2211, f"Expected 2211 full-union rows, got {len(union)}")
    union["query_id"] = union["query_id"].astype(str)
    union["isbn13"] = union["isbn13"].map(canonical_isbn)
    require(not union.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in hybrid union")

    diag = union[union["query_id"].isin(target_qids)].copy()
    require(diag["query_id"].nunique() == 17, "Diagnostic union does not contain all 17 target queries")
    require(set(diag["query_id"]) == set(target_qids), "Diagnostic query IDs differ from frozen target set")

    dense_present = diag["dense_rank"].astype(str).str.strip().ne("")
    bm25_present = diag["bm25_rank"].astype(str).str.strip().ne("")
    counts = diag.loc[dense_present].groupby("query_id").size()
    require((counts == 50).all(), f"Each target query must contain exactly dense50; got {counts.to_dict()}")
    require(diag.loc[~dense_present, "candidate_source"].eq("bm25_only").all(), "A non-dense diagnostic candidate is not bm25_only")
    both_source = diag["candidate_source"].eq("both")
    require((dense_present & bm25_present).loc[both_source].all(), "candidate_source=both disagrees with ranks")

    prior, conflicts = build_prior_score_map()
    merged = diag.merge(prior, on=["query_id", "isbn13"], how="left", validate="one_to_one")
    merged["score_status"] = merged["judge_score"].apply(lambda x: "reused" if pd.notna(x) else "novel")

    reused = merged["score_status"].eq("reused")
    novel = ~reused
    dense_present_merged = merged["dense_rank"].astype(str).str.strip().ne("")
    require(merged.loc[dense_present_merged, "score_status"].eq("reused").all(), "A dense-top50 diagnostic pair lacks a frozen reusable score")
    require(merged.loc[novel, "candidate_source"].eq("bm25_only").all(), "Novel diagnostic judging input contains a non-BM25-only pair")
    require(merged.loc[novel, "dense_rank"].astype(str).str.strip().eq("").all(), "Novel diagnostic pair unexpectedly has dense rank")

    merged["judge_score"] = merged["judge_score"].astype("object")
    merged.loc[reused, "judge_score"] = pd.to_numeric(merged.loc[reused, "judge_score"], errors="raise").astype(int)
    merged.loc[novel, "judge_score"] = ""
    for col in ["score_source"] + PAYLOAD_COLS[1:]:
        merged.loc[novel, col] = ""

    inventory_cols = [
        "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "candidate_source", "dense_rank", "bm25_rank", "bm25_score", "hybrid_retrieval_rrf_score", "hybrid_retrieval_rank",
        "score_status", "score_source", "judge_score", "match_level", "judge_reason", "scoring_explanation",
        "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json",
    ]
    inventory = merged[inventory_cols].copy()
    inventory["_rank"] = pd.to_numeric(inventory["hybrid_retrieval_rank"], errors="raise").astype(int)
    inventory = inventory.sort_values(["query_id", "_rank", "isbn13"], kind="stable").drop(columns=["_rank"]).reset_index(drop=True)

    novel_pairs = inventory[inventory["score_status"] == "novel"].copy()
    novel_pairs.insert(0, "case_id", novel_pairs.apply(lambda r: f"V4A_RETRDIAG_{r['query_id']}_{r['isbn13']}", axis=1))
    novel_cols = [
        "case_id", "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "candidate_source", "bm25_rank", "bm25_score", "hybrid_retrieval_rank",
    ]
    novel_pairs = novel_pairs[novel_cols].copy()
    novel_pairs["_bm25"] = pd.to_numeric(novel_pairs["bm25_rank"], errors="raise").astype(int)
    novel_pairs = novel_pairs.sort_values(["query_id", "_bm25", "isbn13"], kind="stable").drop(columns=["_bm25"]).reset_index(drop=True)
    require(not novel_pairs["case_id"].duplicated().any(), "Novel diagnostic case IDs are not unique")

    source_counts = inventory["candidate_source"].value_counts().to_dict()
    reused_sources = inventory.loc[reused, "score_source"].value_counts().to_dict()
    stats = {
        "diagnostic_union_rows": int(len(inventory)),
        "target_queries": 17,
        "dense_branch_rows": int(dense_present.sum()),
        "bm25_only_rows": int(inventory["candidate_source"].eq("bm25_only").sum()),
        "both_rows": int(inventory["candidate_source"].eq("both").sum()),
        "reused_scores": int(reused.sum()),
        "novel_bm25_only_pairs_to_judge": int(novel.sum()),
        "candidate_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "reused_score_source_counts": {str(k): int(v) for k, v in reused_sources.items()},
        "frozen_score_conflicts": int(conflicts),
    }
    return inventory, novel_pairs, stats


def main() -> None:
    print("Semantic relevance v4a - prepare 17-query retrieval diagnostic judging input")
    print("-----------------------------------------------------------------------")
    print("Judge execution:            DISABLED")
    print("Model execution:            DISABLED")
    print("Development selection:      FROZEN / CANNOT BE OVERRIDDEN")
    print("Release metrics/decision:   DISABLED")
    print()

    for path in [OUT_INVENTORY, OUT_NOVEL, OUT_METADATA, OUT_LOCK]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing retrieval-diagnostic artifact: {path}")

    target_qids = verify_frozen_inputs()
    inventory, novel_pairs, stats = build_inventory(target_qids)

    OUT_INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT_INVENTORY, index=False, encoding="utf-8")
    novel_pairs.to_csv(OUT_NOVEL, index=False, encoding="utf-8")

    metadata = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_INPUT_PREPARATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_role": "explanatory v4-development evidence only; cannot override frozen v3b-over-v4a selection",
        "target_query_set": "17 frozen v3 posthoc Stage-B target queries",
        "frozen_inputs": EXPECTED,
        **stats,
        "judge_calls_performed": 0,
        "model_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_selection_reopened": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "outputs": {
            OUT_INVENTORY.name: sha256(OUT_INVENTORY),
            OUT_NOVEL.name: sha256(OUT_NOVEL),
        },
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_INPUT_FROZEN_BEFORE_JUDGING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_role": "explanatory_only",
        "target_query_count": 17,
        "hybrid_union_sha256": EXPECTED["hybrid_union"],
        "v3_posthoc_complete_map_sha256": EXPECTED["v3_posthoc_complete_map"],
        "v4a_complete_display_map_sha256": EXPECTED["v4a_complete_display_map"],
        "v4a_comparison_selection_lock_sha256": EXPECTED["v4a_comparison_lock"],
        **stats,
        "score_inventory_sha256": sha256(OUT_INVENTORY),
        "novel_bm25_judging_input_sha256": sha256(OUT_NOVEL),
        "metadata_sha256": sha256(OUT_METADATA),
        "judge_execution_performed": False,
        "model_execution_performed": False,
        "human_relevance_labels_used": False,
        "development_selection_reopened": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "JUDGE_ONLY_FROZEN_NOVEL_BM25_ONLY_RETRIEVAL_DIAGNOSTIC_PAIRS",
    }
    OUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  frozen 17-query hybrid-union diagnostic inventory: {stats['diagnostic_union_rows']} pairs")
    print(f"INFO  dense-branch pairs with reusable scores: {stats['dense_branch_rows']}")
    print(f"INFO  total BM25-only pairs: {stats['bm25_only_rows']}")
    print(f"INFO  reused frozen scores: {stats['reused_scores']}")
    print(f"INFO  novel BM25-only pairs to judge: {stats['novel_bm25_only_pairs_to_judge']}")
    print(f"INFO  reused sources: {stats['reused_score_source_counts']}")
    print("PASS  retrieval-diagnostic judging input frozen before any new judge execution")
    print()
    print("V4A 17-QUERY RETRIEVAL DIAGNOSTIC INPUT PREPARATION: PASS")
    print(f"Score inventory SHA-256: {sha256(OUT_INVENTORY)}")
    print(f"Novel BM25 judging input SHA-256: {sha256(OUT_NOVEL)}")
    print(f"Diagnostic input lock SHA-256: {sha256(OUT_LOCK)}")
    print("Judge calls performed: 0")
    print("Model calls performed: 0")
    print("Development selection reopened: NO")
    print("Release decision produced: NO")
    print("Next: prepare judge execution package for only the frozen novel BM25-only diagnostic pairs.")


if __name__ == "__main__":
    main()
