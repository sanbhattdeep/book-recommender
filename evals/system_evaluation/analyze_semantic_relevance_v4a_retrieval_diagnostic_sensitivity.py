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
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
V3_CLASS_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v3_classification"

CONTRACT = SYSTEM / "semantic_relevance_v4a_retrieval_diagnostic_sensitivity_contract.v0.1.0.json"
SENSITIVITY_INPUT_LOCK = RUN / "v4a_retrieval_diagnostic_sensitivity_input.lock.json"
INVENTORY = RUN / "v4a_retrieval_diagnostic_score_inventory.csv"
PARTIAL_SCORES = RUN / "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv"
QUARANTINE = RUN / "v4a_retrieval_diagnostic_quarantined_cases.json"
PARTIAL_FREEZE_LOCK = RUN / "v4a_retrieval_diagnostic_partial_judging.freeze.lock.json"
V3_CLASS_LOCK = V3_CLASS_RUN / "classification_lock.json"
V3_CLASSIFICATION = V3_CLASS_RUN / "query_retrieval_reranker_classification.csv"
V4A_COMPARISON_LOCK = RUN / "v4a_dev_comparison_selection.lock.json"
FINAL_SCORING_LOCK = RUN / "v4a_retrieval_diagnostic_novel_judge_scoring.lock.json"

OUT_QUERY = RUN / "v4a_retrieval_diagnostic_sensitivity_query_results.csv"
OUT_SUMMARY = RUN / "v4a_retrieval_diagnostic_sensitivity_scenario_summary.csv"
OUT_INVARIANCE = RUN / "v4a_retrieval_diagnostic_sensitivity_invariance.json"
OUT_REPORT = RUN / "v4a_retrieval_diagnostic_sensitivity_report.md"
OUT_METADATA = RUN / "v4a_retrieval_diagnostic_sensitivity_metadata.json"
OUT_LOCK = RUN / "v4a_retrieval_diagnostic_sensitivity_analysis.lock.json"

EXPECTED = {
    "contract": "96b0a1a8a4365b8cdd75018235964dbd46f018cd8dfd086e06fbb8d8b02c9f90",
    "sensitivity_input_lock": "c7aafd11e4620fe0e9031b6367c1137e28efa1372d3891c3be64558e01c15b04",
    "inventory": "250841f0bf9cefc5a461e09f3963842187bdac8bfdaafe03e7e07487d07c7e16",
    "partial_scores": "ceb643cbff823007beb1f698b9fd42a78af70ead73e1c8bc90de94d14764be47",
    "quarantine": "ce16b3da279d4435d3b2eea7cf1aa190e6e804ecb6e275e8c12001703cbb0a38",
    "partial_freeze_lock": "ae22f150f5f64f228f7970010103b1229c7b4c100f348f55faec172034319e55",
    "v3_class_lock": "cbaaaa1795773e46867ee5c135822b4b559c7805eee8e4264609dd4c9a913385",
    "v4a_comparison_lock": "c9a33feabf061e9bec373cbf5a5bdfb8b83ac03fd700f9fa55ffdfaa77e5056f"
}

KNOWN_CASE = "V4A_RETRDIAG_H14_9781582972947"
KNOWN_QID = "H14"
KNOWN_ISBN = "9781582972947"
SCENARIOS = {"S0": 0, "S1": 1, "S2": 2, "S3": 3, "S4": 4}
CLASSES = ["RETRIEVAL_LIMITED", "RERANKER_LIMITED", "MIXED"]
EPS = 1e-12

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical_lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def verify_raw(path: Path, expected: str, label: str) -> None:
    require(path.exists(), f"Missing frozen input: {path}")
    actual = sha256(path)
    require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")

def verify_text(path: Path, expected: str, label: str) -> None:
    require(path.exists(), f"Missing frozen input: {path}")
    raw = sha256(path)
    if raw != expected:
        norm = canonical_lf_sha256(path)
        require(norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} canonical_lf={norm}")

def classify(oracle_mean: float, oracle_clear_rate: float, oracle_irrelevant_rate: float, pool_clear_count: int) -> tuple[str, str]:
    if oracle_mean < 1.5 - EPS or pool_clear_count == 0:
        reasons = []
        if oracle_mean < 1.5 - EPS:
            reasons.append("oracle_mean_relevance_at_10 < 1.5")
        if pool_clear_count == 0:
            reasons.append("hybrid_union_clear_or_strong_count == 0")
        return "RETRIEVAL_LIMITED", "; ".join(reasons)
    if oracle_mean >= 2.5 - EPS and oracle_clear_rate >= 0.60 - EPS and oracle_irrelevant_rate <= 0.10 + EPS:
        return "RERANKER_LIMITED", "hybrid-union oracle top10 meets frozen core headroom criteria"
    return "MIXED", "hybrid union has some headroom but does not meet either frozen extreme classification"

def build_actual_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    partial = pd.read_csv(PARTIAL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")

    base = inventory.copy()
    base["final_score"] = pd.NA
    reused = base["score_status"].eq("reused")
    novel = base["score_status"].eq("novel")
    base.loc[reused, "final_score"] = pd.to_numeric(base.loc[reused, "judge_score"], errors="raise").astype(int)

    novel_scores = partial[["query_id", "isbn13", "judge_score"]].copy()
    novel_scores["judge_score"] = pd.to_numeric(novel_scores["judge_score"], errors="raise").astype(int)
    novel_scores = novel_scores.rename(columns={"judge_score": "persisted_novel_score"})
    base = base.merge(novel_scores, on=["query_id", "isbn13"], how="left", validate="one_to_one")
    base.loc[novel & base["persisted_novel_score"].notna(), "final_score"] = (
        pd.to_numeric(base.loc[novel & base["persisted_novel_score"].notna(), "persisted_novel_score"], errors="raise").astype(int)
    )

    missing = base[base["final_score"].isna()].copy()
    require(len(missing) == 1, f"Expected exactly one unresolved score after merge, got {len(missing)}")
    row = missing.iloc[0]
    require(str(row["query_id"]) == KNOWN_QID and str(row["isbn13"]) == KNOWN_ISBN, "The only unresolved pair is not the frozen quarantine")
    require(str(row["candidate_source"]) == "bm25_only", "Frozen quarantine is no longer BM25-only")

    base["hybrid_retrieval_rank_num"] = pd.to_numeric(base["hybrid_retrieval_rank"], errors="raise").astype(int)
    require(base["hybrid_retrieval_rank_num"].ge(1).all(), "Invalid hybrid retrieval rank")
    require(not base.duplicated(["query_id","isbn13"]).any(), "Duplicate pair in complete diagnostic base")

    class_lock = load_json(V3_CLASS_LOCK)
    cls_hash = class_lock["outputs"]["query_retrieval_reranker_classification.csv"]
    require(sha256(V3_CLASSIFICATION) == cls_hash, "V3 classification CSV no longer matches its lock")
    stage_c = pd.read_csv(V3_CLASSIFICATION, dtype=str, keep_default_na=False, encoding="utf-8")
    stage_c = stage_c[["query_id","posthoc_limitation_class"]].copy()
    require(len(stage_c) == 17 and not stage_c["query_id"].duplicated().any(), "Invalid Stage-C classification shape")
    return base, stage_c

def scenario_query_rows(base: pd.DataFrame, stage_c: pd.DataFrame, scenario_id: str, hypothetical_score: int) -> pd.DataFrame:
    scenario = base.copy()
    mask = scenario["query_id"].eq(KNOWN_QID) & scenario["isbn13"].astype(str).eq(KNOWN_ISBN)
    require(int(mask.sum()) == 1, "Quarantine pair not unique")
    scenario.loc[mask, "final_score"] = int(hypothetical_score)
    scores = pd.to_numeric(scenario["final_score"], errors="raise").astype(int)
    require(scores.between(0,4).all(), "Scenario contains score outside 0..4")
    scenario["final_score"] = scores

    rows: list[dict[str, Any]] = []
    stage_map = dict(zip(stage_c["query_id"].astype(str), stage_c["posthoc_limitation_class"].astype(str)))

    for qid, q in scenario.groupby("query_id", sort=True):
        q = q.copy()
        bm25_only = q[q["candidate_source"].eq("bm25_only")].copy()
        require(len(bm25_only) > 0, f"{qid} has no BM25-only candidates; required diagnostic undefined")

        bm25_clear_count = int((bm25_only["final_score"] >= 3).sum())
        best_bm25_only = int(bm25_only["final_score"].max())
        union_clear_count = int((q["final_score"] >= 3).sum())

        oracle = q.sort_values(
            ["final_score","hybrid_retrieval_rank_num","isbn13"],
            ascending=[False, True, True],
            kind="stable",
        ).head(10).copy()
        require(len(oracle) == 10, f"{qid} oracle top10 has {len(oracle)} rows")
        oracle_mean = float(oracle["final_score"].mean())
        oracle_clear = float((oracle["final_score"] >= 3).mean())
        oracle_irrelevant = float((oracle["final_score"] == 0).mean())

        hclass, reason = classify(oracle_mean, oracle_clear, oracle_irrelevant, union_clear_count)
        sclass = stage_map[str(qid)]
        rows.append({
            "scenario_id": scenario_id,
            "hypothetical_quarantine_score": int(hypothetical_score),
            "query_id": str(qid),
            "query": str(q.iloc[0]["query"]),
            "stratum": str(q.iloc[0]["stratum"]),
            "bm25_only_candidate_count": int(len(bm25_only)),
            "bm25_only_clear_or_strong_count": bm25_clear_count,
            "has_bm25_only_clear_or_strong_candidate": bool(bm25_clear_count > 0),
            "best_bm25_only_score": best_bm25_only,
            "hybrid_union_candidate_count": int(len(q)),
            "hybrid_union_clear_or_strong_count": union_clear_count,
            "hybrid_union_oracle_mean_at_10": oracle_mean,
            "hybrid_union_oracle_clear_or_strong_rate_at_10": oracle_clear,
            "hybrid_union_oracle_irrelevant_rate_at_10": oracle_irrelevant,
            "stage_c_limitation_class": sclass,
            "hybrid_union_limitation_class": hclass,
            "stage_c_to_hybrid_limitation_class_transition": f"{sclass}->{hclass}",
            "hybrid_classification_reason": reason,
        })
    result = pd.DataFrame(rows)
    require(len(result) == 17, f"{scenario_id} produced {len(result)} query rows")
    return result

def main() -> None:
    print("Semantic relevance v4a - deterministic retrieval-diagnostic sensitivity analysis")
    print("----------------------------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Human labels: DISABLED")
    print("Actual quarantine score assignment: DISABLED")
    print("Development selection: FROZEN / NOT REOPENED")
    print("Release analysis: DISABLED")
    print()

    for out in [OUT_QUERY, OUT_SUMMARY, OUT_INVARIANCE, OUT_REPORT, OUT_METADATA, OUT_LOCK]:
        require(not out.exists(), f"Refusing to overwrite existing analysis output: {out}")
    require(not FINAL_SCORING_LOCK.exists(), "Final diagnostic judge scoring lock unexpectedly exists")

    verify_text(CONTRACT, EXPECTED["contract"], "sensitivity contract")
    for path, key, label in [
        (SENSITIVITY_INPUT_LOCK, "sensitivity_input_lock", "sensitivity input lock"),
        (INVENTORY, "inventory", "diagnostic score inventory"),
        (PARTIAL_SCORES, "partial_scores", "674-case partial score artifact"),
        (QUARANTINE, "quarantine", "single-case quarantine artifact"),
        (PARTIAL_FREEZE_LOCK, "partial_freeze_lock", "partial judging freeze lock"),
        (V3_CLASS_LOCK, "v3_class_lock", "v3 Stage-C classification lock"),
        (V4A_COMPARISON_LOCK, "v4a_comparison_lock", "v4a comparison/selection lock"),
    ]:
        verify_raw(path, EXPECTED[key], label)

    silock = load_json(SENSITIVITY_INPUT_LOCK)
    require(silock.get("status") == "V4A_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_CONTRACT_FROZEN_BEFORE_ANALYSIS", "Sensitivity input lock status changed")
    require(silock.get("hypothetical_scores_to_analyze") == [0,1,2,3,4], "Sensitivity scenario set changed")
    require(silock.get("quarantined_actual_score_assigned") is False, "Sensitivity input lock says actual quarantine score assigned")
    comparison = load_json(V4A_COMPARISON_LOCK)
    require(comparison.get("selected_candidate") == "v3b", "Frozen development selection changed from v3b")

    base, stage_c = build_actual_base()
    query_frames = [scenario_query_rows(base, stage_c, sid, score) for sid, score in SCENARIOS.items()]
    query_results = pd.concat(query_frames, ignore_index=True)
    require(len(query_results) == 85, f"Expected 85 query-scenario rows, got {len(query_results)}")

    summary_rows: list[dict[str, Any]] = []
    for sid, score in SCENARIOS.items():
        q = query_results[query_results["scenario_id"] == sid].copy()
        counts = q["hybrid_union_limitation_class"].value_counts().reindex(CLASSES, fill_value=0)
        summary_rows.append({
            "scenario_id": sid,
            "hypothetical_quarantine_score": score,
            "total_bm25_only_clear_or_strong_candidates": int(q["bm25_only_clear_or_strong_count"].sum()),
            "queries_gaining_bm25_only_clear_or_strong_candidate": int(q["has_bm25_only_clear_or_strong_candidate"].sum()),
            "hybrid_RETRIEVAL_LIMITED": int(counts["RETRIEVAL_LIMITED"]),
            "hybrid_RERANKER_LIMITED": int(counts["RERANKER_LIMITED"]),
            "hybrid_MIXED": int(counts["MIXED"]),
        })
    scenario_summary = pd.DataFrame(summary_rows)

    invariant_by_query = []
    for qid, q in query_results.groupby("query_id", sort=True):
        classes = sorted(set(q["hybrid_union_limitation_class"].astype(str)))
        transitions = sorted(set(q["stage_c_to_hybrid_limitation_class_transition"].astype(str)))
        invariant_by_query.append({
            "query_id": str(qid),
            "hybrid_classifications_observed": classes,
            "transitions_observed": transitions,
            "classification_and_transition_invariant": len(classes) == 1 and len(transitions) == 1,
            "oracle_mean_min": float(q["hybrid_union_oracle_mean_at_10"].min()),
            "oracle_mean_max": float(q["hybrid_union_oracle_mean_at_10"].max()),
            "oracle_clear_rate_min": float(q["hybrid_union_oracle_clear_or_strong_rate_at_10"].min()),
            "oracle_clear_rate_max": float(q["hybrid_union_oracle_clear_or_strong_rate_at_10"].max()),
            "oracle_irrelevant_rate_min": float(q["hybrid_union_oracle_irrelevant_rate_at_10"].min()),
            "oracle_irrelevant_rate_max": float(q["hybrid_union_oracle_irrelevant_rate_at_10"].max()),
            "bm25_only_clear_count_min": int(q["bm25_only_clear_or_strong_count"].min()),
            "bm25_only_clear_count_max": int(q["bm25_only_clear_or_strong_count"].max()),
            "best_bm25_only_score_min": int(q["best_bm25_only_score"].min()),
            "best_bm25_only_score_max": int(q["best_bm25_only_score"].max()),
        })

    all_invariant = all(x["classification_and_transition_invariant"] for x in invariant_by_query)
    affected = [x["query_id"] for x in invariant_by_query if not x["classification_and_transition_invariant"]]

    transition_matrix_by_scenario: dict[str, dict[str, int]] = {}
    for sid in SCENARIOS:
        q = query_results[query_results["scenario_id"] == sid]
        transition_matrix_by_scenario[sid] = {
            str(k): int(v)
            for k, v in q["stage_c_to_hybrid_limitation_class_transition"].value_counts().sort_index().to_dict().items()
        }

    invariance = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_ANALYSIS_COMPLETE",
        "actual_quarantine_score_known": False,
        "actual_quarantine_score_assigned": False,
        "quarantined_case_id": KNOWN_CASE,
        "scenarios_analyzed": SCENARIOS,
        "classification_and_transition_invariant_for_all_17_queries": bool(all_invariant),
        "queries_with_noninvariant_classification_or_transition": affected,
        "query_sensitivity": invariant_by_query,
        "aggregate_ranges": {
            "total_bm25_only_clear_or_strong_candidates": {
                "min": int(scenario_summary["total_bm25_only_clear_or_strong_candidates"].min()),
                "max": int(scenario_summary["total_bm25_only_clear_or_strong_candidates"].max()),
            },
            "queries_gaining_bm25_only_clear_or_strong_candidate": {
                "min": int(scenario_summary["queries_gaining_bm25_only_clear_or_strong_candidate"].min()),
                "max": int(scenario_summary["queries_gaining_bm25_only_clear_or_strong_candidate"].max()),
            },
        },
        "transition_matrix_by_scenario": transition_matrix_by_scenario,
        "development_selection_reopened": False,
        "release_analysis_performed": False,
    }

    OUT_QUERY.parent.mkdir(parents=True, exist_ok=True)
    query_results.to_csv(OUT_QUERY, index=False, encoding="utf-8")
    scenario_summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8")
    OUT_INVARIANCE.write_text(json.dumps(invariance, indent=2) + "\n", encoding="utf-8")

    h14 = query_results[query_results["query_id"] == KNOWN_QID].sort_values("hypothetical_quarantine_score")
    scenario_lines = "\n".join(
        f"| {r.scenario_id} | {int(r.hypothetical_quarantine_score)} | "
        f"{int(r.bm25_only_clear_or_strong_count)} | {int(r.best_bm25_only_score)} | "
        f"{r.hybrid_union_oracle_mean_at_10:.2f} | {r.hybrid_union_oracle_clear_or_strong_rate_at_10:.0%} | "
        f"{r.hybrid_union_oracle_irrelevant_rate_at_10:.0%} | {r.hybrid_union_limitation_class} |"
        for _, r in h14.iterrows()
    )
    summary_lines = "\n".join(
        f"| {r.scenario_id} | {int(r.hypothetical_quarantine_score)} | "
        f"{int(r.total_bm25_only_clear_or_strong_candidates)} | "
        f"{int(r.queries_gaining_bm25_only_clear_or_strong_candidate)} | "
        f"{int(r.hybrid_RETRIEVAL_LIMITED)} | {int(r.hybrid_RERANKER_LIMITED)} | {int(r.hybrid_MIXED)} |"
        for _, r in scenario_summary.iterrows()
    )

    if all_invariant:
        invariant_text = (
            "The Stage-C→hybrid limitation-class conclusion is invariant for all 17 queries "
            "under every allowed score 0–4 for the unresolved case."
        )
    else:
        invariant_text = (
            "The Stage-C→hybrid limitation-class conclusion is NOT invariant. "
            f"Affected queries: {', '.join(affected)}. Report the bounded alternatives; "
            "no preferred quarantine score is selected."
        )

    report = f"""# V4A Retrieval Diagnostic — Quarantine Sensitivity Analysis

## Authority boundary

This is **explanatory development evidence only**. The actual semantic score for
`{KNOWN_CASE}` remains **unknown and unassigned**. Scenarios S0–S4 are
counterfactual sensitivity cases only.

The frozen V4A development selection remains **v3b**. This analysis does not
recalculate release metrics or gates and cannot reopen that selection.

## Sensitivity conclusion

{invariant_text}

## H14 sensitivity

| Scenario | Hypothetical score | BM25-only clear count | Best BM25-only score | Hybrid oracle mean@10 | Oracle clear@10 | Oracle irrelevant@10 | Hybrid diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|
{scenario_lines}

## Aggregate diagnostic sensitivity

| Scenario | Hypothetical score | BM25-only clear candidates | Queries gaining BM25-only clear | Retrieval-limited | Reranker-limited | Mixed |
|---|---:|---:|---:|---:|---:|---:|
{summary_lines}

## Interpretation rule

No scenario is treated as the true score. Numeric diagnostics are reported as
bounded ranges where they change. Limitation-class transitions may be collapsed
to one conclusion only if they are identical in all five scenarios.

Judge calls: **0**  
Model calls: **0**  
Human relevance labels: **0**  
Development selection reopened: **NO**  
Release analysis: **NO**
"""
    OUT_REPORT.write_text(report, encoding="utf-8")

    metadata = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_ANALYSIS_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": EXPECTED["contract"],
        "sensitivity_input_lock_sha256": EXPECTED["sensitivity_input_lock"],
        "diagnostic_score_inventory_sha256": EXPECTED["inventory"],
        "partial_novel_score_sha256": EXPECTED["partial_scores"],
        "quarantine_sha256": EXPECTED["quarantine"],
        "v3_stage_c_classification_lock_sha256": EXPECTED["v3_class_lock"],
        "v4a_comparison_selection_lock_sha256": EXPECTED["v4a_comparison_lock"],
        "scenarios_analyzed": SCENARIOS,
        "query_scenario_rows": int(len(query_results)),
        "classification_and_transition_invariant_for_all_17_queries": bool(all_invariant),
        "queries_with_noninvariant_classification_or_transition": affected,
        "actual_quarantine_score_assigned": False,
        "judge_calls_performed": 0,
        "model_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_selection_reopened": False,
        "release_metrics_calculated": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    outputs = [OUT_QUERY, OUT_SUMMARY, OUT_INVARIANCE, OUT_REPORT, OUT_METADATA]
    lock = {
        "status": "V4A_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_ANALYSIS_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": EXPECTED["contract"],
        "sensitivity_input_lock_sha256": EXPECTED["sensitivity_input_lock"],
        "partial_judging_freeze_lock_sha256": EXPECTED["partial_freeze_lock"],
        "diagnostic_score_inventory_sha256": EXPECTED["inventory"],
        "partial_novel_score_sha256": EXPECTED["partial_scores"],
        "quarantine_sha256": EXPECTED["quarantine"],
        "v3_stage_c_classification_lock_sha256": EXPECTED["v3_class_lock"],
        "v4a_comparison_selection_lock_sha256": EXPECTED["v4a_comparison_lock"],
        "quarantined_case_id": KNOWN_CASE,
        "actual_quarantine_score_assigned": False,
        "scenarios_analyzed": SCENARIOS,
        "classification_and_transition_invariant_for_all_17_queries": bool(all_invariant),
        "queries_with_noninvariant_classification_or_transition": affected,
        "judge_calls_performed": 0,
        "model_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_selection_reopened": False,
        "release_metrics_calculated": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "outputs": {p.name: sha256(p) for p in outputs},
        "metadata_sha256": sha256(OUT_METADATA),
        "next_stage": "INTERPRET_FROZEN_V4A_RETRIEVAL_DIAGNOSTIC_AND_FINALIZE_V4B_DESIGN",
    }
    OUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print("PASS  deterministic five-scenario analysis completed with zero judge/model calls")
    print(f"INFO  classification/transition invariant across all 17 queries: {all_invariant}")
    if affected:
        print(f"INFO  non-invariant queries: {affected}")
    print()
    print("V4A RETRIEVAL DIAGNOSTIC SENSITIVITY ANALYSIS: PASS")
    print(f"Query-scenario rows: {len(query_results)}")
    print(f"Analysis lock SHA-256: {sha256(OUT_LOCK)}")
    print(f"Report: {OUT_REPORT}")
    print("Actual quarantine score assigned: NO")
    print("Judge calls performed: 0")
    print("Model calls performed: 0")
    print("Development selection reopened: NO")
    print("Release analysis: NO")
    print("Next: interpret the frozen retrieval diagnostic and finalize V4B design.")

if __name__ == "__main__":
    main()
