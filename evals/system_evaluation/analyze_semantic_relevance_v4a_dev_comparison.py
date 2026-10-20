from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

SPEC = REPO_ROOT / "evals/system_evaluation/semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json"
SPEC_LOCK = REPO_ROOT / "evals/system_evaluation/semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.lock.json"
COMPLETE_MAP = RUN / "v4a_dev_complete_display_score_map.csv"
FINALIZATION_LOCK = RUN / "v4a_dev_complete_score_map.lock.json"

V3_QUERY_METRICS = RUN / "v3b_dev_query_metrics.csv"
V4_QUERY_METRICS = RUN / "v4a_dev_query_metrics.csv"
V3_AGG = RUN / "v3b_dev_aggregate_metrics.json"
V4_AGG = RUN / "v4a_dev_aggregate_metrics.json"
V3_CI = RUN / "v3b_dev_bootstrap_confidence_intervals.json"
V4_CI = RUN / "v4a_dev_bootstrap_confidence_intervals.json"
QUERY_DELTAS = RUN / "v3b_vs_v4a_dev_query_deltas.csv"
PAIRED = RUN / "v3b_vs_v4a_dev_paired_comparison.json"
SELECTION = RUN / "v4a_dev_selection_result.json"
REPORT = RUN / "v3b_vs_v4a_dev_comparison_report.md"
METADATA = RUN / "v4a_dev_comparison_metadata.json"
LOCK = RUN / "v4a_dev_comparison_selection.lock.json"

EXPECTED_SPEC_SHA256 = "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9"
EXPECTED_SPEC_LOCK_SHA256 = "c9fee869cda6ccfd26c1ad0b2624cbcf693b9c896366b6baaa5c0b7b98371db2"
EXPECTED_COMPLETE_MAP_SHA256 = "60fb8b44f6eb13d486eaa6ffae483dca374e1415af8708d833912a18ec4d2000"
EXPECTED_FINALIZATION_LOCK_SHA256 = "fe5b4865df05c4086247360fc7693e71d3cfe63f8877e944eaea61016d418f23"

NQ = 24
K = 10
REPS = 5000
SEED = 20261002

METRICS = [
    "macro_mean_relevance_at_10",
    "clear_or_strong_rate_at_10",
    "irrelevant_rate_at_10",
    "hit_rate_at_5_clear_or_strong",
    "top1_clear_or_strong_rate",
    "mrr_at_10_clear_or_strong",
    "ndcg_at_10",
    "catastrophic_query_rate",
]

SELECTION_CRITERIA = [
    ("catastrophic_query_rate", "lower"),
    ("irrelevant_rate_at_10", "lower"),
    ("clear_or_strong_rate_at_10", "higher"),
    ("macro_mean_relevance_at_10", "higher"),
    ("ndcg_at_10", "higher"),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def bool_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def ndcg(scores: list[int]) -> float:
    gains = np.asarray([2 ** int(x) - 1 for x in scores], dtype=float)
    discounts = np.log2(np.arange(2, len(scores) + 2, dtype=float))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal / discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def query_summary(group: pd.DataFrame) -> dict[str, Any]:
    g = group.sort_values("rank", kind="stable")
    scores = g["judge_score"].astype(int).tolist()
    require(len(scores) == K, "Expected exactly top10 for a query")
    clear = np.asarray(scores) >= 3
    clear_ranks = [i + 1 for i, flag in enumerate(clear) if flag]
    first = min(clear_ranks) if clear_ranks else None
    return {
        "query_id": str(g["query_id"].iloc[0]),
        "query": str(g["query"].iloc[0]),
        "stratum": str(g["stratum"].iloc[0]),
        "mean_relevance_at_10": float(np.mean(scores)),
        "median_relevance_at_10": float(np.median(scores)),
        "clear_or_strong_rate_at_10": float(np.mean(clear)),
        "irrelevant_rate_at_10": float(np.mean(np.asarray(scores) == 0)),
        "hit_at_5_clear_or_strong": int(np.any(clear[:5])),
        "top1_clear_or_strong": int(bool(clear[0])),
        "first_clear_or_strong_rank": first,
        "mrr_at_10_clear_or_strong": 0.0 if first is None else 1.0 / float(first),
        "ndcg_at_10": ndcg(scores),
        "catastrophic_query": int(float(np.mean(scores)) < 1.5),
    }


def aggregate(qm: pd.DataFrame) -> dict[str, float]:
    return {
        "macro_mean_relevance_at_10": float(qm["mean_relevance_at_10"].mean()),
        "clear_or_strong_rate_at_10": float(qm["clear_or_strong_rate_at_10"].mean()),
        "irrelevant_rate_at_10": float(qm["irrelevant_rate_at_10"].mean()),
        "hit_rate_at_5_clear_or_strong": float(qm["hit_at_5_clear_or_strong"].mean()),
        "top1_clear_or_strong_rate": float(qm["top1_clear_or_strong"].mean()),
        "mrr_at_10_clear_or_strong": float(qm["mrr_at_10_clear_or_strong"].mean()),
        "ndcg_at_10": float(qm["ndcg_at_10"].mean()),
        "catastrophic_query_rate": float(qm["catastrophic_query"].mean()),
    }


def bootstrap(qm: pd.DataFrame) -> dict[str, Any]:
    columns = {
        "macro_mean_relevance_at_10": "mean_relevance_at_10",
        "clear_or_strong_rate_at_10": "clear_or_strong_rate_at_10",
        "irrelevant_rate_at_10": "irrelevant_rate_at_10",
        "hit_rate_at_5_clear_or_strong": "hit_at_5_clear_or_strong",
        "top1_clear_or_strong_rate": "top1_clear_or_strong",
        "mrr_at_10_clear_or_strong": "mrr_at_10_clear_or_strong",
        "ndcg_at_10": "ndcg_at_10",
        "catastrophic_query_rate": "catastrophic_query",
    }
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(qm), size=(REPS, len(qm)))
    out: dict[str, Any] = {}
    for metric, column in columns.items():
        values = qm[column].astype(float).to_numpy()
        distribution = np.mean(values[idx], axis=1)
        out[metric] = {
            "lower_95": float(np.quantile(distribution, 0.025)),
            "upper_95": float(np.quantile(distribution, 0.975)),
            "bootstrap_mean": float(np.mean(distribution)),
            "bootstrap_standard_error": float(np.std(distribution, ddof=1)),
        }
    return out


def reconstruct(frame: pd.DataFrame, *, membership: str, rank_col: str, label: str) -> pd.DataFrame:
    mask = frame[membership].map(bool_value)
    out = frame.loc[mask].copy()
    require(len(out) == NQ * K, f"{label}: expected {NQ*K} rows")
    out["rank"] = pd.to_numeric(out[rank_col], errors="raise").astype(int)
    out["judge_score"] = pd.to_numeric(out["final_judge_score"], errors="raise").astype(int)
    require(out["judge_score"].between(0, 4).all(), f"{label}: score outside 0..4")
    require(not out.duplicated(["query_id", "rank"]).any(), f"{label}: duplicate query/rank")
    counts = out.groupby("query_id").size()
    require(len(counts) == NQ and (counts == K).all(), f"{label}: not exact 24x10")
    for qid, group in out.groupby("query_id", sort=False):
        require(sorted(group["rank"].tolist()) == list(range(1, K + 1)), f"{label}/{qid}: ranks are not 1..10")
    return out


def standalone(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    qm = pd.DataFrame([query_summary(g) for _, g in frame.groupby("query_id", sort=True)])
    return qm, aggregate(qm), bootstrap(qm)


def paired_comparison(v3q: pd.DataFrame, v4q: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    require(list(v3q["query_id"]) == list(v4q["query_id"]), "Paired query IDs differ")
    mapping = {
        "macro_mean_relevance_at_10": ("mean_relevance_at_10", True),
        "clear_or_strong_rate_at_10": ("clear_or_strong_rate_at_10", True),
        "irrelevant_rate_at_10": ("irrelevant_rate_at_10", False),
        "hit_rate_at_5_clear_or_strong": ("hit_at_5_clear_or_strong", True),
        "top1_clear_or_strong_rate": ("top1_clear_or_strong", True),
        "mrr_at_10_clear_or_strong": ("mrr_at_10_clear_or_strong", True),
        "ndcg_at_10": ("ndcg_at_10", True),
        "catastrophic_query_rate": ("catastrophic_query", False),
    }
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, NQ, size=(REPS, NQ))
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for metric, (column, higher_is_better) in mapping.items():
        before = v3q[column].astype(float).to_numpy()
        after = v4q[column].astype(float).to_numpy()
        delta = after - before
        improvement = delta if higher_is_better else -delta
        distribution = np.mean(delta[idx], axis=1)
        metrics[metric] = {
            "v3b": float(np.mean(before)),
            "v4a": float(np.mean(after)),
            "raw_delta_v4a_minus_v3b": float(np.mean(delta)),
            "paired_ci_lower": float(np.quantile(distribution, 0.025)),
            "paired_ci_upper": float(np.quantile(distribution, 0.975)),
            "wins": int(np.sum(improvement > 0.0)),
            "ties": int(np.sum(improvement == 0.0)),
            "losses": int(np.sum(improvement < 0.0)),
            "higher_is_better": higher_is_better,
        }
        for i, qid in enumerate(v3q["query_id"].astype(str)):
            rows.append({
                "query_id": qid,
                "query": str(v3q.iloc[i]["query"]),
                "stratum": str(v3q.iloc[i]["stratum"]),
                "metric": metric,
                "v3b_value": float(before[i]),
                "v4a_value": float(after[i]),
                "raw_delta_v4a_minus_v3b": float(delta[i]),
                "outcome_for_v4a": "WIN" if improvement[i] > 0.0 else "LOSS" if improvement[i] < 0.0 else "TIE",
            })
    payload = {
        "status": "COMPLETE",
        "analysis_role": "development_secondary_evidence",
        "query_count": NQ,
        "bootstrap_replicates": REPS,
        "bootstrap_seed": SEED,
        "metrics": metrics,
    }
    return payload, pd.DataFrame(rows)


def select_candidate(v3a: dict[str, float], v4a: dict[str, float]) -> dict[str, Any]:
    evaluations: list[dict[str, Any]] = []
    selected = "v3b"
    decisive_metric: str | None = None
    decisive_direction: str | None = None

    for metric, direction in SELECTION_CRITERIA:
        baseline = float(v3a[metric])
        candidate = float(v4a[metric])
        if candidate == baseline:
            outcome = "TIE_CONTINUE"
        elif direction == "lower":
            outcome = "V4A_BETTER" if candidate < baseline else "V3B_BETTER"
        else:
            outcome = "V4A_BETTER" if candidate > baseline else "V3B_BETTER"

        evaluations.append({
            "metric": metric,
            "preferred_direction": direction,
            "v3b": baseline,
            "v4a": candidate,
            "outcome": outcome,
        })

        if outcome == "V4A_BETTER":
            selected = "v4a"
            decisive_metric = metric
            decisive_direction = direction
            break
        if outcome == "V3B_BETTER":
            selected = "v3b"
            decisive_metric = metric
            decisive_direction = direction
            break

    exact_tie = decisive_metric is None
    if exact_tie:
        selected = "v3b"

    return {
        "status": "V4A_DEVELOPMENT_SELECTION_COMPLETE",
        "selection_type": "pre_registered_lexicographic_v3b_vs_v4a",
        "ordered_criteria": [m for m, _ in SELECTION_CRITERIA],
        "evaluations_until_decision": evaluations,
        "selected_candidate": selected,
        "decisive_metric": decisive_metric,
        "decisive_preferred_direction": decisive_direction,
        "exact_tie_retains_v3b": exact_tie,
        "retrieval_diagnostics_can_override_selection": False,
        "parameter_search_after_results_seen": False,
        "development_only": True,
        "release_decision": False,
        "requires_new_unseen_v4_holdout": selected == "v4a",
        "next_stage": (
            "AUTHOR_AND_FREEZE_V4H01_V4H24_UNSEEN_HOLDOUT_BEFORE_RETRIEVAL"
            if selected == "v4a"
            else "V4A_NOT_SELECTED_STOP_OR_DESIGN_SEPARATE_V4B_EXPERIMENT"
        ),
    }


def verify_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    frozen = [
        ("v4a spec", SPEC, EXPECTED_SPEC_SHA256),
        ("v4a spec lock", SPEC_LOCK, EXPECTED_SPEC_LOCK_SHA256),
        ("complete score map", COMPLETE_MAP, EXPECTED_COMPLETE_MAP_SHA256),
        ("finalization lock", FINALIZATION_LOCK, EXPECTED_FINALIZATION_LOCK_SHA256),
    ]
    for label, path, expected in frozen:
        require(path.exists(), f"Missing frozen input: {path}")
        actual = sha256(path)
        require(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
        print(f"PASS  frozen {label} SHA-256 verified")

    spec = load_json(SPEC)
    require(spec.get("status") == "FROZEN_BEFORE_V4A_IMPLEMENTATION", "Unexpected v4a spec status")
    protocol = spec.get("development_protocol", {})
    require(protocol.get("query_set") == "H01-H24", "Development query set changed")
    bootstrap_contract = protocol.get("bootstrap", {})
    require(int(bootstrap_contract.get("replicates", -1)) == REPS, "Bootstrap replicate count changed")
    require(int(bootstrap_contract.get("seed", -1)) == SEED, "Bootstrap seed changed")
    selection = spec.get("development_selection_rule", {})
    expected_order = [
        "lower catastrophic_query_rate",
        "lower irrelevant_rate_at_10",
        "higher clear_or_strong_rate_at_10",
        "higher macro_mean_relevance_at_10",
        "higher ndcg_at_10",
    ]
    require(selection.get("ordered_criteria") == expected_order, "Development selection criteria/order changed")
    require(selection.get("final_tie_break") == "retain v3b", "Development final tie-break changed")
    require(selection.get("retrieval_diagnostics_can_explain_but_not_override_selection") is True, "Retrieval diagnostic override rule changed")

    lock = load_json(FINALIZATION_LOCK)
    require(lock.get("status") == "V4A_DEV_COMPLETE_DISPLAY_SCORE_MAP_FROZEN_BEFORE_COMPARISON", "Unexpected finalization lock status")
    require(lock.get("complete_score_map_sha256") == EXPECTED_COMPLETE_MAP_SHA256, "Finalization lock map hash mismatch")
    require(int(lock.get("displayed_pair_count", -1)) == 347, "Complete-map pair count changed")
    require(lock.get("development_comparison_calculated") is False, "Comparison was already calculated during finalization")
    require(lock.get("development_selection_decision_produced") is False, "Selection was already produced during finalization")
    require(lock.get("release_metrics_calculated") is False, "Release metrics were calculated during finalization")

    frame = pd.read_csv(COMPLETE_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "query", "stratum", "isbn13", "in_v3b", "v3b_rank", "in_v4a", "v4a_rank", "final_judge_score"
    }
    missing = required - set(frame.columns)
    require(not missing, f"Complete score map missing columns: {sorted(missing)}")
    require(len(frame) == 347, "Complete score map is not 347 rows")
    require(frame["query_id"].nunique() == NQ, "Complete score map is not H01-H24")
    require(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in complete score map")
    scores = pd.to_numeric(frame["final_judge_score"], errors="raise").astype(int)
    require(scores.between(0, 4).all(), "Complete score map contains score outside 0..4")
    print("PASS  frozen comparison structure verified: 347 pairs / 24 queries / score range 0..4")
    return frame, spec


def main() -> None:
    print("Semantic relevance v4a - v3b vs v4a DEVELOPMENT comparison")
    print("-----------------------------------------------------------")
    print("Development query set: H01-H24")
    print("Judge execution: DISABLED")
    print("Model execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release gates/decision: NOT CALCULATED")
    print("Selection rule: FROZEN LEXICOGRAPHIC")
    print()

    for path in [V3_QUERY_METRICS, V4_QUERY_METRICS, V3_AGG, V4_AGG, V3_CI, V4_CI,
                 QUERY_DELTAS, PAIRED, SELECTION, REPORT, METADATA, LOCK]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing comparison artifact: {path}")

    frame, _ = verify_inputs()
    v3 = reconstruct(frame, membership="in_v3b", rank_col="v3b_rank", label="v3b")
    v4 = reconstruct(frame, membership="in_v4a", rank_col="v4a_rank", label="v4a")
    print("PASS  reconstructed v3b and v4a as exact 24x10 systems")

    v3q, v3a, v3ci = standalone(v3)
    v4q, v4a, v4ci = standalone(v4)
    paired, delta_rows = paired_comparison(v3q, v4q)
    selection = select_candidate(v3a, v4a)

    V3_QUERY_METRICS.parent.mkdir(parents=True, exist_ok=True)
    v3q.to_csv(V3_QUERY_METRICS, index=False, encoding="utf-8")
    v4q.to_csv(V4_QUERY_METRICS, index=False, encoding="utf-8")
    V3_AGG.write_text(json.dumps(v3a, indent=2) + "\n", encoding="utf-8")
    V4_AGG.write_text(json.dumps(v4a, indent=2) + "\n", encoding="utf-8")
    V3_CI.write_text(json.dumps(v3ci, indent=2) + "\n", encoding="utf-8")
    V4_CI.write_text(json.dumps(v4ci, indent=2) + "\n", encoding="utf-8")
    delta_rows.to_csv(QUERY_DELTAS, index=False, encoding="utf-8")
    PAIRED.write_text(json.dumps(paired, indent=2) + "\n", encoding="utf-8")
    SELECTION.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Semantic Relevance v4a — Development Comparison",
        "",
        "H01-H24 are development-only for v4. This report does not make a release claim.",
        "",
        f"**Selected development candidate: {selection['selected_candidate']}**",
        f"**Decisive criterion: {selection['decisive_metric'] or 'exact tie; retain v3b'}**",
        "",
        "## Standalone development metrics",
        "",
        "| Metric | v3b | v4a |",
        "|---|---:|---:|",
    ]
    for metric in METRICS:
        lines.append(f"| `{metric}` | {v3a[metric]:.6f} | {v4a[metric]:.6f} |")
    lines += [
        "",
        "## Pre-registered lexicographic selection",
        "",
    ]
    for item in selection["evaluations_until_decision"]:
        lines.append(
            f"- `{item['metric']}` ({item['preferred_direction']} is better): "
            f"v3b={item['v3b']:.6f}, v4a={item['v4a']:.6f} → {item['outcome']}"
        )
    lines += [
        "",
        "Retrieval diagnostics may explain the result but cannot override this selection.",
        "No parameter search is permitted after these results are observed.",
        "A new unseen V4H01-V4H24 holdout is required only if v4a is selected.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    outputs = [V3_QUERY_METRICS, V4_QUERY_METRICS, V3_AGG, V4_AGG, V3_CI, V4_CI,
               QUERY_DELTAS, PAIRED, SELECTION, REPORT]
    metadata = {
        "status": "V4A_DEVELOPMENT_COMPARISON_AND_SELECTION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "development_query_count": NQ,
        "complete_score_map_sha256": EXPECTED_COMPLETE_MAP_SHA256,
        "finalization_lock_sha256": EXPECTED_FINALIZATION_LOCK_SHA256,
        "bootstrap_replicates": REPS,
        "bootstrap_seed": SEED,
        "selected_candidate": selection["selected_candidate"],
        "decisive_metric": selection["decisive_metric"],
        "judge_calls_performed_during_comparison": 0,
        "model_calls_performed_during_comparison": 0,
        "human_relevance_labels_used": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "requires_new_unseen_v4_holdout": selection["requires_new_unseen_v4_holdout"],
        "next_stage": selection["next_stage"],
    }
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock_payload = {
        "status": "V4A_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": EXPECTED_SPEC_SHA256,
        "spec_lock_sha256": EXPECTED_SPEC_LOCK_SHA256,
        "complete_score_map_sha256": EXPECTED_COMPLETE_MAP_SHA256,
        "finalization_lock_sha256": EXPECTED_FINALIZATION_LOCK_SHA256,
        "development_query_set": "H01-H24",
        "development_query_count": NQ,
        "bootstrap_replicates": REPS,
        "bootstrap_seed": SEED,
        "selection_type": "pre_registered_lexicographic_v3b_vs_v4a",
        "selection_order": [m for m, _ in SELECTION_CRITERIA],
        "final_tie_break": "retain v3b",
        "selected_candidate": selection["selected_candidate"],
        "decisive_metric": selection["decisive_metric"],
        "requires_new_unseen_v4_holdout": selection["requires_new_unseen_v4_holdout"],
        "judge_calls_performed_during_comparison": 0,
        "model_calls_performed_during_comparison": 0,
        "human_relevance_labels_used": False,
        "release_gates_calculated": False,
        "release_decision_produced": False,
        "outputs": {p.name: sha256(p) for p in outputs},
        "metadata_sha256": sha256(METADATA),
        "next_stage": selection["next_stage"],
    }
    LOCK.write_text(json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8")

    print()
    print("V3B DEVELOPMENT")
    print(
        f"Mean@10={v3a['macro_mean_relevance_at_10']:.3f} "
        f"Clear={v3a['clear_or_strong_rate_at_10']:.1%} "
        f"Irrelevant={v3a['irrelevant_rate_at_10']:.1%} "
        f"Hit@5={v3a['hit_rate_at_5_clear_or_strong']:.1%} "
        f"Top1={v3a['top1_clear_or_strong_rate']:.1%} "
        f"MRR={v3a['mrr_at_10_clear_or_strong']:.3f} "
        f"nDCG={v3a['ndcg_at_10']:.3f} "
        f"Cat={v3a['catastrophic_query_rate']:.1%}"
    )
    print("V4A DEVELOPMENT")
    print(
        f"Mean@10={v4a['macro_mean_relevance_at_10']:.3f} "
        f"Clear={v4a['clear_or_strong_rate_at_10']:.1%} "
        f"Irrelevant={v4a['irrelevant_rate_at_10']:.1%} "
        f"Hit@5={v4a['hit_rate_at_5_clear_or_strong']:.1%} "
        f"Top1={v4a['top1_clear_or_strong_rate']:.1%} "
        f"MRR={v4a['mrr_at_10_clear_or_strong']:.3f} "
        f"nDCG={v4a['ndcg_at_10']:.3f} "
        f"Cat={v4a['catastrophic_query_rate']:.1%}"
    )
    print()
    print("PRE-REGISTERED DEVELOPMENT SELECTION")
    for item in selection["evaluations_until_decision"]:
        print(
            f"{item['metric']}: v3b={item['v3b']:.6f} v4a={item['v4a']:.6f} "
            f"preferred={item['preferred_direction']} -> {item['outcome']}"
        )
    print(f"SELECTED DEVELOPMENT CANDIDATE: {selection['selected_candidate']}")
    print(f"Decisive metric: {selection['decisive_metric'] or 'exact tie -> retain v3b'}")
    print(f"New unseen V4 holdout required: {'YES' if selection['requires_new_unseen_v4_holdout'] else 'NO'}")
    print("Judge calls during comparison: 0")
    print("Model calls during comparison: 0")
    print("Human relevance labels used: NO")
    print("Release gates calculated: NO")
    print("Release decision produced: NO")
    print(f"Comparison/selection lock SHA-256: {sha256(LOCK)}")
    print("V4A DEVELOPMENT COMPARISON AND SELECTION: PASS")


if __name__ == "__main__":
    main()
