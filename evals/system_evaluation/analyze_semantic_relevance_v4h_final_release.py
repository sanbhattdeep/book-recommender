from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"
SYS = EVALS / "system_evaluation"
RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"

SCORES = RUN / "v4h_judge_scores.csv"
SCORING_LOCK = RUN / "v4h_judge_scoring.lock.json"
JUDGING_INPUT = RUN / "v4h_judging_input.csv"
DISPLAY_UNION = RUN / "v4h_v3b_v4b_display_union.csv"
V3B_RECS = RUN / "v4h_v3b_recommendations.csv"
V4B_RECS = RUN / "v4h_v4b_recommendations.csv"
RERANKING_LOCK = RUN / "v4h_reranking.lock.json"
ANALYSIS_CONTRACT = SYS / "semantic_relevance_v4h_release_analysis_contract.v0.1.0.json"
SYSTEM_EVAL_CONTRACT = SYS / "semantic_relevance_system_eval.v1.0.0.json"


EXPECTED = {
    "judge_scores": "8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906",
    "judge_scoring_lock": "ac7a96c89eb2d4a51eda1b56756ed3b943311e7f300b661659cff51285423a8f",
    "judging_input": "64335ac9e30a2d32aa71ad4306bffdb82f22ef7feef18dd1c92459f2f33a7645",
    "display_union": "444fa3722dc1e8815e111ab95db9a35b501ae9e57d692d560c6c5fe5955f096a",
    "v3b_recommendations": "2f1aec34740f35127822a634e00f772db716e35e2d8aeaeaf729d54ecc95b505",
    "v4b_recommendations": "27c8fb4502909f0a3f89d6b6dfbcec55ac82ae489eab800d51334f312d16fec2",
    "reranking_lock": "6d3a7014f5995d978a90df627c720c0b81aa2e8b28e8bba99cfa2c64d27811b5"
}
NQ = 24
K = 10
REPS = 5000
SEED = 20261002
EXPECTED_CONTRACT_SHA256 = "c44a4699aef5d041a8cd33c07eff481d540624404a40588563c84552104bab9a"
EXPECTED_QUERY_IDS = [f"V4H{i:02d}" for i in range(1, 25)]
EXPECTED_GATES = {'macro_mean_relevance_at_10_ge_2_50': ('macro_mean_relevance_at_10', '>=', 2.5), 'macro_mean_relevance_ci_lower_ge_2_00': ('macro_mean_relevance_at_10_ci_lower_95', '>=', 2.0), 'clear_or_strong_rate_at_10_ge_0_60': ('clear_or_strong_rate_at_10', '>=', 0.6), 'irrelevant_rate_at_10_le_0_10': ('irrelevant_rate_at_10', '<=', 0.1), 'hit_rate_at_5_clear_or_strong_ge_0_90': ('hit_rate_at_5_clear_or_strong', '>=', 0.9), 'top1_clear_or_strong_rate_ge_0_75': ('top1_clear_or_strong_rate', '>=', 0.75), 'ndcg_at_10_ge_0_85': ('ndcg_at_10', '>=', 0.85), 'catastrophic_query_rate_eq_0': ('catastrophic_query_rate', '==', 0.0)}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def bool_value(value: Any) -> bool:
    s = str(value).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"Invalid boolean {value!r}")

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def verify_frozen_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    for path, key, label in [
        (SCORES, "judge_scores", "V4H judge scores"),
        (SCORING_LOCK, "judge_scoring_lock", "V4H judge scoring lock"),
        (JUDGING_INPUT, "judging_input", "V4H frozen judging input"),
        (DISPLAY_UNION, "display_union", "V4H displayed-pair union"),
        (V3B_RECS, "v3b_recommendations", "V3B V4H recommendations"),
        (V4B_RECS, "v4b_recommendations", "V4B V4H recommendations"),
        (RERANKING_LOCK, "reranking_lock", "V4H reranking lock"),
        (ANALYSIS_CONTRACT, None, "V4H release-analysis contract"),
    ]:
        expected = EXPECTED_CONTRACT_SHA256 if key is None else EXPECTED[key]
        verify_raw(path, expected, label)

    lock = load_json(SCORING_LOCK)
    req(lock.get("status") == "V4H_JUDGE_SCORES_FROZEN_BEFORE_RELEASE_ANALYSIS",
        "Unexpected V4H scoring-lock status")
    req(int(lock.get("case_count", -1)) == 382, "Scoring lock case count changed")
    req(lock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5",
        "Judge candidate changed")
    req(lock.get("judge_scores_sha256") == EXPECTED["judge_scores"],
        "Scoring lock score hash mismatch")
    req(lock.get("human_relevance_labels_used") is False,
        "Scoring lock says human labels were used")
    req(lock.get("release_metrics_calculated") is False,
        "Release metrics were already calculated during judging")
    req(lock.get("release_decision_produced") is False,
        "Release decision was already produced during judging")

    frame = pd.read_csv(SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "case_id", "query_id", "query", "stratum", "isbn13",
        "pair_source", "in_v3b", "v3b_rank", "in_v4b", "v4b_rank", "judge_score"
    }
    missing = required - set(frame.columns)
    req(not missing, f"Score file missing columns: {sorted(missing)}")
    req(len(frame) == 382, f"Expected 382 score rows, got {len(frame)}")
    req(not frame["case_id"].duplicated().any(), "Duplicate score case IDs")
    req(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate scored query-book pair")
    req(sorted(frame["query_id"].unique().tolist()) == EXPECTED_QUERY_IDS,
        "Score file query IDs changed")
    frame["judge_score"] = pd.to_numeric(frame["judge_score"], errors="raise").astype(int)
    req(frame["judge_score"].between(0, 4).all(), "Judge score outside 0..4")

    for membership, rank_col, label in [
        ("in_v3b", "v3b_rank", "V3B"),
        ("in_v4b", "v4b_rank", "V4B"),
    ]:
        subset = frame[frame[membership].map(bool_value)].copy()
        req(len(subset) == NQ*K, f"{label} expected 240 displayed rows, got {len(subset)}")
        subset["rank"] = pd.to_numeric(subset[rank_col], errors="raise").astype(int)
        req(not subset.duplicated(["query_id", "rank"]).any(),
            f"{label} duplicate query/rank")
        counts = subset.groupby("query_id").size()
        req(len(counts) == NQ and (counts == K).all(),
            f"{label} is not exact 24x10")
        for qid, group in subset.groupby("query_id", sort=False):
            req(sorted(group["rank"].tolist()) == list(range(1, K+1)),
                f"{label}/{qid} ranks are not 1..10")
    print("PASS  frozen score structure reconstructs V3B and V4B as exact 24x10")
    return frame, lock

def verify_release_contract() -> dict[str, Any]:
    contract = load_json(SYSTEM_EVAL_CONTRACT)
    stat = contract.get("statistical_uncertainty", {})
    req(int(stat.get("bootstrap_replicates", -1)) == REPS,
        "System-eval bootstrap replicate count changed")
    req(int(stat.get("seed", -1)) == SEED,
        "System-eval bootstrap seed changed")

    gates = contract.get("release_gates", {})
    req(list(gates.keys()) == list(EXPECTED_GATES.keys()),
        "Release gate names/order changed")
    for name, (metric, op, value) in EXPECTED_GATES.items():
        gate = gates[name]
        req(
            (gate.get("metric"), gate.get("op"), float(gate.get("value"))) ==
            (metric, op, float(value)),
            f"Release gate changed: {name}"
        )
    print("PASS  original eight release gates and bootstrap contract verified")
    return contract


CI_METRICS = [
    "macro_mean_relevance_at_10",
    "clear_or_strong_rate_at_10",
    "irrelevant_rate_at_10",
    "hit_rate_at_5_clear_or_strong",
    "top1_clear_or_strong_rate",
    "mrr_at_10_clear_or_strong",
    "ndcg_at_10",
]
PAIR_MAP = {
    "macro_mean_relevance_at_10": ("mean_relevance_at_10", True),
    "clear_or_strong_rate_at_10": ("clear_or_strong_rate_at_10", True),
    "irrelevant_rate_at_10": ("irrelevant_rate_at_10", False),
    "hit_rate_at_5_clear_or_strong": ("hit_at_5_clear_or_strong", True),
    "top1_clear_or_strong_rate": ("top1_clear_or_strong", True),
    "mrr_at_10_clear_or_strong": ("mrr_at_10_clear_or_strong", True),
    "ndcg_at_10": ("ndcg_at_10", True),
    "catastrophic_query_rate": ("catastrophic_query", False),
}

def ndcg(scores: list[int]) -> float:
    gains = np.array([2**int(x)-1 for x in scores], dtype=float)
    discounts = np.log2(np.arange(2, len(scores)+2, dtype=float))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal / discounts))
    return 0.0 if idcg == 0 else dcg / idcg

def reconstruct(frame: pd.DataFrame, membership: str, rank_col: str, label: str) -> pd.DataFrame:
    out = frame[frame[membership].map(bool_value)].copy()
    out["rank"] = pd.to_numeric(out[rank_col], errors="raise").astype(int)
    out["judge_score"] = pd.to_numeric(out["judge_score"], errors="raise").astype(int)
    req(len(out) == NQ*K, f"{label}: expected 240 rows")
    req(not out.duplicated(["query_id","rank"]).any(), f"{label}: duplicate rank")
    for qid, g in out.groupby("query_id", sort=False):
        req(sorted(g["rank"].tolist()) == list(range(1,11)), f"{label}/{qid}: bad ranks")
    return out

def query_summary(group: pd.DataFrame) -> dict[str, Any]:
    g = group.sort_values("rank", kind="stable")
    scores = g["judge_score"].astype(int).tolist()
    req(len(scores) == K, "Expected exactly top10")
    clear = np.asarray(scores) >= 3
    zero = np.asarray(scores) == 0
    ranks = [i+1 for i, flag in enumerate(clear) if flag]
    first = min(ranks) if ranks else None
    return {
        "query_id": str(g["query_id"].iloc[0]),
        "query": str(g["query"].iloc[0]),
        "stratum": str(g["stratum"].iloc[0]),
        "mean_relevance_at_10": float(np.mean(scores)),
        "median_relevance_at_10": float(np.median(scores)),
        "clear_or_strong_rate_at_10": float(np.mean(clear)),
        "irrelevant_rate_at_10": float(np.mean(zero)),
        "hit_at_5_clear_or_strong": int(np.any(clear[:5])),
        "top1_clear_or_strong": int(bool(clear[0])),
        "first_clear_or_strong_rank": first,
        "mrr_at_10_clear_or_strong": 0.0 if first is None else 1.0/float(first),
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
    }
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(qm), size=(REPS, len(qm)))
    out = {}
    for metric, column in columns.items():
        values = qm[column].astype(float).to_numpy()
        dist = np.mean(values[idx], axis=1)
        out[metric] = {
            "lower_95": float(np.quantile(dist, 0.025)),
            "upper_95": float(np.quantile(dist, 0.975)),
            "bootstrap_mean": float(np.mean(dist)),
            "bootstrap_standard_error": float(np.std(dist, ddof=1)),
        }
    return out

def standalone(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float], dict[str, Any]]:
    qm = pd.DataFrame([query_summary(g) for _, g in frame.groupby("query_id", sort=True)])
    agg = aggregate(qm)
    ci = bootstrap(qm)
    agg["macro_mean_relevance_at_10_ci_lower_95"] = ci["macro_mean_relevance_at_10"]["lower_95"]
    agg["macro_mean_relevance_at_10_ci_upper_95"] = ci["macro_mean_relevance_at_10"]["upper_95"]
    return qm, agg, ci

def gate_result(agg: dict[str, Any], system_contract: dict[str, Any]) -> dict[str, Any]:
    checks, failed = {}, []
    for name, gate in system_contract["release_gates"].items():
        actual = float(agg[gate["metric"]])
        threshold = float(gate["value"])
        op = gate["op"]
        passed = actual >= threshold if op == ">=" else actual <= threshold if op == "<=" else actual == threshold
        checks[name] = {
            "passed": passed,
            "metric": gate["metric"],
            "actual": actual,
            "operator": op,
            "threshold": threshold,
        }
        if not passed:
            failed.append(name)
    cat = float(agg["catastrophic_query_rate"])
    decision = "PASS" if not failed else ("REVIEW" if len(failed) <= 2 and cat == 0 else "FAIL")
    return {
        "decision": decision,
        "passed_gate_count": len(checks)-len(failed),
        "total_gate_count": len(checks),
        "failed_gate_count": len(failed),
        "failed_gates": failed,
        "checks": checks,
        "decision_policy": system_contract["decision_policy"],
    }

def paired(v3q: pd.DataFrame, v4q: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    req(list(v3q["query_id"]) == list(v4q["query_id"]), "Paired query IDs differ")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, NQ, size=(REPS, NQ))
    metrics, rows = {}, []
    for metric, (column, higher_is_better) in PAIR_MAP.items():
        before = v3q[column].astype(float).to_numpy()
        after = v4q[column].astype(float).to_numpy()
        delta = after-before
        improvement = delta if higher_is_better else -delta
        dist = np.mean(delta[idx], axis=1)
        metrics[metric] = {
            "v3b": float(np.mean(before)),
            "v4b": float(np.mean(after)),
            "raw_delta_v4b_minus_v3b": float(np.mean(delta)),
            "paired_ci_lower": float(np.quantile(dist, 0.025)),
            "paired_ci_upper": float(np.quantile(dist, 0.975)),
            "wins": int(np.sum(improvement > 1e-15)),
            "ties": int(np.sum(np.abs(improvement) <= 1e-15)),
            "losses": int(np.sum(improvement < -1e-15)),
            "higher_is_better": higher_is_better,
        }
        for i, qid in enumerate(v3q["query_id"].astype(str)):
            rows.append({
                "query_id": qid,
                "query": str(v3q.iloc[i]["query"]),
                "stratum": str(v3q.iloc[i]["stratum"]),
                "metric": metric,
                "v3b_value": float(before[i]),
                "v4b_value": float(after[i]),
                "raw_delta_v4b_minus_v3b": float(delta[i]),
                "outcome_for_v4b": (
                    "WIN" if improvement[i] > 1e-15 else
                    "LOSS" if improvement[i] < -1e-15 else "TIE"
                ),
            })
    return {
        "status": "COMPLETE",
        "analysis_role": "secondary",
        "cannot_override_standalone_v4b_release_decision": True,
        "matched_query_count": NQ,
        "bootstrap_replicates": REPS,
        "bootstrap_seed": SEED,
        "metrics": metrics,
    }, pd.DataFrame(rows)

def main() -> None:
    print("Semantic relevance V4H - FINAL V4B release analysis")
    print("----------------------------------------------------")
    print("Authoritative system: V4B")
    print("V3B comparison role: SECONDARY")
    print("Judge/model/retrieval/reranker calls: 0")
    print("Human relevance labels: NOT USED")
    print()

    frame, _ = verify_frozen_inputs()
    system_contract = verify_release_contract()

    outputs = [
        "v3b_v4h_query_metrics.csv", "v4b_v4h_query_metrics.csv",
        "v3b_v4h_aggregate_metrics.json", "v4b_v4h_aggregate_metrics.json",
        "v3b_v4h_bootstrap_confidence_intervals.json", "v4b_v4h_bootstrap_confidence_intervals.json",
        "v3b_v4h_release_gate_result.json", "v4b_v4h_release_gate_result.json",
        "v3b_vs_v4b_v4h_query_deltas.csv", "v3b_vs_v4b_v4h_paired_comparison.json",
        "v4h_stratum_metrics.csv", "v4h_final_release_report.md",
        "v4h_final_release_analysis_metadata.json", "v4h_final_release_analysis.lock.json",
    ]
    for name in outputs:
        req(not (RUN/name).exists(), f"Refusing to overwrite existing release-analysis artifact: {name}")

    v3 = reconstruct(frame, "in_v3b", "v3b_rank", "V3B")
    v4 = reconstruct(frame, "in_v4b", "v4b_rank", "V4B")
    v3q, v3a, v3ci = standalone(v3)
    v4q, v4a, v4ci = standalone(v4)
    req(list(v3q["query_id"]) == list(v4q["query_id"]), "V3B/V4B query mismatch")
    v3g = gate_result(v3a, system_contract)
    v4g = gate_result(v4a, system_contract)
    paired_payload, delta_rows = paired(v3q, v4q)

    v3q.to_csv(RUN/"v3b_v4h_query_metrics.csv", index=False, encoding="utf-8")
    v4q.to_csv(RUN/"v4b_v4h_query_metrics.csv", index=False, encoding="utf-8")
    (RUN/"v3b_v4h_aggregate_metrics.json").write_text(json.dumps(v3a, indent=2)+"\n", encoding="utf-8")
    (RUN/"v4b_v4h_aggregate_metrics.json").write_text(json.dumps(v4a, indent=2)+"\n", encoding="utf-8")
    (RUN/"v3b_v4h_bootstrap_confidence_intervals.json").write_text(json.dumps(v3ci, indent=2)+"\n", encoding="utf-8")
    (RUN/"v4b_v4h_bootstrap_confidence_intervals.json").write_text(json.dumps(v4ci, indent=2)+"\n", encoding="utf-8")
    (RUN/"v3b_v4h_release_gate_result.json").write_text(json.dumps(v3g, indent=2)+"\n", encoding="utf-8")
    (RUN/"v4b_v4h_release_gate_result.json").write_text(json.dumps(v4g, indent=2)+"\n", encoding="utf-8")
    delta_rows.to_csv(RUN/"v3b_vs_v4b_v4h_query_deltas.csv", index=False, encoding="utf-8")
    (RUN/"v3b_vs_v4b_v4h_paired_comparison.json").write_text(json.dumps(paired_payload, indent=2)+"\n", encoding="utf-8")

    # Descriptive stratum breakdown; diagnostic only, not release-gating.
    stratum_rows = []
    for system, qm in [("v3b", v3q), ("v4b", v4q)]:
        for stratum, g in qm.groupby("stratum", sort=True):
            stratum_rows.append({
                "system": system,
                "stratum": stratum,
                "query_count": len(g),
                "mean_relevance_at_10": float(g["mean_relevance_at_10"].mean()),
                "clear_or_strong_rate_at_10": float(g["clear_or_strong_rate_at_10"].mean()),
                "irrelevant_rate_at_10": float(g["irrelevant_rate_at_10"].mean()),
                "hit_rate_at_5_clear_or_strong": float(g["hit_at_5_clear_or_strong"].mean()),
                "top1_clear_or_strong_rate": float(g["top1_clear_or_strong"].mean()),
                "mrr_at_10_clear_or_strong": float(g["mrr_at_10_clear_or_strong"].mean()),
                "ndcg_at_10": float(g["ndcg_at_10"].mean()),
                "catastrophic_query_rate": float(g["catastrophic_query"].mean()),
                "role": "diagnostic_only_not_release_gating",
            })
    pd.DataFrame(stratum_rows).to_csv(RUN/"v4h_stratum_metrics.csv", index=False, encoding="utf-8")

    report = [
        "# Semantic Relevance V4B — Final Unseen V4H Release Report",
        "",
        f"**Authoritative standalone V4B decision: {v4g['decision']} ({v4g['passed_gate_count']}/8 gates)**",
        "",
        "The standalone V4B V4H decision is authoritative. The V3B comparison and paired analysis are secondary evidence and cannot override it.",
        "",
        "- V4H queries: 24",
        "- Frozen unique displayed pairs judged: 382",
        "- V4B displayed recommendations evaluated: 240",
        "- V3B baseline recommendations evaluated: 240",
        "- Human relevance labels: No",
        "- Judge/model calls during analysis: 0",
        f"- Frozen judge-score SHA-256: `8f8249a609287477a7f094ef774dd8f8b2af4b9b9b142d58e6c22123ae15f906`",
        "",
        "## Standalone metrics",
        "",
        "| Metric | V3B | V4B |",
        "|---|---:|---:|",
    ]
    for metric in CI_METRICS:
        report.append(f"| `{metric}` | {v3a[metric]:.4f} | {v4a[metric]:.4f} |")
    report += [
        "",
        f"- V3B catastrophic query rate: {v3a['catastrophic_query_rate']:.1%}",
        f"- V4B catastrophic query rate: {v4a['catastrophic_query_rate']:.1%}",
        f"- V3B gate result: {v3g['decision']} ({v3g['passed_gate_count']}/8)",
        f"- **V4B authoritative gate result: {v4g['decision']} ({v4g['passed_gate_count']}/8)**",
        "",
        "## V4B release gates",
        "",
    ]
    for name, check in v4g["checks"].items():
        report.append(
            f"- {'PASS' if check['passed'] else 'FAIL'} `{name}`: "
            f"actual={check['actual']:.6f} {check['operator']} {check['threshold']:.6f}"
        )
    report += [
        "",
        "## Secondary paired V3B → V4B evidence",
        "",
        "| Metric | V4B−V3B | 95% paired CI | W/T/L |",
        "|---|---:|---:|---:|",
    ]
    for metric, result in paired_payload["metrics"].items():
        report.append(
            f"| `{metric}` | {result['raw_delta_v4b_minus_v3b']:+.4f} | "
            f"[{result['paired_ci_lower']:+.4f}, {result['paired_ci_upper']:+.4f}] | "
            f"{result['wins']}/{result['ties']}/{result['losses']} |"
        )
    report += [
        "",
        "Standalone and paired confidence intervals use 5,000 query-level cluster bootstrap replicates with seed `20261002`.",
        "Stratum metrics are descriptive diagnostics only and do not modify the release gates.",
        "The paired comparison cannot override the standalone V4B release decision.",
    ]
    (RUN/"v4h_final_release_report.md").write_text("\n".join(report)+"\n", encoding="utf-8")

    metadata = {
        "status": "V4H_FINAL_RELEASE_ANALYSIS_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_count": NQ,
        "judge_scores_sha256": EXPECTED["judge_scores"],
        "judge_scoring_lock_sha256": EXPECTED["judge_scoring_lock"],
        "bootstrap_replicates": REPS,
        "bootstrap_seed": SEED,
        "authoritative_system": "v4b",
        "v4b_final_release_decision": v4g["decision"],
        "v4b_passed_gates": v4g["passed_gate_count"],
        "v3b_secondary_gate_result": v3g["decision"],
        "paired_comparison_role": "secondary",
        "paired_cannot_override_standalone_v4b": True,
        "judge_calls_performed_during_analysis": 0,
        "model_calls_performed_during_analysis": 0,
        "human_relevance_labels_used": False,
    }
    meta_path = RUN/"v4h_final_release_analysis_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")

    output_names = [
        "v3b_v4h_query_metrics.csv", "v4b_v4h_query_metrics.csv",
        "v3b_v4h_aggregate_metrics.json", "v4b_v4h_aggregate_metrics.json",
        "v3b_v4h_bootstrap_confidence_intervals.json", "v4b_v4h_bootstrap_confidence_intervals.json",
        "v3b_v4h_release_gate_result.json", "v4b_v4h_release_gate_result.json",
        "v3b_vs_v4b_v4h_query_deltas.csv", "v3b_vs_v4b_v4h_paired_comparison.json",
        "v4h_stratum_metrics.csv", "v4h_final_release_report.md",
        "v4h_final_release_analysis_metadata.json",
    ]
    lock = {
        "status": "V4H_FINAL_RELEASE_ANALYSIS_FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_scores_sha256": EXPECTED["judge_scores"],
        "judge_scoring_lock_sha256": EXPECTED["judge_scoring_lock"],
        "release_analysis_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "authoritative_system": "v4b",
        "v4b_final_release_decision": v4g["decision"],
        "v4b_passed_gates": v4g["passed_gate_count"],
        "v4b_total_gates": 8,
        "v3b_comparison_role": "secondary",
        "paired_comparison_cannot_override_v4b_decision": True,
        "bootstrap_replicates": REPS,
        "bootstrap_seed": SEED,
        "judge_calls_performed_during_analysis": 0,
        "model_calls_performed_during_analysis": 0,
        "human_relevance_labels_used": False,
        "outputs": {name: sha256(RUN/name) for name in output_names},
    }
    lock_path = RUN/"v4h_final_release_analysis.lock.json"
    lock_path.write_text(json.dumps(lock, indent=2)+"\n", encoding="utf-8")

    print()
    print("V3B SECONDARY STANDALONE ON V4H")
    print(
        f"Mean@10={v3a['macro_mean_relevance_at_10']:.3f} "
        f"CI=[{v3ci['macro_mean_relevance_at_10']['lower_95']:.3f},{v3ci['macro_mean_relevance_at_10']['upper_95']:.3f}] "
        f"Clear={v3a['clear_or_strong_rate_at_10']:.1%} "
        f"Irrelevant={v3a['irrelevant_rate_at_10']:.1%} "
        f"Hit@5={v3a['hit_rate_at_5_clear_or_strong']:.1%} "
        f"Top1={v3a['top1_clear_or_strong_rate']:.1%} "
        f"MRR={v3a['mrr_at_10_clear_or_strong']:.3f} "
        f"nDCG={v3a['ndcg_at_10']:.3f} "
        f"Cat={v3a['catastrophic_query_rate']:.1%}"
    )
    print(f"Secondary gate result={v3g['decision']} ({v3g['passed_gate_count']}/8)")
    print()
    print("V4B AUTHORITATIVE FINAL UNSEEN HOLDOUT")
    print(
        f"Mean@10={v4a['macro_mean_relevance_at_10']:.3f} "
        f"CI=[{v4ci['macro_mean_relevance_at_10']['lower_95']:.3f},{v4ci['macro_mean_relevance_at_10']['upper_95']:.3f}] "
        f"Clear={v4a['clear_or_strong_rate_at_10']:.1%} "
        f"Irrelevant={v4a['irrelevant_rate_at_10']:.1%} "
        f"Hit@5={v4a['hit_rate_at_5_clear_or_strong']:.1%} "
        f"Top1={v4a['top1_clear_or_strong_rate']:.1%} "
        f"MRR={v4a['mrr_at_10_clear_or_strong']:.3f} "
        f"nDCG={v4a['ndcg_at_10']:.3f} "
        f"Cat={v4a['catastrophic_query_rate']:.1%}"
    )
    print(f"FINAL V4B RELEASE DECISION={v4g['decision']} ({v4g['passed_gate_count']}/8 gates)")
    print()
    print("SECONDARY PAIRED V3B -> V4B")
    for metric, result in paired_payload["metrics"].items():
        print(
            f"{metric}: delta={result['raw_delta_v4b_minus_v3b']:+.4f} "
            f"CI=[{result['paired_ci_lower']:+.4f},{result['paired_ci_upper']:+.4f}] "
            f"W/T/L={result['wins']}/{result['ties']}/{result['losses']}"
        )
    print()
    print("Judge calls during analysis: 0")
    print("Model/retrieval/reranker calls during analysis: 0")
    print("Human relevance labels used: NO")
    print(f"Final analysis lock SHA-256: {sha256(lock_path)}")
    print("V4H FINAL RELEASE ANALYSIS: PASS")

if __name__ == "__main__":
    main()
