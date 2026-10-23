from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
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


def main() -> None:
    print("Semantic relevance V4H - final release-analysis preflight")
    print("---------------------------------------------------------")
    print("Judge execution: DISABLED")
    print("Retrieval/reranker execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Release metrics: NOT CALCULATED")
    print("Release decision: NOT PRODUCED")
    print()

    frame, _ = verify_frozen_inputs()
    verify_release_contract()

    # Structural-only checks. No aggregate metric or release gate is calculated here.
    req(frame["query_id"].nunique() == 24, "Expected 24 V4H queries")
    req(frame["stratum"].nunique() == 6, "Expected six frozen V4H strata")

    analysis_lock = RUN / "v4h_final_release_analysis.lock.json"
    req(not analysis_lock.exists(),
        "Final release-analysis lock already exists; refusing to restart analysis")

    print()
    print("V4H FINAL RELEASE-ANALYSIS PREFLIGHT: PASS")
    print("Frozen unique scored pairs: 382")
    print("V3B displayed rows: 240")
    print("V4B displayed rows: 240")
    print("Holdout queries: 24")
    print("Bootstrap: 5000 query-cluster replicates / seed 20261002")
    print("Authoritative release system: V4B")
    print("Paired V3B-vs-V4B role: SECONDARY / CANNOT OVERRIDE")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: calculate standalone V4B release metrics and secondary paired V3B-vs-V4B evidence.")

if __name__ == "__main__":
    main()
