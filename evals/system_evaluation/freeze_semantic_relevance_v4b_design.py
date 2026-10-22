from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SYS = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

SPEC_JSON = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.json"
SPEC_MD = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.md"
LOCK = SYS / "semantic_relevance_v4b_reranker_only_hybrid_experiment.v0.1.0.lock.json"

V4A_SPEC = SYS / "semantic_relevance_v4a_hybrid_retrieval_experiment.v0.1.0.json"
RETRIEVAL_LOCK = RUN / "v4a_dev_retrieval.lock.json"
HYBRID_UNION = RUN / "v4a_dev_hybrid_union.csv"
RERANKER_SCORES = RUN / "v4a_dev_reranker_scores.csv"
RERANKER_LOCK = RUN / "v4a_dev_reranker.lock.json"
COMPARISON_LOCK = RUN / "v4a_dev_comparison_selection.lock.json"
SENSITIVITY_LOCK = RUN / "v4a_retrieval_diagnostic_sensitivity_analysis.lock.json"
SELECTED_V3_LOCK = SYS / "semantic_relevance_v3_selected_candidate.lock.json"
V3_RERANKER = SYS / "semantic_relevance_v3_reranker.py"

EXPECTED = {
    "spec_json": "c72b5edbc330feae0620314fa5d5c6f90a55b356174d70a075ae77b47a8d0fca",
    "spec_md": "dae818b2519be70d49880e28642d76c414fa2a6ccda139242af5174984d29850",
    "v4a_spec": "6aba9c8354f201ad41044dd82d4476612f81ebd27a5ce4f8bf4f327a8b3281b9",
    "retrieval_lock": "7587aa140c1cb26f8ac5addaf6873f3bc7ec05b6ec34775a6cbfe98d7ffa40d1",
    "hybrid_union": "71be13c297f6449e6a53c54c7b9db590f6f033d69a11f16b21377797d9390155",
    "reranker_scores": "c94d4208257b2e17cc92f2a9f675c8db67a1ab675f945a0f7ddd96a362f34b49",
    "reranker_lock": "97a8cc18cf2e8c2cf5abec679f41de81546a945b6ae9b36ab06abd97d2508a22",
    "comparison_lock": "c9a33feabf061e9bec373cbf5a5bdfb8b83ac03fd700f9fa55ffdfaa77e5056f",
    "sensitivity_lock": "7d7f446a028d6227f3f3da0b0bd8ee54661add294d2021f0a82313bbf23c8e43",
    "selected_v3_lock": "87484c845fe332882a8ee97eba3824cdc69121f1e10b6614d4532d58fb82e053",
    "v3_reranker": "5e7a7d6b35c0674f32644e758218830d8a340a61588e2ff0b3f20582fdbb0f60",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def req(x: bool, msg: str) -> None:
    if not x:
        raise ValueError(msg)

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        norm = lf_sha256(path)
        req(norm == expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={norm}")
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def main() -> None:
    print("Semantic relevance V4B - design freeze")
    print("--------------------------------------")
    print("Implementation execution: DISABLED")
    print("Retrieval/model/judge execution: DISABLED")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    req(not LOCK.exists(), f"Refusing to overwrite existing V4B design lock: {LOCK}")

    verify_text(SPEC_JSON, EXPECTED["spec_json"], "V4B JSON spec")
    verify_text(SPEC_MD, EXPECTED["spec_md"], "V4B Markdown spec")
    verify_text(V4A_SPEC, EXPECTED["v4a_spec"], "V4A frozen spec")

    for path, key, label in [
        (RETRIEVAL_LOCK, "retrieval_lock", "V4A retrieval lock"),
        (HYBRID_UNION, "hybrid_union", "V4A frozen hybrid union"),
        (RERANKER_SCORES, "reranker_scores", "V4A frozen full-union reranker scores"),
        (RERANKER_LOCK, "reranker_lock", "V4A reranker lock"),
        (COMPARISON_LOCK, "comparison_lock", "V4A comparison/selection lock"),
        (SENSITIVITY_LOCK, "sensitivity_lock", "V4A retrieval-diagnostic sensitivity lock"),
        (SELECTED_V3_LOCK, "selected_v3_lock", "selected V3 candidate lock"),
    ]:
        verify_raw(path, EXPECTED[key], label)
    verify_text(V3_RERANKER, EXPECTED["v3_reranker"], "frozen V3 reranker source")

    spec = load(SPEC_JSON)
    req(spec.get("status") == "PROPOSED_FOR_FREEZE_BEFORE_V4B_IMPLEMENTATION", "Unexpected V4B spec status")
    req(spec["final_v4b_ranking"]["method"] == "reranker_score_only", "V4B ranking method changed")
    req(spec["final_v4b_ranking"]["hybrid_retrieval_rank_has_non_tie_weight"] is False, "Retrieval rank has non-tie V4B weight")
    req(spec["reranker"]["reuse_v4a_reranker_scores"] is True, "V4B no longer reuses frozen reranker scores")
    req(spec["candidate_generation"]["reuse_frozen_hybrid_union"] is True, "V4B no longer reuses frozen hybrid union")

    cmp_lock = load(COMPARISON_LOCK)
    req(cmp_lock.get("status") == "V4A_DEVELOPMENT_COMPARISON_AND_SELECTION_FROZEN", "Unexpected V4A comparison status")
    req(cmp_lock.get("selected_candidate") == "v3b", "V4A development selection no longer v3b")

    sens = load(SENSITIVITY_LOCK)
    req(sens.get("status") == "V4A_RETRIEVAL_DIAGNOSTIC_SENSITIVITY_ANALYSIS_FROZEN", "Unexpected sensitivity lock status")
    req(sens.get("classification_and_transition_invariant_for_all_17_queries") is True, "Retrieval diagnostic sensitivity is not invariant")
    req(sens.get("actual_quarantine_score_assigned") is False, "Quarantine unexpectedly has an assigned score")

    rlock = load(RERANKER_LOCK)
    req(rlock.get("status") == "V4A_DEVELOPMENT_RERANKING_FROZEN_BEFORE_JUDGING", "Unexpected V4A reranker lock status")
    req(rlock.get("candidate_rows") == 2211, "Frozen reranker candidate count changed")
    req(rlock.get("repeated_inference_rank_order_identical") is True, "Frozen reranker repeat-order check changed")

    payload = {
        "status": "V4B_DESIGN_FROZEN_BEFORE_IMPLEMENTATION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "semantic-relevance-hybrid-retrieval-v4b",
        "version": "0.1.0",
        "spec_json_sha256": EXPECTED["spec_json"],
        "spec_md_sha256": EXPECTED["spec_md"],
        "v4a_hybrid_union_sha256": EXPECTED["hybrid_union"],
        "v4a_reranker_scores_sha256": EXPECTED["reranker_scores"],
        "v4a_reranker_lock_sha256": EXPECTED["reranker_lock"],
        "v4a_comparison_selection_lock_sha256": EXPECTED["comparison_lock"],
        "v4a_retrieval_diagnostic_sensitivity_lock_sha256": EXPECTED["sensitivity_lock"],
        "selected_v3_candidate_lock_sha256": EXPECTED["selected_v3_lock"],
        "v3_reranker_source_sha256": EXPECTED["v3_reranker"],
        "ranking_method": "reranker_score_only",
        "non_tie_retrieval_rank_weight": False,
        "reuse_frozen_union": True,
        "reuse_frozen_reranker_scores": True,
        "retrieval_calls_performed": 0,
        "model_calls_performed": 0,
        "judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_selection_calculated": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "IMPLEMENT_DETERMINISTIC_V4B_RANKING_FROM_FROZEN_V4A_RERANKER_SCORES"
    }
    LOCK.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("PASS  V4B hypothesis isolated to final ranking only")
    print("PASS  frozen hybrid union and frozen reranker scores verified for reuse")
    print("PASS  V4A selection remains v3b; retrieval diagnostic sensitivity is invariant")
    print()
    print("V4B DESIGN FREEZE: PASS")
    print(f"V4B JSON spec SHA-256: {sha256(SPEC_JSON)}")
    print(f"V4B Markdown spec SHA-256: {sha256(SPEC_MD)}")
    print(f"V4B design lock SHA-256: {sha256(LOCK)}")
    print("Retrieval calls: 0")
    print("Model calls: 0")
    print("Judge calls: 0")
    print("Development selection calculated: NO")
    print("Release analysis: NO")
    print("Next: deterministic V4B ranking implementation from frozen V4A reranker scores.")

if __name__ == "__main__":
    main()
