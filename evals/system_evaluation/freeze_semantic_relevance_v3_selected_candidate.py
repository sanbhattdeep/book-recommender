from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
SPEC_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.json"
SPEC_LOCK_FILE = EVALS / "system_evaluation/semantic_relevance_v3_reranker_experiment.v0.1.0.lock.json"
RERANKER_SOURCE = EVALS / "system_evaluation/semantic_relevance_v3_reranker.py"
RUNNER_SOURCE = EVALS / "system_evaluation/run_semantic_relevance_v3_dev.py"
ANALYZER_SOURCE = EVALS / "system_evaluation/analyze_semantic_relevance_v3_dev.py"
DEV_RUN = EVALS / "runs/semantic_relevance_v3_dev/20261007T115716Z_spec_v0_1_0"
RUN_METADATA = DEV_RUN / "v3_dev_reranker_run_metadata.json"
COMPARISON_FILE = DEV_RUN / "v3_dev_variant_comparison.json"
DEV_SELECTION_LOCK = DEV_RUN / "v3_selected_candidate_lock.json"
V3A_FILE = DEV_RUN / "v3_dev_recommendations_v3a.csv"
V3B_FILE = DEV_RUN / "v3_dev_recommendations_v3b.csv"
FINAL_LOCK = EVALS / "system_evaluation/semantic_relevance_v3_selected_candidate.lock.json"

EXPECTED_SPEC_SHA256 = "5bb1f07f68db828b8db1a8d6d1c62831a5309651b7c62c91d04184bf41bb9261"
EXPECTED_RERANKER_SOURCE_SHA256 = "5e7a7d6b35c0674f32644e758218830d8a340a61588e2ff0b3f20582fdbb0f60"
EXPECTED_RUNNER_SOURCE_SHA256 = "acce4f5f1595c4d3dbf818f13db3902c7437d2e827ca6f978a120288fe261180"
EXPECTED_ANALYZER_SOURCE_SHA256 = "c037ad94dd4854b997894a47c6a3af9cdfd0baf749707b14d218ada67ce65a9c"
EXPECTED_V3A_SHA256 = "da79795f088c8c5c339e03a0e819accb42e6eeabc4b05de2a1354036df361eb9"
EXPECTED_V3B_SHA256 = "0dbaf85e557586368bb9d929fdc7d6f56909b64b53c064f02e08dac7db21bc60"
EXPECTED_SELECTED = "v3b_rrf_hybrid"
EXPECTED_MODEL_ID = "BAAI/bge-reranker-base"
EXPECTED_MODEL_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path):
    if not path.exists(): raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(cond, msg):
    if not cond: raise ValueError(msg)

def main():
    print("Semantic relevance v3 - selected candidate freeze")
    print("-------------------------------------------------")
    print("Reranker inference: DISABLED")
    print("Judge calls: 0")
    print("Holdout retrieval: DISABLED")
    print()
    req(sha256(SPEC_FILE)==EXPECTED_SPEC_SHA256, "Reviewed spec hash changed.")
    sl=load_json(SPEC_LOCK_FILE)
    req(sl.get("status")=="FROZEN_BEFORE_V3_DEV_EXECUTION", "Spec lock not frozen.")
    req(sl.get("spec_sha256")==EXPECTED_SPEC_SHA256, "Spec lock SHA mismatch.")
    req(sha256(RERANKER_SOURCE)==EXPECTED_RERANKER_SOURCE_SHA256, "Reranker source changed.")
    req(sha256(RUNNER_SOURCE)==EXPECTED_RUNNER_SOURCE_SHA256, "Runner source changed.")
    req(sha256(ANALYZER_SOURCE)==EXPECTED_ANALYZER_SOURCE_SHA256, "Analyzer source changed.")
    print("PASS  v3 implementation source hashes unchanged")
    md=load_json(RUN_METADATA)
    req(md.get("status")=="V3_DEV_RERANKER_EXECUTION_COMPLETE", "Unexpected dev run status.")
    req(md.get("judge_labels_loaded") is False and md.get("human_relevance_labels_loaded") is False, "Label-blind boundary failed.")
    req(int(md.get("judge_calls_performed",-1))==0, "Judge calls occurred.")
    req(md.get("repeated_inference_rank_order_identical") is True, "Reproducibility failed.")
    req(float(md.get("max_absolute_score_delta_between_passes",1))==0.0, "Repeated raw scores changed.")
    req(md.get("model_id")==EXPECTED_MODEL_ID and md.get("model_revision")==EXPECTED_MODEL_REVISION, "Model provenance changed.")
    req(sha256(V3A_FILE)==EXPECTED_V3A_SHA256, "v3a artifact changed.")
    req(sha256(V3B_FILE)==EXPECTED_V3B_SHA256, "v3b artifact changed.")
    print("PASS  v3a/v3b recommendation hashes unchanged")
    comp=load_json(COMPARISON_FILE)
    req(comp.get("status")=="V3_DEV_VARIANT_COMPARISON_COMPLETE", "Comparison incomplete.")
    req(comp.get("analysis_role")=="development_only" and comp.get("release_decision") is None, "Development-only boundary failed.")
    req(int(comp.get("judge_calls_performed",-1))==0, "Analysis judge calls occurred.")
    req(comp.get("selected_variant")==EXPECTED_SELECTED, "Selected variant changed.")
    dsl=load_json(DEV_SELECTION_LOCK)
    req(dsl.get("status")=="V3_DEV_CANDIDATE_SELECTED_BEFORE_UNSEEN_HOLDOUT", "Selection lock status changed.")
    req(dsl.get("selected_variant")==EXPECTED_SELECTED, "Selection lock variant changed.")
    spec=load_json(SPEC_FILE)
    fusion=spec["variants"]["v3b_rrf_hybrid"]["fusion"]
    req(float(fusion.get("k"))==60.0 and float(fusion.get("vector_weight"))==1.0 and float(fusion.get("reranker_weight"))==1.0, "Frozen RRF policy changed.")
    payload={
      "status":"V3_SELECTED_CANDIDATE_FROZEN_BEFORE_UNSEEN_HOLDOUT_AUTHORING",
      "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
      "experiment_id":"semantic-relevance-reranker-v3",
      "spec_sha256":EXPECTED_SPEC_SHA256,
      "selected_variant":EXPECTED_SELECTED,
      "selection_source_sha256":sha256(DEV_SELECTION_LOCK),
      "development_comparison_sha256":sha256(COMPARISON_FILE),
      "development_run_metadata_sha256":sha256(RUN_METADATA),
      "model_id":EXPECTED_MODEL_ID,
      "model_revision":EXPECTED_MODEL_REVISION,
      "ranking_policy":{"method":"reciprocal_rank_fusion","candidate_depth":50,"display_depth":10,"rrf_k":60,"vector_weight":1.0,"reranker_weight":1.0},
      "source_hashes":{"semantic_relevance_v3_reranker.py":EXPECTED_RERANKER_SOURCE_SHA256,"run_semantic_relevance_v3_dev.py":EXPECTED_RUNNER_SOURCE_SHA256,"analyze_semantic_relevance_v3_dev.py":EXPECTED_ANALYZER_SOURCE_SHA256},
      "development_output_hashes":{"v3a_recommendations_sha256":EXPECTED_V3A_SHA256,"v3b_recommendations_sha256":EXPECTED_V3B_SHA256},
      "development_release_decision_produced":False,
      "unseen_holdout_required":True,
      "holdout_must_be_authored_and_frozen_before_retrieval":True,
      "judge_calls_performed_during_freeze":0,
      "reranker_inference_performed_during_freeze":False
    }
    if FINAL_LOCK.exists():
        old=load_json(FINAL_LOCK)
        for k,v in payload.items():
            if k=="frozen_at_utc": continue
            req(old.get(k)==v, f"Existing final lock mismatch: {k}")
        print("PASS  existing selected-candidate freeze lock compatible; not rewritten")
    else:
        FINAL_LOCK.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
        print(f"PASS  wrote selected-candidate freeze lock: {FINAL_LOCK.relative_to(REPO_ROOT)}")
    print()
    print("V3 SELECTED CANDIDATE: FROZEN")
    print("Selected variant: v3b_rrf_hybrid")
    print("RRF k: 60; weights: vector=1.0 reranker=1.0")
    print("Reranker inference performed: NO")
    print("Judge calls performed: 0")
    print("Next: author and freeze the unseen 24-query holdout before any holdout retrieval.")

if __name__ == "__main__":
    main()
