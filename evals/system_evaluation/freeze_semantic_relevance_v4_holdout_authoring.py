from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYS = REPO_ROOT / "evals/system_evaluation"
FACETS = REPO_ROOT / "evals/facets/semantic_relevance"
RUN = REPO_ROOT / "evals/runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"

QUERY_FILE = SYS / "semantic_relevance_v4_holdout_queries.v0.1.0.json"
FACET_FILE = FACETS / "semantic_relevance_v4_holdout_query_facets.v0.1.0.json"
SELECTION_LOCK = RUN / "v4b_dev_comparison_selection.lock.json"
PRIOR_MANIFEST = SYS / "semantic_relevance_prior_query_texts_for_v4h.v0.1.0.json"
LOCK = SYS / "semantic_relevance_v4_holdout_authoring.lock.json"
EXPECTED_SELECTION_LOCK = "98ab3a62b978b922dc4b6e1bb031bdc6578a007a93de7ae84b20d36a80c9c9b4"
EXPECTED_PRIOR_MANIFEST = "f4b3f79da07c577b96967db6ff064cb95a9aaa46b06de44c5ad76f65a8009325"

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def req(c: bool,m: str)->None:
    if not c: raise ValueError(m)

def main()->None:
    print("Semantic relevance V4H - freeze unseen holdout authoring")
    print("------------------------------------------------------")
    print("Holdout retrieval: DISABLED")
    print("Judge/model execution: DISABLED")
    print("Release analysis: DISABLED")
    print()
    req(not LOCK.exists(), f"Refusing to overwrite existing V4H authoring lock: {LOCK}")
    req(SELECTION_LOCK.exists() and sha256(SELECTION_LOCK)==EXPECTED_SELECTION_LOCK,
        "Frozen V4B selection lock changed")
    req(PRIOR_MANIFEST.exists() and sha256(PRIOR_MANIFEST)==EXPECTED_PRIOR_MANIFEST,
        "Frozen prior-query anti-leakage manifest changed")
    q=json.loads(QUERY_FILE.read_text(encoding="utf-8-sig"))
    f=json.loads(FACET_FILE.read_text(encoding="utf-8-sig"))
    req(len(q["queries"])==24 and len(f["queries"])==24, "V4H query count changed")
    req([x["query_id"] for x in q["queries"]]==[f"V4H{i:02d}" for i in range(1,25)],
        "V4H IDs changed")
    req([(x["query_id"],x["query"],x["stratum"]) for x in q["queries"]] ==
        [(x["query_id"],x["query"],x["stratum"]) for x in f["queries"]],
        "Query/facet files disagree")
    payload={
        "status":"V4H_QUERY_AND_FACET_SPEC_FROZEN_BEFORE_RETRIEVAL",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "source_v4b_selection_lock_sha256":EXPECTED_SELECTION_LOCK,
        "selected_candidate":"v4b",
        "prior_query_anti_leakage_manifest_sha256":EXPECTED_PRIOR_MANIFEST,
        "query_set":"V4H01-V4H24",
        "query_count":24,
        "query_file_sha256":sha256(QUERY_FILE),
        "facet_file_sha256":sha256(FACET_FILE),
        "strata_counts":{
            "two_core_theme":4,
            "three_core_conjunctive":4,
            "core_plus_qualifier":4,
            "relationship_or_role_constraint":4,
            "hard_exclusion_or_contrast":4,
            "abstract_nonfiction_multi_facet":4,
        },
        "catalog_or_v4h_retrieval_outputs_inspected_during_authoring":False,
        "v4h_judge_outputs_inspected_during_authoring":False,
        "holdout_retrieval_performed":False,
        "judge_execution_performed":False,
        "human_relevance_labels_used":False,
        "release_metrics_calculated":False,
        "next_stage":"PREPARE_V4H_FROZEN_V3B_AND_V4B_RETRIEVAL_PACKAGE",
    }
    LOCK.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    print("PASS  V4H01-V4H24 query/facet artifacts frozen before retrieval")
    print(f"Query file SHA-256: {sha256(QUERY_FILE)}")
    print(f"Facet file SHA-256: {sha256(FACET_FILE)}")
    print(f"Authoring lock SHA-256: {sha256(LOCK)}")
    print("Holdout retrieval performed: NO")
    print("Judge calls performed: 0")
    print("Release metrics calculated: NO")
    print("Next: prepare frozen V3B and V4B retrieval on V4H01-V4H24.")

if __name__=="__main__":
    main()
