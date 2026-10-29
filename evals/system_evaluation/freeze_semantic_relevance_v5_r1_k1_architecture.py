from __future__ import annotations
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "evals/system_evaluation"
V4H_RUN = ROOT / "evals/runs/semantic_relevance_v4_holdout/20261022T103659Z_retrieval"

SPEC_JSON = SYS / "semantic_relevance_v5_r1_k1_architecture.v0.1.0.json"
SPEC_MD = SYS / "semantic_relevance_v5_r1_k1_architecture.v0.1.0.md"
LOCK = SYS / "semantic_relevance_v5_r1_k1_architecture.v0.1.0.lock.json"
V5_DESIGN_JSON = SYS / "semantic_relevance_v5_development_design.v0.1.0.json"
V5_DESIGN_MD = SYS / "semantic_relevance_v5_development_design.v0.1.0.md"
V5_DESIGN_LOCK = SYS / "semantic_relevance_v5_development_design.v0.1.0.lock.json"
V4B_RANKING = SYS / "semantic_relevance_v4b_ranking.py"
V4B_IMPL_LOCK = SYS / "semantic_relevance_v4b_implementation.lock.json"
V4H_CLOSEOUT_LOCK = V4H_RUN / "v4h_headroom_diagnostic_closeout.lock.json"

EXPECTED = {
    "spec_json": "7b2c9d7916b62f4b65e01c97b7b2a751b07bbc051af6cf8c5304eafb2f31b1d2",
    "spec_md": "3b4a9fd427c7658775fbc1adb9212cd2a3e67e79728b985848bddbb7bd18e4ec",
    "v5_design_json": "ef21af8375d3606cf53038f6986a8cd7e3c389fee8345770a39aff73d4cc4551",
    "v5_design_md": "81e896a4b3c74c3a026c5c7379a651e244705064d5cc2499041630517accc7e5",
    "v5_design_lock": "89ee03a4328c268a8128d5e373854b92634c41eff204b53f7b93b78c481311c0",
    "v4b_ranking": "5729b078cee1a2706a8c3d9bcaa9e8449292bc1f9f1a2816053b84758db7d677",
    "v4b_impl_lock": "f30a61e7ca7a5e07766c3f2baa3d4a65abcedb48cdedf23ee6d5c0e05b3171b0",
    "v4h_closeout_lock": "df16cc592716337c5eb27bdf2610e7ba94ebb2c12ec4bb70ca0f875d88f96dad",
}

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def lf_sha256(path: Path) -> str:
    b = path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(cond: bool, msg: str) -> None:
    if not cond: raise ValueError(msg)

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw = sha256(path)
    if raw == expected:
        mode = "raw"
    else:
        normalized = lf_sha256(path)
        req(
            normalized == expected,
            f"{label} hash mismatch: expected={expected} raw={raw} lf={normalized}",
        )
        mode = "canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")

def main() -> None:
    print("Semantic relevance V5 - R1/K1 architecture freeze")
    print("-------------------------------------------------")
    print("V5D authoring/execution: DISABLED")
    print("Retrieval/model/reranker/judge calls: DISABLED")
    print("Development selection: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

    req(not LOCK.exists(), f"Refusing to overwrite existing architecture lock: {LOCK}")
    for p, k, l in [
        (SPEC_JSON, "spec_json", "V5 R1/K1 JSON architecture spec"),
        (SPEC_MD, "spec_md", "V5 R1/K1 Markdown architecture spec"),
        (V5_DESIGN_JSON, "v5_design_json", "V5 development JSON design spec"),
        (V5_DESIGN_MD, "v5_design_md", "V5 development Markdown design spec"),
        (V4B_RANKING, "v4b_ranking", "V4B ranking source"),
    ]:
        verify_text(p, EXPECTED[k], l)

    for p, k, l in [
        (V5_DESIGN_LOCK, "v5_design_lock", "V5 development design lock"),
        (V4B_IMPL_LOCK, "v4b_impl_lock", "V4B implementation lock"),
        (V4H_CLOSEOUT_LOCK, "v4h_closeout_lock", "V4H headroom diagnostic closeout lock"),
    ]:
        verify_raw(p, EXPECTED[k], l)

    design=load(V5_DESIGN_LOCK)
    req(design.get("status")=="V5_DEVELOPMENT_DESIGN_FROZEN_BEFORE_COMPONENT_IMPLEMENTATION", "Unexpected V5 design lock status")
    req(design.get("v4h_tuning_allowed") is False, "V5 design unexpectedly permits V4H tuning")
    req(design.get("r1_max_unique_candidate_budget")==100, "V5 candidate budget changed")

    spec=load(SPEC_JSON)
    req(spec.get("status")=="PROPOSED_FOR_FREEZE_BEFORE_V5_COMPONENT_IMPLEMENTATION", "Unexpected architecture spec status")
    r1=spec["R1"]; k1=spec["K1"]
    req(r1["change_class"]=="dense_retriever_model_only", "R1 change class changed")
    req(r1["dense_branch"]["model_id"]=="BAAI/bge-m3", "R1 model changed")
    req(r1["dense_branch"]["model_revision"]=="5617a9f61b028005a4858fdac845db406aefb181", "R1 model revision changed")
    req(r1["dense_branch"]["candidate_depth"]==50, "R1 dense depth changed")
    req(r1["lexical_branch"]["changed"] is False, "R1 lexical branch unexpectedly changed")
    req(r1["union"]["maximum_unique_candidates"]==100, "R1 candidate budget changed")
    req(r1["union"]["truncate_before_reranker"] is False, "R1 union truncation changed")

    req(k1["change_class"]=="cross_encoder_model_only", "K1 change class changed")
    req(k1["model"]["model_id"]=="BAAI/bge-reranker-v2-m3", "K1 model changed")
    req(k1["model"]["model_revision"]=="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "K1 model revision changed")
    req(k1["model"]["max_length_tokens"]==512, "K1 max length changed")
    req(k1["score_and_order"]["score"]=="single raw sequence-classification logit", "K1 score rule changed")
    req(spec["implementation_freeze_requirements"]["freeze_before"]=="V5D query authoring and any V5D retrieval/model execution", "Implementation-freeze ordering changed")

    payload={
        "status":"V5_R1_K1_ARCHITECTURE_FROZEN_BEFORE_COMPONENT_IMPLEMENTATION",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "architecture_id":spec["architecture_id"],
        "version":spec["version"],
        "spec_json_sha256":EXPECTED["spec_json"],
        "spec_md_sha256":EXPECTED["spec_md"],
        "v5_design_lock_sha256":EXPECTED["v5_design_lock"],
        "v4b_ranking_sha256":EXPECTED["v4b_ranking"],
        "v4b_implementation_lock_sha256":EXPECTED["v4b_impl_lock"],
        "v4h_headroom_closeout_lock_sha256":EXPECTED["v4h_closeout_lock"],
        "r1_model_id":"BAAI/bge-m3",
        "r1_model_revision":"5617a9f61b028005a4858fdac845db406aefb181",
        "r1_dense_candidate_depth":50,
        "r1_max_unique_candidate_budget":100,
        "k1_model_id":"BAAI/bge-reranker-v2-m3",
        "k1_model_revision":"953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
        "k1_max_length_tokens":512,
        "v5d_authored_or_executed":False,
        "retrieval_calls_performed":0,
        "model_calls_performed":0,
        "reranker_calls_performed":0,
        "judge_calls_performed":0,
        "development_selection_calculated":False,
        "release_metrics_calculated":False,
        "next_stage":spec["next_stage"],
    }
    LOCK.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")

    print("PASS  V5 development design boundary verified")
    print("PASS  R1 frozen as BGE-M3 dense-model replacement only")
    print("PASS  R1 BM25/depth/fusion/candidate-budget mechanics remain fixed")
    print("PASS  K1 frozen as bge-reranker-v2-m3 cross-encoder replacement only")
    print("PASS  K1 score-only V4B ordering policy remains fixed")
    print("PASS  V5D remains unauthored and unexecuted")
    print()
    print("V5 R1/K1 ARCHITECTURE FREEZE: PASS")
    print(f"R1/K1 JSON spec SHA-256: {sha256(SPEC_JSON)}")
    print(f"R1/K1 Markdown spec SHA-256: {sha256(SPEC_MD)}")
    print(f"Architecture lock SHA-256: {sha256(LOCK)}")
    print("Retrieval/model/reranker/judge calls performed: 0")
    print("Development selection calculated: NO")
    print("Release metrics calculated: NO")
    print("Next: implement and freeze exact R1 and K1 source + repeatability tests before V5D authoring.")

if __name__ == "__main__": main()
