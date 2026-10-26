from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "evals/system_evaluation"
ARCH_JSON = SYS / "semantic_relevance_v5_r1_k1_architecture.v0.1.0.json"
ARCH_MD = SYS / "semantic_relevance_v5_r1_k1_architecture.v0.1.0.md"
ARCH_LOCK = SYS / "semantic_relevance_v5_r1_k1_architecture.v0.1.0.lock.json"
R1_SOURCE = SYS / "semantic_relevance_v5_r1_retrieval.py"
R1_TEST = SYS / "test_semantic_relevance_v5_r1_retrieval.py"
K1_SOURCE = SYS / "semantic_relevance_v5_k1_reranker.py"
K1_TEST = SYS / "test_semantic_relevance_v5_k1_reranker.py"
CATALOG = ROOT / "books_with_emotions.csv"
TAGGED = ROOT / "tagged_description.txt"
LOCK = SYS / "semantic_relevance_v5_r1_k1_implementation.v0.1.0.lock.json"

EXPECTED = {
    "arch_json": "7b2c9d7916b62f4b65e01c97b7b2a751b07bbc051af6cf8c5304eafb2f31b1d2",
    "arch_md": "3b4a9fd427c7658775fbc1adb9212cd2a3e67e79728b985848bddbb7bd18e4ec",
    "arch_lock": "2b298d3263c7a88090a983700da34c30767a71877a4b32ea2e1ae8f3524ade87",
    "r1_source": "eabadeccc01a98fb2f962e21ef94c4696e066b78d118f597b83120be4eeaf511",
    "r1_test": "a3eda4bc0fd0984e2e84d154045e16642e9d40ed45072e43f25c69b2a4090f0c",
    "k1_source": "4a6ba7a13d03176108c65512a4185845b532ff5d85f3af512af013b8c8d60207",
    "k1_test": "be4c60545921acb71cdf69e889648aec79403f81f524f9f7835a5462866fee90",
}
EXPECTED_VERSIONS = {
    "sentence-transformers": "6.2.0",
    "transformers": "5.15.0",
    "torch": "2.11.0+cu128",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    b=path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n",b"\n").replace(b"\r",b"\n")).hexdigest()


def req(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw=sha256(path)
    if raw==expected:
        mode="raw"
    else:
        lf=lf_sha256(path)
        req(lf==expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={lf}")
        mode="canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")


def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual=sha256(path)
    req(actual==expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")


def ensure_tracked_unchanged(path: Path, label: str) -> None:
    rel=str(path.relative_to(ROOT)).replace("\\","/")
    result=subprocess.run(["git","diff","--quiet","HEAD","--",rel], cwd=ROOT)
    req(result.returncode==0, f"{label} differs from HEAD; refuse to freeze implementation against a modified corpus.")
    print(f"PASS  {label} unchanged from HEAD")


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ValueError(f"Required package {name} is not installed") from exc


def main() -> None:
    print("Semantic relevance V5 - R1/K1 implementation source freeze")
    print("----------------------------------------------------------")
    print("V5D authoring/execution: DISABLED")
    print("Model/retrieval/reranker inference: DISABLED")
    print("Judge execution: DISABLED")
    print("Development selection: NOT CALCULATED")
    print()

    req(not LOCK.exists(), f"Refusing to overwrite existing implementation lock: {LOCK}")
    verify_text(ARCH_JSON, EXPECTED["arch_json"], "V5 R1/K1 architecture JSON")
    verify_text(ARCH_MD, EXPECTED["arch_md"], "V5 R1/K1 architecture Markdown")
    verify_raw(ARCH_LOCK, EXPECTED["arch_lock"], "V5 R1/K1 architecture lock")
    verify_text(R1_SOURCE, EXPECTED["r1_source"], "V5 R1 retrieval source")
    verify_text(R1_TEST, EXPECTED["r1_test"], "V5 R1 retrieval tests")
    verify_text(K1_SOURCE, EXPECTED["k1_source"], "V5 K1 reranker source")
    verify_text(K1_TEST, EXPECTED["k1_test"], "V5 K1 reranker tests")

    arch=json.loads(ARCH_LOCK.read_text(encoding="utf-8-sig"))
    req(arch.get("status")=="V5_R1_K1_ARCHITECTURE_FROZEN_BEFORE_COMPONENT_IMPLEMENTATION", "Unexpected architecture lock status")
    req(arch.get("r1_model_id")=="BAAI/bge-m3", "Architecture lock R1 model changed")
    req(arch.get("k1_model_id")=="BAAI/bge-reranker-v2-m3", "Architecture lock K1 model changed")
    req(arch.get("v5d_authored_or_executed") is False, "Architecture lock says V5D was authored/executed")

    versions={name:package_version(name) for name in EXPECTED_VERSIONS}
    for name, expected in EXPECTED_VERSIONS.items():
        req(versions[name]==expected, f"{name} version mismatch: expected={expected} actual={versions[name]}")
        print(f"PASS  {name} version pinned at {expected}")

    ensure_tracked_unchanged(CATALOG, "books_with_emotions.csv")
    ensure_tracked_unchanged(TAGGED, "tagged_description.txt")
    catalog_sha=sha256(CATALOG); tagged_sha=sha256(TAGGED)
    print(f"PASS  catalog SHA-256 captured: {catalog_sha}")
    print(f"PASS  tagged-description SHA-256 captured: {tagged_sha}")

    test_result=subprocess.run(
        [sys.executable,"-m","pytest",str(R1_TEST),str(K1_TEST),"-q"],
        cwd=ROOT,capture_output=True,text=True
    )
    print(test_result.stdout.rstrip())
    if test_result.stderr.strip(): print(test_result.stderr.rstrip())
    req(test_result.returncode==0, "V5 R1/K1 implementation unit tests failed")
    print("PASS  V5 R1/K1 implementation unit tests passed")

    payload={
        "status":"V5_R1_K1_IMPLEMENTATION_SOURCE_FROZEN_BEFORE_MODEL_QUALIFICATION",
        "frozen_at_utc":datetime.now(timezone.utc).isoformat(),
        "architecture_lock_sha256":EXPECTED["arch_lock"],
        "r1":{
            "model_id":"BAAI/bge-m3",
            "model_revision":"5617a9f61b028005a4858fdac845db406aefb181",
            "source_sha256":EXPECTED["r1_source"],
            "tests_sha256":EXPECTED["r1_test"],
            "candidate_depth":50,
            "batch_size":16,
            "embedding_dimension":1024,
            "max_sequence_length":512,
            "normalize_embeddings":True,
            "similarity":"dot_product_on_l2_normalized_embeddings",
        },
        "k1":{
            "model_id":"BAAI/bge-reranker-v2-m3",
            "model_revision":"953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "source_sha256":EXPECTED["k1_source"],
            "tests_sha256":EXPECTED["k1_test"],
            "batch_size":4,
            "max_length_tokens":512,
            "score":"single_raw_sequence_classification_logit",
            "ordering":"score_desc_exact_tie_retrieval_rank_then_isbn",
        },
        "dependencies":versions,
        "catalog_sha256":catalog_sha,
        "tagged_description_sha256":tagged_sha,
        "unit_tests_passed":True,
        "model_files_downloaded_or_loaded":False,
        "model_inference_performed":False,
        "retrieval_execution_performed":False,
        "reranker_execution_performed":False,
        "judge_calls_performed":0,
        "v5d_authored_or_executed":False,
        "development_selection_calculated":False,
        "release_metrics_calculated":False,
        "next_stage":"QUALIFY_PINNED_R1_K1_MODEL_REVISIONS_AND_REPEATABILITY_ON_FIXED_NON_V5D_PROBES",
    }
    LOCK.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")

    print()
    print("V5 R1/K1 IMPLEMENTATION SOURCE FREEZE: PASS")
    print(f"R1 source SHA-256: {sha256(R1_SOURCE)}")
    print(f"K1 source SHA-256: {sha256(K1_SOURCE)}")
    print(f"Implementation lock SHA-256: {sha256(LOCK)}")
    print("Model inference performed: NO")
    print("V5D authored/executed: NO")
    print("Next: qualify the pinned model revisions and repeatability on fixed non-V5D probes.")

if __name__=="__main__": main()
