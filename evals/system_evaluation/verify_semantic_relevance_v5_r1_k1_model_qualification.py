from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYS = ROOT / "evals/system_evaluation"
CONTRACT = SYS / "semantic_relevance_v5_r1_k1_model_qualification_contract.v0.1.0.json"
ARCH_LOCK = SYS / "semantic_relevance_v5_r1_k1_architecture.v0.1.0.lock.json"
IMPL_LOCK = SYS / "semantic_relevance_v5_r1_k1_implementation.v0.1.0.lock.json"
R1_SOURCE = SYS / "semantic_relevance_v5_r1_retrieval.py"
K1_SOURCE = SYS / "semantic_relevance_v5_k1_reranker.py"
RESULTS = SYS / "semantic_relevance_v5_r1_k1_model_qualification_results.v0.1.0.json"
QUAL_LOCK = SYS / "semantic_relevance_v5_r1_k1_model_qualification.v0.1.0.lock.json"

EXPECTED = {
    "contract": "36bef237ae7b196c39022ce098fc158ee7cd02ffc97393b3984a22113db1194b",
    "arch_lock": "2b298d3263c7a88090a983700da34c30767a71877a4b32ea2e1ae8f3524ade87",
    "impl_lock": "4d166df6b1acff66b23829126ee18c080bd3756de7ed98953e4520db8de9f72b",
    "r1_source": "eabadeccc01a98fb2f962e21ef94c4696e066b78d118f597b83120be4eeaf511",
    "k1_source": "4a6ba7a13d03176108c65512a4185845b532ff5d85f3af512af013b8c8d60207",
}
EXPECTED_VERSIONS = {"sentence-transformers":"6.2.0","transformers":"5.15.0","torch":"2.11.0+cu128"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    b=path.read_bytes()
    return hashlib.sha256(b.replace(b"\r\n",b"\n").replace(b"\r",b"\n")).hexdigest()


def req(cond: bool, msg: str) -> None:
    if not cond: raise ValueError(msg)


def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual=sha256(path)
    req(actual==expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")


def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw=sha256(path)
    if raw==expected: mode="raw"
    else:
        lf=lf_sha256(path)
        req(lf==expected, f"{label} hash mismatch: expected={expected} raw={raw} lf={lf}")
        mode="canonical-LF"
    print(f"PASS  {label} {mode} SHA-256 verified")


def package_version(name: str) -> str:
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc: raise ValueError(f"Required package {name} is not installed") from exc


def verify_boundary(*, require_outputs_absent: bool=True) -> dict:
    verify_raw(CONTRACT, EXPECTED["contract"], "V5 R1/K1 qualification contract")
    verify_raw(ARCH_LOCK, EXPECTED["arch_lock"], "V5 R1/K1 architecture lock")
    verify_raw(IMPL_LOCK, EXPECTED["impl_lock"], "V5 R1/K1 implementation lock")
    verify_text(R1_SOURCE, EXPECTED["r1_source"], "V5 R1 retrieval source")
    verify_text(K1_SOURCE, EXPECTED["k1_source"], "V5 K1 reranker source")

    arch=json.loads(ARCH_LOCK.read_text(encoding="utf-8-sig"))
    impl=json.loads(IMPL_LOCK.read_text(encoding="utf-8-sig"))
    req(arch.get("status")=="V5_R1_K1_ARCHITECTURE_FROZEN_BEFORE_COMPONENT_IMPLEMENTATION", "Unexpected architecture lock status")
    req(impl.get("status")=="V5_R1_K1_IMPLEMENTATION_SOURCE_FROZEN_BEFORE_MODEL_QUALIFICATION", "Unexpected implementation lock status")
    req(impl.get("model_inference_performed") is False, "Implementation freeze says model inference already occurred")
    req(impl.get("v5d_authored_or_executed") is False, "Implementation freeze says V5D was authored/executed")
    req(impl.get("r1",{}).get("model_revision")=="5617a9f61b028005a4858fdac845db406aefb181", "R1 revision changed")
    req(impl.get("k1",{}).get("model_revision")=="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", "K1 revision changed")

    for name, expected in EXPECTED_VERSIONS.items():
        actual=package_version(name)
        req(actual==expected, f"{name} version mismatch: expected={expected} actual={actual}")
        print(f"PASS  {name} version pinned at {expected}")

    c=json.loads(CONTRACT.read_text(encoding="utf-8-sig"))
    req(c.get("status")=="FROZEN_BEFORE_PINNED_MODEL_QUALIFICATION", "Unexpected qualification contract status")
    req(c.get("probe_policy",{}).get("synthetic_only") is True, "Probes are not marked synthetic-only")
    req(c.get("probe_policy",{}).get("not_from_v4h") is True, "Probe boundary does not exclude V4H")
    req(c.get("probe_policy",{}).get("not_from_v5d") is True, "Probe boundary does not exclude V5D")
    req(c.get("probe_policy",{}).get("not_used_for_quality_selection") is True, "Probe boundary permits quality selection")
    qids=[q["probe_query_id"] for q in c["probes"]["queries"]]
    dids=[d["probe_doc_id"] for d in c["probes"]["documents"]]
    req(qids==["P01","P02","P03"], f"Probe query IDs changed: {qids}")
    req(dids==[f"D{i:02d}" for i in range(1,9)], f"Probe document IDs changed: {dids}")
    if require_outputs_absent:
        req(not RESULTS.exists(), f"Refusing to overwrite existing qualification results: {RESULTS}")
        req(not QUAL_LOCK.exists(), f"Refusing to overwrite existing qualification lock: {QUAL_LOCK}")
    return c


def main() -> None:
    print("Semantic relevance V5 - R1/K1 pinned-model qualification preflight")
    print("----------------------------------------------------------------")
    print("Model inference: DISABLED")
    print("V5D authoring/execution: DISABLED")
    print("Judge execution: DISABLED")
    print("Development selection: NOT CALCULATED")
    print()
    c=verify_boundary(require_outputs_absent=True)
    print()
    print("V5 R1/K1 MODEL QUALIFICATION PREFLIGHT: PASS")
    print(f"Synthetic probe queries: {len(c['probes']['queries'])}")
    print(f"Synthetic probe documents: {len(c['probes']['documents'])}")
    print("R1 passes planned: 2")
    print("K1 passes planned: 2")
    print("Model calls performed during preflight: 0")
    print("V5D authored/executed: NO")
    print("Next: execute only the frozen synthetic-probe pinned-model qualification.")

if __name__=="__main__": main()
