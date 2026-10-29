from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"
SYS = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v5_holdout/20261028T055522Z_retrieval"

if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, validate_query_facet_spec

CONTRACT = SYS / "semantic_relevance_v5_holdout_judge_execution_contract.v0.1.0.json"
EXEC_PATCH = SYS / "semantic_relevance_v5_holdout_judge_execution_patch.nonsemantic-normalizations1.json"
JINPUT_CONTRACT = SYS / "semantic_relevance_v5_holdout_judging_input_contract.v0.1.0.json"
SCORE_INVENTORY = RUN / "v5h_display_score_inventory.csv"
NOVEL_INPUT = RUN / "v5h_novel_display_pairs_to_judge.csv"
JINPUT_LOCK = RUN / "v5h_judging_input.lock.json"
DISPLAY_UNION = RUN / "v5h_displayed_pair_union.csv"
DISPLAY_LOCK = RUN / "v5h_displayed_pair_union.lock.json"
FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v5_holdout_query_facets.v0.1.0.json"
AUTHORING_LOCK = SYS / "semantic_relevance_v5_holdout_authoring.v0.1.0.lock.json"

HISTORICAL_SCORING_LOCK = (
    EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval/holdout_judge_scoring_lock.json"
)
JUDGE_SOURCE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_SOURCE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"

SPAN_PATCH = SYS / "semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
DUP_PATCH = SYS / "semantic_relevance_v4h_execution_patch.duplicate-candidate-id-normalization1.json"

SCORES = RUN / "v5h_judge_scores.csv"
METADATA = RUN / "v5h_judging_metadata.json"
SCORE_LOCK = RUN / "v5h_judge_scoring.lock.json"

EXPECTED = {
    "contract": "454d78668ffef002f0f3cf67a182f20f252d18b837354ef0e3d9bed640240fcb",
    "exec_patch": "5330a4bd7fbcbdbfb278fd9459a980452db95b72254bbe5f646643f4d514c589",
    "jinput_contract": "5c7a28bf5f5a902bfcdf75e7b30c1d0ad40f9b36969dc802c7ceb3588f5dbf94",
    "inventory": "cc350ec54f519ff710f34d24a4aedf45ddb2910f52a7a46276f9a783ecb38e20",
    "novel": "fc046bfef77af201aa7fe91bef6649372292b4a277929b0b99f5311a1666feab",
    "jinput_lock": "a2ff301aef1788526d51858394ee4249eeb548fc05981e0246a50a2a8643eea7",
    "display_union": "13ff542ecfe5316484b62eb0f1b406f9f8f8dd4a7491acc369128a5285d6ff67",
    "display_lock": "0a9e7981cb20584b0074af58cd9b7a2e7a27a6689c75a6f685db7bb3617c96b0",
    "facet": "0a3e55b66568bf89aa7cd212d09cec91803ad1d145bf815783622c1fd89c1d73",
    "authoring": "e4836060ba90e49fbf4da05bf12a15246c312bd7926e48ff72d3766f74613751",
    "historical_scoring_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "span_patch": "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266",
    "dup_patch": "09b2acb1d3809da6ed4667d1e3f7b042dc86f85b6f69b732bb9017136f0b2b94",
}
IDS = [f"V5H{i:02d}" for i in range(1, 25)]
CASE_COUNT = 359


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lf_sha256(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    req(path.exists(), f"Missing JSON input: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")


def verify_text(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing {label}: {path}")
    raw = sha256(path)
    normalized = lf_sha256(path)
    req(
        raw == expected or normalized == expected,
        f"{label} hash mismatch: expected={expected} raw={raw} lf={normalized}",
    )
    print(f"PASS  {label} {'raw' if raw == expected else 'canonical-LF'} SHA-256 verified")


def load_facets() -> dict[str, QueryFacetSpec]:
    payload = load_json(FACET_FILE)
    specs: dict[str, QueryFacetSpec] = {}
    for item in payload.get("queries", []):
        spec = QueryFacetSpec(
            query_id=str(item["query_id"]),
            query=str(item["query"]),
            facets=[QueryFacet(**x) for x in item["facets"]],
        )
        validate_query_facet_spec(spec)
        for facet in spec.facets:
            req(bool(facet.required_components), f"{spec.query_id}/{facet.facet_id}: missing required components")
        req(spec.query_id not in specs, f"Duplicate facet spec query ID {spec.query_id}")
        specs[spec.query_id] = spec
    req(sorted(specs) == IDS, "V5H facet IDs changed")
    return specs


def verify_instrument() -> dict[str, Any]:
    verify_raw(HISTORICAL_SCORING_LOCK, EXPECTED["historical_scoring_lock"], "historical frozen judge scoring lock")
    hlock = load_json(HISTORICAL_SCORING_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS", "Unexpected historical scoring-lock status")
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Frozen judge candidate changed")
    req(hlock.get("human_relevance_labels_used") is False, "Historical scoring lock says human labels were used")

    for path, key, label in [
        (JUDGE_SOURCE, "judge_source_sha256", "frozen judge source"),
        (SCORING_SOURCE, "scoring_source_sha256", "frozen deterministic scoring source"),
        (JUDGE_CONFIG, "judge_config_sha256", "frozen judge config"),
        (RUBRIC, "rubric_sha256", "frozen rubric"),
    ]:
        expected = str(hlock.get(key, "")).strip()
        req(bool(expected), f"Historical scoring lock missing {key}")
        verify_text(path, expected, label)

    config = load_json(JUDGE_CONFIG)
    req(config.get("version") == "0.29.0-r5", "Judge config version changed")
    req(config.get("provider") == "ollama", "Judge provider changed")
    req(float(config.get("temperature")) == 0.0, "Judge temperature changed")

    verify_text(SPAN_PATCH, EXPECTED["span_patch"], "accepted span-echo normalization patch")
    verify_text(DUP_PATCH, EXPECTED["dup_patch"], "accepted duplicate-candidate-ID normalization patch")
    verify_raw(EXEC_PATCH, EXPECTED["exec_patch"], "V5H non-semantic execution carry-forward patch")
    ep = load_json(EXEC_PATCH)
    req(ep.get("status") == "FROZEN_NON_SEMANTIC_EXECUTION_AMENDMENT", "Unexpected V5H execution-patch status")
    for key in [
        "judge_source_modified", "judge_prompt_modified", "judge_model_modified",
        "judge_temperature_modified", "semantic_rules_changed",
        "deterministic_scoring_rules_changed", "facet_spec_changed",
        "judge_config_changed", "rubric_changed", "displayed_pair_union_changed",
        "judge_input_changed", "human_relevance_labels_used",
        "release_metrics_or_gates_changed", "release_decision_modified",
    ]:
        req(ep.get(key) is False, f"Non-semantic execution patch changed forbidden boundary: {key}")

    load_facets()
    print("PASS  frozen semantic_relevance_v0.29.0-r5 measurement instrument verified")
    print("PASS  accepted non-semantic output-format normalizations verified")
    return config


def validate(*, allow_existing_scores: bool = False) -> tuple[pd.DataFrame, dict[str, QueryFacetSpec], dict[str, Any]]:
    verify_raw(CONTRACT, EXPECTED["contract"], "V5H judge-execution contract")
    verify_raw(JINPUT_CONTRACT, EXPECTED["jinput_contract"], "V5H judging-input contract")
    verify_raw(SCORE_INVENTORY, EXPECTED["inventory"], "V5H frozen score inventory")
    verify_raw(NOVEL_INPUT, EXPECTED["novel"], "V5H frozen novel judging input")
    verify_raw(JINPUT_LOCK, EXPECTED["jinput_lock"], "V5H judging-input lock")
    verify_raw(DISPLAY_UNION, EXPECTED["display_union"], "V5H displayed-pair union")
    verify_raw(DISPLAY_LOCK, EXPECTED["display_lock"], "V5H displayed-pair-union lock")
    verify_raw(FACET_FILE, EXPECTED["facet"], "V5H facet specification")
    verify_raw(AUTHORING_LOCK, EXPECTED["authoring"], "V5H authoring lock")

    lock = load_json(JINPUT_LOCK)
    req(lock.get("status") == "V5H_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION", "Unexpected V5H judging-input-lock status")
    req(int(lock.get("case_count", -1)) == CASE_COUNT, "V5H judging-input case count changed")
    req(lock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "V5H judging-input judge candidate changed")
    req(lock.get("score_inventory_sha256") == EXPECTED["inventory"], "Judging-input lock inventory binding changed")
    req(lock.get("novel_judging_input_sha256") == EXPECTED["novel"], "Judging-input lock novel-input binding changed")
    req(int(lock.get("reused_frozen_scores", -1)) == 0, "V5H judging-input unexpectedly reuses historical scores")
    req(int(lock.get("fresh_pairs_to_judge", -1)) == CASE_COUNT, "V5H fresh-pair count changed")
    req(lock.get("judge_execution_performed") is False, "Judging-input lock says judge already ran")
    req(lock.get("holdout_metrics_calculated") is False, "Holdout metrics already calculated")
    req(lock.get("release_gates_calculated") is False, "Release gates already calculated")
    req(lock.get("release_decision_produced") is False, "Release decision already produced")

    frame = pd.read_csv(NOVEL_INPUT, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "case_id", "query_id", "query", "stratum", "isbn13", "title", "description",
        "authors", "in_B", "B_rank", "in_K", "K_rank", "system_count",
    }
    req(not(required - set(frame.columns)), f"Novel judge input missing columns: {sorted(required-set(frame.columns))}")
    req(len(frame) == CASE_COUNT, f"Expected {CASE_COUNT} novel cases, got {len(frame)}")
    req(not frame["case_id"].duplicated().any(), "Duplicate V5H judge case_id")
    req(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate V5H query-book pair")
    req(sorted(frame["query_id"].unique()) == IDS, "V5H novel input query IDs changed")
    req(frame["description"].str.strip().ne("").all(), "V5H novel input contains empty description")
    counts = frame.groupby("query_id").size()
    req(int(counts.min()) == 12 and int(counts.max()) == 18, "V5H novel cases/query range changed from 12..18")

    specs = load_facets()
    for qid, g in frame.groupby("query_id", sort=True):
        req(g["query"].nunique() == 1, f"{qid}: multiple query texts")
        req(str(g["query"].iloc[0]) == specs[str(qid)].query, f"{qid}: query/facet text mismatch")

    config = verify_instrument()

    c = load_json(CONTRACT)
    req(c.get("status") == "FROZEN_BY_PACKAGE_BEFORE_V5H_JUDGE_EXECUTION", "Unexpected V5H judge-execution contract status")
    req(int(c.get("case_count", -1)) == CASE_COUNT, "Execution contract case count changed")
    req(c.get("measurement_instrument", {}).get("judge_candidate") == "semantic_relevance_v0.29.0-r5", "Execution contract judge candidate changed")
    eb = c.get("execution_boundary", {})
    req(eb.get("retrieval_calls") == 0 and eb.get("reranker_inference") is False, "Retrieval/reranking unexpectedly enabled")
    req(eb.get("historical_score_reuse") is False, "Historical score reuse unexpectedly enabled")
    req(eb.get("judge_execution") is True, "Judge execution not enabled by contract")
    req(eb.get("human_relevance_labels") is False, "Human relevance labels unexpectedly enabled")
    req(eb.get("holdout_metrics") is False and eb.get("release_metrics") is False, "Metrics unexpectedly enabled")
    req(eb.get("release_gates") is False and eb.get("release_decision") is False, "Release analysis unexpectedly enabled")

    if not allow_existing_scores:
        for p in [SCORES, METADATA, SCORE_LOCK]:
            req(not p.exists(), f"Refusing preflight because V5H judge output already exists: {p}")
    else:
        req(not SCORE_LOCK.exists(), "Final V5H judge lock already exists; execution is complete")

    return frame, specs, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing-scores", action="store_true")
    args = parser.parse_args()

    print("Semantic relevance V5H - v0.29.0-r5 judge-execution preflight")
    print("-------------------------------------------------------------")
    print("Frozen cases: 359")
    print("Resume-safe persistence: ENABLED")
    print("Retrieval/reranking: DISABLED")
    print("Historical score reuse: NO")
    print("Human relevance labels: NOT USED")
    print("Holdout/release metrics and gates: NOT CALCULATED")
    print()
    frame, _, config = validate(allow_existing_scores=args.allow_existing_scores)
    counts = frame.groupby("query_id").size()

    print()
    print("V5H JUDGE-EXECUTION PREFLIGHT: PASS")
    print(f"Frozen input cases: {len(frame)}")
    print(f"Cases/query range: {int(counts.min())}..{int(counts.max())}")
    print("Judge candidate: semantic_relevance_v0.29.0-r5")
    print(f"Ollama model: {config.get('model')}")
    print("Temperature: 0.0")
    print("Known non-semantic format repairs carried from start: 2")
    print("Judge calls performed during preflight: 0")
    print("Retrieval/reranker calls performed: 0")
    print("Human relevance labels used: NO")
    print("Holdout metrics calculated: NO")
    print("Release gates calculated: NO")
    print("Release decision produced: NO")
    print("Next: run a 3-case structural smoke test before full/resume execution.")


if __name__ == "__main__":
    main()
