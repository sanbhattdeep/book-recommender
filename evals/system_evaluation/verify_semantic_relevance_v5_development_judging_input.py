from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"
SYS = EVALS / "system_evaluation"
RUN = EVALS / "runs/semantic_relevance_v5_development/20261026T124615Z_candidate_collection"

if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from semantic_relevance_facet_scoring import (
    QueryFacet,
    QueryFacetSpec,
    validate_query_facet_spec,
)

CONTRACT = SYS / "semantic_relevance_v5_development_judging_input_contract.v0.1.0.json"
DISPLAY_UNION = RUN / "v5d_displayed_pair_union.csv"
DISPLAY_LOCK = RUN / "v5d_displayed_pair_union.lock.json"
DISPLAY_CONTRACT = SYS / "semantic_relevance_v5_development_display_union_contract.v0.1.0.json"

AUTHORING_LOCK = SYS / "semantic_relevance_v5_development_authoring.v0.1.0.lock.json"
FACET_FILE = EVALS / "facets/semantic_relevance/semantic_relevance_v5_development_query_facets.v0.1.0.json"
PRIOR_QUERY_MANIFEST = SYS / "semantic_relevance_prior_query_texts_for_v5d.v0.1.0.json"

HISTORICAL_SCORING_LOCK = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval/holdout_judge_scoring_lock.json"
JUDGE_FILE = EVALS / "semantic_relevance_facet_judge.py"
SCORING_FILE = EVALS / "semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE = EVALS / "judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE = EVALS / "rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE = SYS / "semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"

OUTPUTS = [
    RUN / "v5d_display_score_inventory.csv",
    RUN / "v5d_novel_display_pairs_to_judge.csv",
    RUN / "v5d_judging_input_metadata.json",
    RUN / "v5d_judging_input.lock.json",
]

LATER_JUDGE_OUTPUTS = [
    RUN / "v5d_judge_scores.csv",
    RUN / "v5d_judging_metadata.json",
    RUN / "v5d_judge_scoring.lock.json",
]

EXPECTED = {
    "contract": "d2e013323ddfc8d87154277c79165d242038b29978b6fc20243ebcd6bf5a69dd",
    "display_contract": "ad66b21f6177da1338b961092347df7838872fd76dcc1da07a496ed780fa2e16",
    "display_union": "00fac6ed404e10b25779848f5ced12c6cfc48b6a4f88080364d8b0647b6bbe21",
    "display_lock": "145ebd730b1444399c27706071ec392a1f7da3f12688927312c039bb59f60117",
    "authoring_lock": "ffca0f7a10c4d05ee79fc08d584df5793f93de2242a2ed2bf09436e2df80ba3e",
    "facet": "355ac59604fc73039fe394ba36b9ef37c04cd0be447ff470bce669d8adeb4746",
    "prior_manifest": "32015f2426c1d72b2eb65db0e11c4be7ffdf390e14e00e00f29d0307e9f89056",
    "historical_scoring_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "span_echo_patch": "85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266",
}
EXPECTED_IDS = [f"V5D{i:02d}" for i in range(1, 25)]
EXPECTED_PAIRS = 452


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
    req(raw == expected or normalized == expected,
        f"{label} hash mismatch: expected={expected} raw={raw} lf={normalized}")
    print(f"PASS  {label} {'raw' if raw == expected else 'canonical-LF'} SHA-256 verified")


def canonical_isbn(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    req(bool(text), "ISBN is empty after normalization")
    return text


def load_and_validate_v5d_facets() -> dict[str, QueryFacetSpec]:
    payload = load_json(FACET_FILE)
    raw_queries = payload.get("queries")
    req(isinstance(raw_queries, list), "V5D facet specification must contain a queries list")
    specs: dict[str, QueryFacetSpec] = {}
    for raw in raw_queries:
        qid = str(raw["query_id"])
        req(qid not in specs, f"Duplicate V5D facet query ID: {qid}")
        spec = QueryFacetSpec(
            query_id=qid,
            query=str(raw["query"]),
            facets=[QueryFacet(**item) for item in raw["facets"]],
        )
        validate_query_facet_spec(spec)
        for facet in spec.facets:
            req(bool(facet.required_components), f"{qid}/{facet.facet_id}: missing required components")
            component_ids = [c.component_id for c in facet.required_components]
            req(len(component_ids) == len(set(component_ids)),
                f"{qid}/{facet.facet_id}: duplicate required-component IDs")
        specs[qid] = spec
    req(sorted(specs) == EXPECTED_IDS, "V5D facet query IDs changed")
    return specs


def verify_measurement_instrument() -> None:
    verify_raw(HISTORICAL_SCORING_LOCK, EXPECTED["historical_scoring_lock"],
               "historical frozen judge scoring lock")
    hlock = load_json(HISTORICAL_SCORING_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS",
        "Unexpected historical scoring-lock status")
    req(hlock.get("judge_candidate") == "semantic_relevance_v0.29.0-r5",
        "Frozen judge candidate changed")
    req(hlock.get("human_relevance_labels_used") is False,
        "Historical judge lock says human relevance labels were used")

    for path, key, label in [
        (JUDGE_FILE, "judge_source_sha256", "frozen judge source"),
        (SCORING_FILE, "scoring_source_sha256", "frozen deterministic scoring source"),
        (JUDGE_CONFIG_FILE, "judge_config_sha256", "frozen judge config"),
        (RUBRIC_FILE, "rubric_sha256", "frozen rubric"),
    ]:
        expected = str(hlock.get(key, "")).strip()
        req(bool(expected), f"Historical scoring lock missing {key}")
        verify_text(path, expected, label)

    config = load_json(JUDGE_CONFIG_FILE)
    req(config.get("version") == "0.29.0-r5", "Judge config version changed")
    req(config.get("provider") == "ollama", "Judge provider changed")
    req(float(config.get("temperature")) == 0.0, "Judge temperature changed")
    req(config.get("rubric_id") == "semantic-recommendation-relevance", "Judge rubric ID changed")

    rubric = load_json(RUBRIC_FILE)
    req(rubric.get("status") == "frozen", "Rubric is not frozen")
    req(rubric.get("version") == "0.1.0", "Rubric version changed")

    verify_text(SPAN_ECHO_PATCH_FILE, EXPECTED["span_echo_patch"],
                "accepted non-semantic span-echo patch")
    patch = load_json(SPAN_ECHO_PATCH_FILE)
    req(patch.get("status") == "APPLIED_NON_SEMANTIC_EXECUTION_PATCH",
        "Unexpected span-echo patch status")
    for key in [
        "judge_source_modified", "judge_prompt_modified", "semantic_rules_changed",
        "deterministic_scoring_rules_changed", "facet_spec_changed",
        "judge_config_changed", "rubric_changed", "release_metrics_or_gates_changed",
    ]:
        req(patch.get(key) is False, f"Span-echo patch changed forbidden contract: {key}")

    load_and_validate_v5d_facets()
    print("PASS  frozen semantic_relevance_v0.29.0-r5 measurement instrument verified")
    print("PASS  frozen V5D facet specification validated against judge schema")


def validate() -> pd.DataFrame:
    verify_raw(CONTRACT, EXPECTED["contract"], "V5D judging-input contract")
    verify_raw(DISPLAY_CONTRACT, EXPECTED["display_contract"], "V5D displayed-pair-union contract")
    verify_raw(DISPLAY_UNION, EXPECTED["display_union"], "V5D displayed-pair union")
    verify_raw(DISPLAY_LOCK, EXPECTED["display_lock"], "V5D displayed-pair-union lock")
    verify_raw(AUTHORING_LOCK, EXPECTED["authoring_lock"], "V5D authoring lock")
    verify_raw(FACET_FILE, EXPECTED["facet"], "V5D facet specification")
    verify_raw(PRIOR_QUERY_MANIFEST, EXPECTED["prior_manifest"], "V5D frozen prior-query manifest")

    dlock = load_json(DISPLAY_LOCK)
    req(dlock.get("status") == "V5D_DISPLAYED_PAIR_UNION_FROZEN_BEFORE_JUDGING",
        "Unexpected V5D display-union lock status")
    req(int(dlock.get("unique_query_book_pairs", -1)) == EXPECTED_PAIRS,
        "Display-union lock pair count changed")
    req(dlock.get("displayed_pair_union_sha256") == EXPECTED["display_union"],
        "Display-union lock hash mismatch")
    req(dlock.get("judge_execution_performed") is False,
        "Display-union lock says V5D judge already ran")
    req(dlock.get("development_selection_calculated") is False,
        "Display-union lock says development selection already ran")

    alock = load_json(AUTHORING_LOCK)
    req(alock.get("status") == "V5D_QUERY_AND_FACET_SPEC_FROZEN_BEFORE_DEVELOPMENT_RETRIEVAL",
        "Unexpected V5D authoring-lock status")
    req(alock.get("query_file_sha256") == "1d93979d21d081afaecac63ee46c56330e80a2e7dbeeeea5d3bfa55a4b56387a",
        "V5D authoring lock query hash changed")
    req(alock.get("facet_file_sha256") == EXPECTED["facet"],
        "V5D authoring lock facet hash changed")

    frame = pd.read_csv(DISPLAY_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id", "query", "stratum", "isbn13", "title", "description", "authors",
        "in_B", "B_rank", "in_R", "R_rank", "in_K", "K_rank", "in_RK", "RK_rank", "system_count",
    }
    missing = required - set(frame.columns)
    req(not missing, f"Displayed union missing columns: {sorted(missing)}")
    req(len(frame) == EXPECTED_PAIRS, f"Expected {EXPECTED_PAIRS} displayed pairs, got {len(frame)}")
    frame = frame.copy()
    frame["isbn13"] = frame["isbn13"].map(canonical_isbn)
    req(not frame.duplicated(["query_id", "isbn13"]).any(), "Duplicate pair in displayed union")
    req(sorted(frame["query_id"].unique()) == EXPECTED_IDS, "Displayed-union query IDs changed")
    req(frame["description"].str.strip().ne("").all(), "Displayed union contains empty description")

    for system in ["B", "R", "K", "RK"]:
        truth = frame[f"in_{system}"].astype(str).str.lower().isin(["true", "1", "yes"])
        req(int(truth.sum()) == 240, f"{system}: displayed membership is not exactly 240")

    # Exact same-query reuse is impossible if every V5D query text is exactly
    # disjoint from the frozen prior query universe that covers all compatible
    # historical Q/H/V4H score families.
    manifest = load_json(PRIOR_QUERY_MANIFEST)
    req(int(manifest.get("query_count", -1)) == 63, "Prior-query manifest count changed")
    prior_entries = manifest.get("queries")
    req(isinstance(prior_entries, list), "Prior-query manifest missing queries list")
    prior_texts = {str(item["query"]) for item in prior_entries}
    v5d_query_rows = frame[["query_id", "query"]].drop_duplicates()
    req(len(v5d_query_rows) == 24, "Expected exactly 24 distinct V5D queries")
    collisions = [
        (str(row.query_id), str(row.query))
        for row in v5d_query_rows.itertuples(index=False)
        if str(row.query) in prior_texts
    ]
    req(not collisions, f"V5D exact query text collides with prior query universe: {collisions}")
    print("PASS  exact-query historical reuse eligibility is zero by frozen query-text disjointness")

    verify_measurement_instrument()

    for path in OUTPUTS + LATER_JUDGE_OUTPUTS:
        req(not path.exists(), f"Refusing because V5D judging artifact already exists: {path}")

    contract = load_json(CONTRACT)
    req(contract.get("status") == "FROZEN_BY_PACKAGE_BEFORE_V5D_JUDGING_INPUT_MATERIALIZATION",
        "Unexpected judging-input contract status")
    policy = contract.get("score_reuse_policy", {})
    req(policy.get("cross_query_reuse") is False, "Cross-query score reuse became allowed")
    req(policy.get("approximate_query_reuse") is False, "Approximate-query score reuse became allowed")
    req(int(policy.get("expected_reused_frozen_scores", -1)) == 0,
        "Contract no longer expects zero reusable scores")
    req(int(policy.get("expected_novel_pairs_to_judge", -1)) == EXPECTED_PAIRS,
        "Contract novel-pair count changed")

    return frame


def main() -> None:
    print("Semantic relevance V5 - judging-input / score-reuse preflight")
    print("-----------------------------------------------------------")
    print("Retrieval/reranking: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Development metrics/selection: NOT CALCULATED")
    print()

    frame = validate()
    per_query = frame.groupby("query_id").size()

    print()
    print("V5D JUDGING-INPUT PREFLIGHT: PASS")
    print(f"Frozen displayed pairs: {len(frame)}")
    print("Reusable frozen scores under exact-query + exact-ISBN rule: 0")
    print(f"Novel pairs requiring fresh judge execution: {len(frame)}")
    print(f"Novel pairs/query range: {int(per_query.min())}..{int(per_query.max())}")
    print("Judge candidate: semantic_relevance_v0.29.0-r5")
    print("Judge calls performed: 0")
    print("Human relevance labels used: NO")
    print("Development metrics calculated: NO")
    print("Development selection calculated: NO")
    print("Next: freeze the 452-case novel V5D judging input before any judge call.")


if __name__ == "__main__":
    main()
