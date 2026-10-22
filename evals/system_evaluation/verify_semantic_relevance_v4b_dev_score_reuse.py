from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS = REPO_ROOT / "evals"
V4_RUN = EVALS / "runs/semantic_relevance_v4a_development/20261019T104415Z_retrieval"
V3_HOLDOUT_RUN = EVALS / "runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
V3_POSTHOC_RUN = EVALS / "runs/semantic_relevance_v3_posthoc_diagnostic/v2_deep_candidate_prepare"

DISPLAY_UNION = V4_RUN / "v3b_v4b_dev_display_union.csv"
DISPLAY_UNION_LOCK = V4_RUN / "v3b_v4b_dev_display_union.lock.json"

V4A_COMPLETE_MAP = V4_RUN / "v4a_dev_complete_display_score_map.csv"
V4A_COMPLETE_MAP_LOCK = V4_RUN / "v4a_dev_complete_score_map.lock.json"

V3_HOLDOUT_SCORES = V3_HOLDOUT_RUN / "holdout_judge_scores.csv"
V3_HOLDOUT_LOCK = V3_HOLDOUT_RUN / "holdout_judge_scoring_lock.json"

V3_POSTHOC_MAP = V3_POSTHOC_RUN / "target_top50_complete_score_map.csv"
V3_POSTHOC_LOCK = V3_POSTHOC_RUN / "deep_candidate_scoring_lock.json"

V4A_DIAG_INVENTORY = V4_RUN / "v4a_retrieval_diagnostic_score_inventory.csv"
V4A_DIAG_PARTIAL_SCORES = V4_RUN / "v4a_retrieval_diagnostic_novel_bm25_judge_scores.csv"
V4A_DIAG_PARTIAL_FREEZE = V4_RUN / "v4a_retrieval_diagnostic_partial_judging.freeze.lock.json"
V4A_DIAG_QUARANTINE = V4_RUN / "v4a_retrieval_diagnostic_quarantined_cases.json"

OUT_INVENTORY = V4_RUN / "v4b_dev_display_score_inventory.csv"
OUT_NOVEL = V4_RUN / "v4b_dev_novel_display_pairs_to_judge.csv"
OUT_METADATA = V4_RUN / "v4b_dev_score_reuse_metadata.json"
OUT_LOCK = V4_RUN / "v4b_dev_judging_input.lock.json"

EXPECTED = {
    "display_union": "8625117c8b62a12c8cc4519f2690d353732db3477d1f425328b94679fb61209b",
    "display_union_lock": "696e5ce23491ea245e03f52369a3e64f2162e9ce0a06465547a16033137d9fe8",
    "v4a_complete_map": "60fb8b44f6eb13d486eaa6ffae483dca374e1415af8708d833912a18ec4d2000",
    "v4a_complete_map_lock": "fe5b4865df05c4086247360fc7693e71d3cfe63f8877e944eaea61016d418f23",
    "v3_holdout_scores": "d1c9ba1848fb4875c1e925b6c5bdafc3483db37a613bad49286a634d96803fa3",
    "v3_holdout_lock": "4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
    "v3_posthoc_map": "723fa27ecb8947334d25b3b7423a6597da39d594eb7ec2f979056b1c3cc516e5",
    "v3_posthoc_lock": "af67cc8090d0613973da4c927b573d29bb956be34d4c44255594cb12bde6fd1f",
    "v4a_diag_inventory": "250841f0bf9cefc5a461e09f3963842187bdac8bfdaafe03e7e07487d07c7e16",
    "v4a_diag_partial_scores": "ceb643cbff823007beb1f698b9fd42a78af70ead73e1c8bc90de94d14764be47",
    "v4a_diag_partial_freeze": "ae22f150f5f64f228f7970010103b1229c7b4c100f348f55faec172034319e55",
    "v4a_diag_quarantine": "ce16b3da279d4435d3b2eea7cf1aa190e6e804ecb6e275e8c12001703cbb0a38"
}
EXPECTED_TOTAL = 367
EXPECTED_QUERIES = 24
EXPECTED_SYSTEM_ROWS = 240
KNOWN_QUARANTINE_CASE = "V4A_RETRDIAG_H14_9781582972947"
KNOWN_QUARANTINE_QUERY = "H14"
KNOWN_QUARANTINE_ISBN = "9781582972947"

SCORE_PAYLOAD = [
    "judge_score", "match_level", "judge_reason", "scoring_explanation",
    "facet_assessments_json", "facet_evidence_json", "component_evidence_ledger_json",
]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def req(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def verify_raw(path: Path, expected: str, label: str) -> None:
    req(path.exists(), f"Missing frozen input: {path}")
    actual = sha256(path)
    req(actual == expected, f"{label} hash mismatch: expected={expected} actual={actual}")
    print(f"PASS  {label} raw SHA-256 verified")

def bool_value(value: Any) -> bool:
    text = str(value).strip().lower()
    if text in {"true","1","yes"}:
        return True
    if text in {"false","0","no",""}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")

def normalize_payload_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in SCORE_PAYLOAD:
        if col not in out.columns:
            out[col] = ""
    out["judge_score"] = pd.to_numeric(out["judge_score"], errors="raise").astype(int)
    req(out["judge_score"].between(0,4).all(), "Judge score outside 0..4")
    return out[["query_id","isbn13",*SCORE_PAYLOAD]].copy()

def source_v4a_complete() -> pd.DataFrame:
    frame = pd.read_csv(V4A_COMPLETE_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id","isbn13","final_judge_score","final_match_level","final_judge_reason",
        "final_scoring_explanation","final_facet_assessments_json","final_facet_evidence_json",
        "final_component_evidence_ledger_json"
    }
    req(not (required-set(frame.columns)), f"V4A complete map missing columns: {sorted(required-set(frame.columns))}")
    out = frame[list(required)].copy().rename(columns={
        "final_judge_score":"judge_score",
        "final_match_level":"match_level",
        "final_judge_reason":"judge_reason",
        "final_scoring_explanation":"scoring_explanation",
        "final_facet_assessments_json":"facet_assessments_json",
        "final_facet_evidence_json":"facet_evidence_json",
        "final_component_evidence_ledger_json":"component_evidence_ledger_json",
    })
    out = normalize_payload_columns(out)
    out["score_source"] = "v4a_complete_display_frozen"
    return out

def source_v3_holdout() -> pd.DataFrame:
    frame = pd.read_csv(V3_HOLDOUT_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id","isbn13",*SCORE_PAYLOAD}
    req(not (required-set(frame.columns)), f"V3 holdout scores missing columns: {sorted(required-set(frame.columns))}")
    out = normalize_payload_columns(frame[list(required)])
    out["score_source"] = "v3_holdout_frozen"
    return out

def source_v3_posthoc() -> pd.DataFrame:
    frame = pd.read_csv(V3_POSTHOC_MAP, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {
        "query_id","isbn13","judge_score","final_match_level","final_judge_reason",
        "final_scoring_explanation","final_facet_assessments_json","final_facet_evidence_json",
        "final_component_evidence_ledger_json"
    }
    req(not (required-set(frame.columns)), f"V3 posthoc map missing columns: {sorted(required-set(frame.columns))}")
    out = frame[list(required)].copy().rename(columns={
        "final_match_level":"match_level",
        "final_judge_reason":"judge_reason",
        "final_scoring_explanation":"scoring_explanation",
        "final_facet_assessments_json":"facet_assessments_json",
        "final_facet_evidence_json":"facet_evidence_json",
        "final_component_evidence_ledger_json":"component_evidence_ledger_json",
    })
    out = normalize_payload_columns(out)
    out["score_source"] = "v3_posthoc_deep_frozen"
    return out

def source_v4a_diag_reused() -> pd.DataFrame:
    frame = pd.read_csv(V4A_DIAG_INVENTORY, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"query_id","isbn13","score_status","judge_score",*SCORE_PAYLOAD[1:]}
    req(not (required-set(frame.columns)), f"V4A diagnostic inventory missing columns: {sorted(required-set(frame.columns))}")
    out = frame[frame["score_status"].eq("reused")][["query_id","isbn13",*SCORE_PAYLOAD]].copy()
    out = normalize_payload_columns(out)
    out["score_source"] = "v4a_retrdiag_reused_frozen"
    return out

def source_v4a_diag_new() -> pd.DataFrame:
    frame = pd.read_csv(V4A_DIAG_PARTIAL_SCORES, dtype=str, keep_default_na=False, encoding="utf-8")
    required = {"case_id","query_id","isbn13","judge_score","match_level","judge_reason","scoring_explanation",
                "facet_assessments_json","facet_evidence_json","component_evidence_ledger_json"}
    req(not (required-set(frame.columns)), f"V4A diagnostic partial scores missing columns: {sorted(required-set(frame.columns))}")
    req(KNOWN_QUARANTINE_CASE not in set(frame["case_id"].astype(str)), "Quarantined case unexpectedly appears in actual diagnostic scores")
    out = normalize_payload_columns(frame[["query_id","isbn13",*SCORE_PAYLOAD]])
    out["score_source"] = "v4a_retrdiag_novel_frozen"
    return out

def build_precedence_map() -> tuple[pd.DataFrame, dict[str,int], int]:
    # Fixed before viewing V4B reuse counts. Earlier entries have higher precedence.
    sources = [
        source_v4a_complete(),
        source_v3_holdout(),
        source_v3_posthoc(),
        source_v4a_diag_reused(),
        source_v4a_diag_new(),
    ]
    names = [
        "v4a_complete_display_frozen",
        "v3_holdout_frozen",
        "v3_posthoc_deep_frozen",
        "v4a_retrdiag_reused_frozen",
        "v4a_retrdiag_novel_frozen",
    ]

    selected: dict[tuple[str,str], dict[str,Any]] = {}
    availability = {name: 0 for name in names}
    overlap_score_disagreements = 0

    for source_name, frame in zip(names, sources):
        req(not frame.duplicated(["query_id","isbn13"]).any(), f"Duplicate pair inside score source {source_name}")
        for _, row in frame.iterrows():
            key = (str(row["query_id"]), str(row["isbn13"]))
            availability[source_name] += 1
            candidate = row.to_dict()
            if key in selected:
                if int(selected[key]["judge_score"]) != int(candidate["judge_score"]):
                    overlap_score_disagreements += 1
                continue
            selected[key] = candidate

    out = pd.DataFrame(selected.values())
    req(not out.duplicated(["query_id","isbn13"]).any(), "Precedence score map contains duplicate pair")
    return out, availability, overlap_score_disagreements

def main() -> None:
    print("Semantic relevance V4B - score reuse / novel-input preflight")
    print("------------------------------------------------------------")
    print("Score inventory generation: DISABLED")
    print("Judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Development comparison: NOT CALCULATED")
    print("Development selection: NOT PRODUCED")
    print("Release analysis: DISABLED")
    print()

    for out in [OUT_INVENTORY, OUT_NOVEL, OUT_METADATA, OUT_LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing V4B score-reuse artifact: {out}")

    for path, key, label in [
        (DISPLAY_UNION, "display_union", "V3B/V4B displayed union"),
        (DISPLAY_UNION_LOCK, "display_union_lock", "V3B/V4B displayed-union lock"),
        (V4A_COMPLETE_MAP, "v4a_complete_map", "V4A complete display score map"),
        (V4A_COMPLETE_MAP_LOCK, "v4a_complete_map_lock", "V4A complete-score-map lock"),
        (V3_HOLDOUT_SCORES, "v3_holdout_scores", "V3 holdout judge scores"),
        (V3_HOLDOUT_LOCK, "v3_holdout_lock", "V3 holdout scoring lock"),
        (V3_POSTHOC_MAP, "v3_posthoc_map", "V3 posthoc complete score map"),
        (V3_POSTHOC_LOCK, "v3_posthoc_lock", "V3 posthoc scoring lock"),
        (V4A_DIAG_INVENTORY, "v4a_diag_inventory", "V4A retrieval-diagnostic score inventory"),
        (V4A_DIAG_PARTIAL_SCORES, "v4a_diag_partial_scores", "V4A retrieval-diagnostic partial scores"),
        (V4A_DIAG_PARTIAL_FREEZE, "v4a_diag_partial_freeze", "V4A retrieval-diagnostic partial freeze"),
        (V4A_DIAG_QUARANTINE, "v4a_diag_quarantine", "V4A retrieval-diagnostic quarantine artifact"),
    ]:
        verify_raw(path, EXPECTED[key], label)

    dlock = load_json(DISPLAY_UNION_LOCK)
    req(dlock.get("status") == "V4B_DEV_DISPLAY_UNION_FROZEN_BEFORE_SCORE_REUSE_OR_JUDGING", "Unexpected V4B display-union lock status")
    req(int(dlock.get("union_rows",-1)) == EXPECTED_TOTAL, "V4B display-union row count changed")
    req(dlock.get("known_h14_quarantine_pair_present") is False, "Display-union lock says H14 quarantine pair is present")

    cmap_lock = load_json(V4A_COMPLETE_MAP_LOCK)
    req(cmap_lock.get("status") == "V4A_DEV_COMPLETE_DISPLAY_SCORE_MAP_FROZEN_BEFORE_COMPARISON", "Unexpected V4A complete-map lock status")
    req(cmap_lock.get("human_relevance_labels_used") is False, "V4A complete map used human labels")

    hlock = load_json(V3_HOLDOUT_LOCK)
    req(hlock.get("status") == "V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS", "Unexpected V3 holdout lock status")

    plock = load_json(V3_POSTHOC_LOCK)
    req(plock.get("status") == "V3_POSTHOC_DEEP_JUDGE_SCORES_FROZEN_BEFORE_CLASSIFICATION", "Unexpected V3 posthoc lock status")
    req(plock.get("human_relevance_labels_used") is False, "V3 posthoc used human labels")

    pfreeze = load_json(V4A_DIAG_PARTIAL_FREEZE)
    req(int(pfreeze.get("persisted_scored_pairs",-1)) == 674, "V4A diagnostic persisted score count changed")
    req(int(pfreeze.get("quarantined_unscored_pairs",-1)) == 1, "V4A diagnostic quarantine count changed")
    req(pfreeze.get("quarantined_case_id") == KNOWN_QUARANTINE_CASE, "V4A diagnostic quarantine identity changed")
    req(pfreeze.get("quarantined_score_assigned") is False, "V4A diagnostic quarantine has an assigned score")

    quarantine = load_json(V4A_DIAG_QUARANTINE)
    req(isinstance(quarantine, list) and len(quarantine) == 1, "Expected exactly one diagnostic quarantine row")
    req(quarantine[0].get("case_id") == KNOWN_QUARANTINE_CASE, "Quarantine artifact identity changed")
    req(quarantine[0].get("status") == "QUARANTINED_UNSCORED", "Quarantine status changed")

    display = pd.read_csv(DISPLAY_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    req(len(display) == EXPECTED_TOTAL, f"Expected {EXPECTED_TOTAL} displayed union rows, got {len(display)}")
    req(display["query_id"].nunique() == EXPECTED_QUERIES, "Display union is not H01-H24")
    req(not display.duplicated(["query_id","isbn13"]).any(), "Duplicate pair in V4B display union")
    req(not ((display["query_id"] == KNOWN_QUARANTINE_QUERY) & (display["isbn13"] == KNOWN_QUARANTINE_ISBN)).any(),
        "Unresolved H14 quarantine pair is present in V4B displayed union")

    score_map, availability, disagreement_count = build_precedence_map()
    req(len(score_map) > 0, "Frozen precedence score map is empty")
    print(f"PASS  frozen score sources normalize under pre-registered precedence")
    print(f"INFO  source rows available: {availability}")
    print(f"INFO  overlapping-pair score disagreements observed: {disagreement_count}")
    print("PASS  disagreement audit is informational only; source precedence is fixed independently of score values")
    print()
    print("V4B SCORE REUSE PREFLIGHT: PASS")
    print("Display union rows: 367")
    print("Score precedence: V4A complete display > V3 holdout > V3 posthoc > V4A retrdiag reused > V4A retrdiag novel")
    print("Score inventory generated: NO")
    print("Judge calls: 0")
    print("Human relevance labels used: NO")
    print("Development comparison calculated: NO")
    print("Development selection produced: NO")
    print("Release analysis: NO")
    print("Next: deterministically freeze reused-vs-novel score inventory.")

if __name__ == "__main__":
    main()
