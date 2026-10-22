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

from datetime import datetime, timezone

def main() -> None:
    print("Semantic relevance V4B - prepare frozen score reuse / novel judging input")
    print("------------------------------------------------------------------------")
    print("New judge execution: DISABLED")
    print("Human relevance labels: NOT USED")
    print("Development comparison: NOT CALCULATED")
    print("Release analysis: DISABLED")
    print()

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

    for out in [OUT_INVENTORY, OUT_NOVEL, OUT_METADATA, OUT_LOCK]:
        req(not out.exists(), f"Refusing to overwrite existing V4B score-reuse artifact: {out}")

    display = pd.read_csv(DISPLAY_UNION, dtype=str, keep_default_na=False, encoding="utf-8")
    required_display = {
        "query_id","query","stratum","isbn13","title","description","authors",
        "pair_source","in_v3b","v3b_rank","in_v4b","v4b_rank",
    }
    req(not (required_display-set(display.columns)), f"Display union missing columns: {sorted(required_display-set(display.columns))}")
    req(len(display) == EXPECTED_TOTAL, "Display union row count changed")
    req(not display.duplicated(["query_id","isbn13"]).any(), "Duplicate pair in display union")

    score_map, availability, disagreement_count = build_precedence_map()
    merged = display.merge(score_map, on=["query_id","isbn13"], how="left", validate="one_to_one")
    merged["score_status"] = merged["judge_score"].apply(lambda x: "reused" if pd.notna(x) else "novel")

    reused_mask = merged["score_status"].eq("reused")
    novel_mask = ~reused_mask
    reused = int(reused_mask.sum())
    novel = int(novel_mask.sum())
    req(reused + novel == EXPECTED_TOTAL, "Reuse/novel counts do not cover V4B display union")

    in_v3b = merged["in_v3b"].map(bool_value)
    in_v4b = merged["in_v4b"].map(bool_value)
    req(int(in_v3b.sum()) == EXPECTED_SYSTEM_ROWS, "V3B displayed membership is not 240")
    req(int(in_v4b.sum()) == EXPECTED_SYSTEM_ROWS, "V4B displayed membership is not 240")
    req(merged.loc[in_v3b, "score_status"].eq("reused").all(), "A frozen V3B displayed pair lacks a reusable score")
    req(merged.loc[novel_mask, "pair_source"].eq("v4b_only").all(), "Novel set contains a pair that is not V4B-only")
    req(not (
        merged.loc[novel_mask, "query_id"].eq(KNOWN_QUARANTINE_QUERY)
        & merged.loc[novel_mask, "isbn13"].eq(KNOWN_QUARANTINE_ISBN)
    ).any(), "Unresolved H14 quarantine pair entered V4B novel judging input")

    merged["judge_score"] = merged["judge_score"].astype("object")
    merged.loc[reused_mask, "judge_score"] = pd.to_numeric(
        merged.loc[reused_mask, "judge_score"], errors="raise"
    ).astype(int)
    merged.loc[novel_mask, "judge_score"] = ""
    merged.loc[novel_mask, "score_source"] = ""
    for col in SCORE_PAYLOAD[1:]:
        merged.loc[novel_mask, col] = ""

    inventory_cols = [
        "query_id","query","stratum","isbn13","title","description","authors",
        "pair_source","in_v3b","v3b_rank","in_v4b","v4b_rank",
        "score_status","score_source","judge_score","match_level","judge_reason",
        "scoring_explanation","facet_assessments_json","facet_evidence_json",
        "component_evidence_ledger_json",
    ]
    inventory = merged[inventory_cols].sort_values(["query_id","isbn13"], kind="stable").reset_index(drop=True)

    novel_pairs = inventory[inventory["score_status"].eq("novel")].copy()
    novel_pairs.insert(0, "case_id", novel_pairs.apply(lambda r: f"V4B_DEV_{r['query_id']}_{r['isbn13']}", axis=1))
    novel_pairs = novel_pairs[[
        "case_id","query_id","query","stratum","isbn13","title","description","authors",
        "pair_source","in_v3b","v3b_rank","in_v4b","v4b_rank",
    ]].copy()
    novel_pairs["_v4"] = pd.to_numeric(novel_pairs["v4b_rank"], errors="raise").astype(int)
    novel_pairs = novel_pairs.sort_values(["query_id","_v4","isbn13"], kind="stable").drop(columns=["_v4"]).reset_index(drop=True)
    req(not novel_pairs["case_id"].duplicated().any(), "Novel V4B case IDs are not unique")

    OUT_INVENTORY.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(OUT_INVENTORY, index=False, encoding="utf-8")
    novel_pairs.to_csv(OUT_NOVEL, index=False, encoding="utf-8")

    reused_sources = inventory.loc[reused_mask, "score_source"].value_counts().to_dict()
    novel_sources = inventory.loc[novel_mask, "pair_source"].value_counts().to_dict()

    metadata = {
        "status": "V4B_DEV_SCORE_REUSE_PREPARATION_COMPLETE",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "display_union_sha256": EXPECTED["display_union"],
        "display_union_lock_sha256": EXPECTED["display_union_lock"],
        "display_union_rows": EXPECTED_TOTAL,
        "score_source_precedence": [
            "v4a_complete_display_frozen",
            "v3_holdout_frozen",
            "v3_posthoc_deep_frozen",
            "v4a_retrdiag_reused_frozen",
            "v4a_retrdiag_novel_frozen",
        ],
        "source_row_availability": {k:int(v) for k,v in availability.items()},
        "overlapping_pair_score_disagreements_observed": int(disagreement_count),
        "overlap_disagreements_affect_selection": False,
        "reused_frozen_scores": reused,
        "novel_pairs_to_judge": novel,
        "reused_score_source_counts": {str(k):int(v) for k,v in reused_sources.items()},
        "novel_pair_source_counts": {str(k):int(v) for k,v in novel_sources.items()},
        "known_h14_quarantine_pair_present": False,
        "new_judge_calls_performed": 0,
        "human_relevance_labels_used": False,
        "development_comparison_calculated": False,
        "development_selection_decision_produced": False,
        "release_metrics_calculated": False,
        "outputs": {
            OUT_INVENTORY.name: sha256(OUT_INVENTORY),
            OUT_NOVEL.name: sha256(OUT_NOVEL),
        },
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    lock = {
        "status": "V4B_DEV_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_query_set": "H01-H24",
        "display_union_sha256": EXPECTED["display_union"],
        "display_union_lock_sha256": EXPECTED["display_union_lock"],
        "display_union_rows": EXPECTED_TOTAL,
        "score_source_precedence": metadata["score_source_precedence"],
        "reused_frozen_scores": reused,
        "novel_pairs_to_judge": novel,
        "score_inventory_sha256": sha256(OUT_INVENTORY),
        "novel_judging_input_sha256": sha256(OUT_NOVEL),
        "score_reuse_metadata_sha256": sha256(OUT_METADATA),
        "overlapping_pair_score_disagreements_observed": int(disagreement_count),
        "known_h14_quarantine_pair_present": False,
        "judge_execution_performed": False,
        "human_relevance_labels_used": False,
        "development_comparison_calculated": False,
        "development_selection_decision_produced": False,
        "release_metrics_calculated": False,
        "release_decision_produced": False,
        "next_stage": "PREPARE_V4B_DEVELOPMENT_NOVEL_PAIR_JUDGE_EXECUTION_PACKAGE",
    }
    OUT_LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    print(f"PASS  V4B display score inventory frozen: {len(inventory)} rows")
    print(f"INFO  reused frozen scores: {reused}")
    print(f"INFO  novel displayed pairs to judge: {novel}")
    print(f"INFO  reused sources: {reused_sources}")
    print(f"INFO  novel pair sources: {novel_sources}")
    print(f"INFO  overlapping-pair score disagreements observed under fixed precedence: {disagreement_count}")
    print("PASS  unresolved H14 quarantine pair absent from novel judging input")
    print("PASS  V4B novel judging input frozen before any new judge execution")
    print()
    print("V4B DEVELOPMENT SCORE REUSE PREPARATION: PASS")
    print(f"Score inventory SHA-256: {sha256(OUT_INVENTORY)}")
    print(f"Novel judging input SHA-256: {sha256(OUT_NOVEL)}")
    print(f"Judging-input lock SHA-256: {sha256(OUT_LOCK)}")
    print("New judge calls performed: 0")
    print("Human relevance labels used: NO")
    print("Development comparison calculated: NO")
    print("Release analysis: NO")
    print("Next: prepare judge execution only for the frozen novel V4B displayed pairs.")

if __name__ == "__main__":
    main()
