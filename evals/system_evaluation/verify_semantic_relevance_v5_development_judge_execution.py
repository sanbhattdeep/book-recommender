from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
EVALS=ROOT/"evals"
SYS=EVALS/"system_evaluation"
RUN=EVALS/"runs/semantic_relevance_v5_development/20261026T124615Z_candidate_collection"
HIST_RUN=EVALS/"runs/semantic_relevance_v3_holdout/20261007T124150Z_retrieval"
if str(EVALS) not in sys.path: sys.path.insert(0,str(EVALS))

from semantic_relevance_facet_scoring import QueryFacet, QueryFacetSpec, validate_query_facet_spec

JUDGING_INPUT=RUN/"v5d_novel_display_pairs_to_judge.csv"
JUDGING_INPUT_LOCK=RUN/"v5d_judging_input.lock.json"
SCORE_INVENTORY=RUN/"v5d_display_score_inventory.csv"
V5D_FACET_FILE=EVALS/"facets/semantic_relevance/semantic_relevance_v5_development_query_facets.v0.1.0.json"
HISTORICAL_SCORING_LOCK=HIST_RUN/"holdout_judge_scoring_lock.json"
JUDGE_FILE=EVALS/"semantic_relevance_facet_judge.py"
SCORING_FILE=EVALS/"semantic_relevance_facet_scoring.py"
JUDGE_CONFIG_FILE=EVALS/"judge_configs/semantic_relevance_judge.v0.29.0-r5.json"
RUBRIC_FILE=EVALS/"rubrics/semantic_relevance/semantic_relevance_rubric.v0.1.0.json"
SPAN_ECHO_PATCH_FILE=SYS/"semantic_relevance_v3_posthoc_execution_patch.span-echo-normalization1.json"
ORIGINAL_DUP_PATCH_FILE=SYS/"semantic_relevance_v4h_execution_patch.duplicate-candidate-id-normalization1.json"
V5_CARRY_DUP_PATCH_FILE=SYS/"semantic_relevance_v5_development_execution_patch.duplicate-candidate-id-carryforward1.json"
EXEC_CONTRACT=SYS/"semantic_relevance_v5_development_judge_execution_contract.v0.1.0.json"
SCORES=RUN/"v5d_judge_scores.csv"
METADATA=RUN/"v5d_judging_metadata.json"
FINAL_LOCK=RUN/"v5d_judge_scoring.lock.json"

EXPECTED={
 "judging_input":"e9aa5eb04d9847e289c7621a0aaeec50e8fe057ae975286e05fe22e79c8396d0",
 "judging_input_lock":"6cdc1f17928e7b3b05bdc1b5efb62a9c8da2d760ae2c4032c33df9a18696d826",
 "score_inventory":"99117bd78a5d9f42bc50110ec575c5c3df65ef9b2717d6892c60d3d4739c3f2b",
 "facet":"355ac59604fc73039fe394ba36b9ef37c04cd0be447ff470bce669d8adeb4746",
 "historical_scoring_lock":"4a4be17bdec6a3d1cd95667bde9fbae1fb4a6077e10b5dc4270710787ffe6777",
 "span_echo_patch":"85488eb50ae0d9d4b8f0f2c213f9874a3f3d24881131af57e1bf61e5776e4266",
 "original_dup_patch":"09b2acb1d3809da6ed4667d1e3f7b042dc86f85b6f69b732bb9017136f0b2b94",
 "v5_carry_patch":"d079bb55f2dfd98b5ab78ee2138faf1f1f2d5d4d90681616f962686817461e57",
 "exec_contract":"aac315cbea6a1994f0ecd897267e34db38558a87c5d8a3184f1b9cd2d3abc033"
}
EXPECTED_IDS=[f"V5D{i:02d}" for i in range(1,25)]
EXPECTED_CASES=452
SMOKE_QUERY_IDS=["V5D01","V5D17","V5D21"]

def sha256(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lf_sha256(p:Path)->str:
    b=p.read_bytes();return hashlib.sha256(b.replace(b"\r\n",b"\n").replace(b"\r",b"\n")).hexdigest()
def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load_json(p:Path)->dict[str,Any]:
    req(p.exists(),f"Missing JSON input: {p}");return json.loads(p.read_text(encoding="utf-8-sig"))
def verify_raw(p:Path,e:str,l:str)->None:
    req(p.exists(),f"Missing {l}: {p}");a=sha256(p);req(a==e,f"{l} hash mismatch: expected={e} actual={a}");print(f"PASS  {l} raw SHA-256 verified")
def verify_text(p:Path,e:str,l:str)->None:
    req(p.exists(),f"Missing {l}: {p}");r,n=sha256(p),lf_sha256(p);req(r==e or n==e,f"{l} hash mismatch: expected={e} raw={r} lf={n}");print(f"PASS  {l} {'raw' if r==e else 'canonical-LF'} SHA-256 verified")
def canonical_isbn(v:Any)->str:
    t=str(v).strip().strip('"').strip("'")
    if t.endswith(".0") and t[:-2].isdigit(): t=t[:-2]
    req(bool(t),"ISBN is empty");return t

def load_facet_specs(payload:dict[str,Any])->dict[str,QueryFacetSpec]:
    raw=payload.get("queries");req(isinstance(raw,list),"Facet spec must contain queries list");specs={}
    for item in raw:
        qid=str(item["query_id"]);req(qid not in specs,f"Duplicate query ID: {qid}")
        spec=QueryFacetSpec(query_id=qid,query=str(item["query"]),facets=[QueryFacet(**x) for x in item["facets"]])
        validate_query_facet_spec(spec)
        for facet in spec.facets:req(bool(facet.required_components),f"{qid}/{facet.facet_id} missing required components")
        specs[qid]=spec
    req(sorted(specs)==EXPECTED_IDS,"V5D facet query IDs changed");return specs

def verify_measurement_instrument()->dict[str,Any]:
    verify_raw(HISTORICAL_SCORING_LOCK,EXPECTED["historical_scoring_lock"],"historical frozen judge scoring lock")
    h=load_json(HISTORICAL_SCORING_LOCK)
    req(h.get("status")=="V3_HOLDOUT_JUDGE_SCORES_FROZEN_BEFORE_ANALYSIS","Unexpected historical scoring-lock status")
    req(h.get("judge_candidate")=="semantic_relevance_v0.29.0-r5","Frozen judge candidate changed")
    req(h.get("human_relevance_labels_used") is False,"Historical judge used human labels")
    for p,key,label in [
      (JUDGE_FILE,"judge_source_sha256","frozen judge source"),
      (SCORING_FILE,"scoring_source_sha256","frozen deterministic scoring"),
      (JUDGE_CONFIG_FILE,"judge_config_sha256","frozen judge config"),
      (RUBRIC_FILE,"rubric_sha256","frozen rubric")]:
        e=str(h.get(key,""));req(bool(e),f"Historical scoring lock missing {key}");verify_text(p,e,label)
    verify_raw(V5D_FACET_FILE,EXPECTED["facet"],"V5D facet specification")
    verify_text(SPAN_ECHO_PATCH_FILE,EXPECTED["span_echo_patch"],"accepted span-echo patch")
    verify_text(ORIGINAL_DUP_PATCH_FILE,EXPECTED["original_dup_patch"],"accepted original duplicate-ID patch")
    verify_text(V5_CARRY_DUP_PATCH_FILE,EXPECTED["v5_carry_patch"],"V5D duplicate-ID carry-forward amendment")
    cp=load_json(V5_CARRY_DUP_PATCH_FILE);req(cp.get("based_on_patch_sha256")==EXPECTED["original_dup_patch"],"Carry-forward patch binding changed")
    for k in ["none_semantics_changed","unknown_id_validation_changed","judge_source_modified","judge_prompt_modified","judge_model_modified","judge_temperature_modified","semantic_rules_changed","deterministic_scoring_rules_changed","facet_spec_changed","judge_config_changed","rubric_changed","candidate_set_changed","displayed_pair_union_changed","historical_score_reuse_changed","human_relevance_labels_used","development_metrics_or_selection_changed","release_metrics_or_gates_changed"]:
        req(cp.get(k) is False,f"Carry-forward patch changes forbidden contract: {k}")
    verify_text(EXEC_CONTRACT,EXPECTED["exec_contract"],"V5D judge-execution contract")
    ec=load_json(EXEC_CONTRACT);req(ec.get("status")=="FROZEN_BY_PACKAGE_BEFORE_V5D_JUDGE_EXECUTION","Unexpected execution-contract status")
    cfg=load_json(JUDGE_CONFIG_FILE);req(cfg.get("version")=="0.29.0-r5","Judge config version changed");req(float(cfg.get("temperature"))==0.0,"Judge temperature changed")
    rub=load_json(RUBRIC_FILE);req(rub.get("status")=="frozen","Rubric is not frozen")
    specs=load_facet_specs(load_json(V5D_FACET_FILE))
    print("PASS  frozen semantic_relevance_v0.29.0-r5 measurement instrument verified")
    print("PASS  accepted non-semantic execution patches verified")
    print("PASS  frozen V5D facets validated against judge schema")
    return {"lock":h,"config":cfg,"rubric":rub,"facet_specs":specs}

def verify_input_boundary()->tuple[pd.DataFrame,dict[str,Any]]:
    verify_raw(JUDGING_INPUT,EXPECTED["judging_input"],"V5D fresh judging input")
    verify_raw(JUDGING_INPUT_LOCK,EXPECTED["judging_input_lock"],"V5D judging-input lock")
    verify_raw(SCORE_INVENTORY,EXPECTED["score_inventory"],"V5D score inventory")
    lock=load_json(JUDGING_INPUT_LOCK)
    req(lock.get("status")=="V5D_JUDGING_INPUT_FROZEN_BEFORE_JUDGE_EXECUTION","Unexpected judging-input lock status")
    req(int(lock.get("case_count",-1))==EXPECTED_CASES,"Case count changed")
    req(lock.get("novel_judging_input_sha256")==EXPECTED["judging_input"],"Input lock hash mismatch")
    req(lock.get("score_inventory_sha256")==EXPECTED["score_inventory"],"Inventory lock hash mismatch")
    req(int(lock.get("reused_frozen_scores",-1))==0,"Historical reuse count changed")
    req(lock.get("historical_score_reuse_performed") is False,"Historical reuse unexpectedly performed")
    req(lock.get("judge_execution_performed") is False,"Input lock says judge already ran")
    req(lock.get("development_selection_calculated") is False,"Input lock says selection already ran")
    f=pd.read_csv(JUDGING_INPUT,dtype=str,keep_default_na=False,encoding="utf-8")
    required={"case_id","query_id","query","stratum","isbn13","title","description","authors","in_B","B_rank","in_R","R_rank","in_K","K_rank","in_RK","RK_rank","system_count"}
    req(not(required-set(f.columns)),f"Missing columns: {sorted(required-set(f.columns))}")
    req(len(f)==EXPECTED_CASES,f"Expected {EXPECTED_CASES} cases, got {len(f)}")
    f=f.copy();f["isbn13"]=f["isbn13"].map(canonical_isbn)
    req(not f["case_id"].duplicated().any(),"Duplicate case IDs");req(not f.duplicated(["query_id","isbn13"]).any(),"Duplicate query-book pairs")
    req(sorted(f["query_id"].unique())==EXPECTED_IDS,"V5D query IDs changed");req(f["description"].str.strip().ne("").all(),"Empty description")
    expected={f"V5D_JUDGE_{q}_{canonical_isbn(i)}" for q,i in zip(f["query_id"],f["isbn13"])}
    req(set(f["case_id"])==expected,"Deterministic case-ID rule changed")
    specs=load_facet_specs(load_json(V5D_FACET_FILE))
    for q,g in f.groupby("query_id",sort=True):
        req(g["query"].nunique()==1,f"{q} has multiple query texts");req(str(g["query"].iloc[0])==specs[q].query,f"{q} query text disagrees with facet spec")
    return f,lock

def main()->None:
    print("Semantic relevance V5 - v0.29.0-r5 judge execution preflight")
    print("-----------------------------------------------------------")
    print("Fresh cases: 452")
    print("Judge execution: DISABLED")
    print("Retrieval/reranking: DISABLED")
    print("Development metrics/selection: NOT CALCULATED")
    print()
    f,_=verify_input_boundary();verify_measurement_instrument()
    req(not FINAL_LOCK.exists(),"Final V5D scoring lock already exists")
    if SCORES.exists():
        e=pd.read_csv(SCORES,dtype=str,keep_default_na=False,encoding="utf-8")
        req("case_id" in e.columns,"Persisted score artifact lacks case_id");req(not e["case_id"].duplicated().any(),"Duplicate persisted case IDs");req(set(e["case_id"])<=set(f["case_id"]),"Persisted score outside frozen input")
        print(f"INFO  resume-state score artifact exists: {len(e)}/{EXPECTED_CASES} persisted")
    else: print("INFO  no persisted V5D score artifact exists yet")
    print()
    print("V5D JUDGE EXECUTION PREFLIGHT: PASS")
    print("Frozen cases: 452")
    print("Measurement instrument: semantic_relevance_v0.29.0-r5")
    print("Accepted execution patches: span-echo normalization + duplicate candidate-ID normalization")
    print("Resume-safe persistence: ENABLED")
    print("Smoke query IDs: V5D01, V5D17, V5D21")
    print("Judge calls performed during preflight: 0")
    print("Development metrics calculated: NO")
    print("Development selection calculated: NO")
    print("Next: run the 3-case cross-stratum smoke test.")

if __name__=="__main__":main()
