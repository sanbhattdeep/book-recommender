from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from deepeval.models import OllamaModel

ROOT=Path(__file__).resolve().parents[2]
EVALS=ROOT/"evals"
if str(EVALS) not in sys.path: sys.path.insert(0,str(EVALS))
import semantic_relevance_facet_judge as facet_judge
from semantic_relevance_facet_scoring import compute_facet_score
import verify_semantic_relevance_v5_development_judge_execution as boundary

JUDGING_INPUT=boundary.JUDGING_INPUT
SCORES=boundary.SCORES
METADATA=boundary.METADATA
FINAL_LOCK=boundary.FINAL_LOCK
EXPECTED_VERIFIER_SHA256="0bf32c2f6899475314baf20542a035a3b34cfed3149d1cdd6554cbe9be9699d6"

RESULT_BASE=["case_id","query_id","query","stratum","isbn13","title","authors","in_B","B_rank","in_R","R_rank","in_K","K_rank","in_RK","RK_rank","system_count"]
RESULT_DIAG=["judge_score","match_level","judge_reason","semantic_generation_mode","semantic_stage_retry_count","book_subject_summary","book_subject_span_ids_json","book_subject_reason","book_subject_analysis_retry_count","deterministic_direct_cue_count","deterministic_cue_polarity_blocked_count","composite_verification_attempt_count","composite_verification_count","hard_exclusion_precheck_attempt_count","hard_exclusion_precheck_trigger_count","core_facet_count","incidental_core_count","meaningful_core_count","meaningful_core_coverage","strong_core_count","direct_core_count","entailed_core_count","adjacent_core_count","unsupported_core_count","qualifier_count","satisfied_qualifier_count","qualifier_cap_applied","clear_rule_applied","scoring_explanation","facet_assessments_json","facet_evidence_json","component_evidence_ledger_json"]
RESULT_COLUMNS=RESULT_BASE+RESULT_DIAG

def sha256(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def lf_sha256(p:Path)->str:
    b=p.read_bytes();return hashlib.sha256(b.replace(b"\r\n",b"\n").replace(b"\r",b"\n")).hexdigest()
def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def verify_verifier_unchanged()->None:
    p=Path(boundary.__file__).resolve();r,n=sha256(p),lf_sha256(p);req(r==EXPECTED_VERIFIER_SHA256 or n==EXPECTED_VERIFIER_SHA256,"Judge-execution verifier changed after package freeze");print("PASS  judge-execution preflight verifier content verified")

def install_execution_patches()->None:
    original_prominence=facet_judge._validate_prominence_result
    def patched_prominence(result:Any,spans:dict[str,str],config:dict[str,Any])->None:
        norm=[]
        for raw in list(result.supporting_span_ids):
            text=str(raw).strip()
            if text in spans: norm.append(text);continue
            rep=None
            for sid,span_text in spans.items():
                if text==f"{sid}: {span_text}": rep=sid;break
            norm.append(rep if rep is not None else raw)
        result.supporting_span_ids=norm
        original_prominence(result=result,spans=spans,config=config)
    facet_judge._validate_prominence_result=patched_prominence

    original_candidate=facet_judge.normalize_candidate_span_ids
    def patched_candidate(selection:Any,spans:dict[str,str])->list[str]:
        ids=[str(x).strip() for x in selection.evidence_span_ids]
        if facet_judge.NO_EVIDENCE_SPAN in ids:
            return original_candidate(selection=selection,spans=spans)
        if len(ids)!=len(set(ids)):
            uniq=list(dict.fromkeys(ids))
            print("INFO  non-semantic duplicate candidate-ID normalization: "+f"{ids} -> {uniq}")
            selection=facet_judge.EvidenceSelection(evidence_span_ids=uniq,reason=selection.reason)
        return original_candidate(selection=selection,spans=spans)
    facet_judge.normalize_candidate_span_ids=patched_candidate

def atomic_write_csv(f:pd.DataFrame,p:Path)->None:
    t=p.with_suffix(p.suffix+".tmp");f.to_csv(t,index=False,encoding="utf-8");os.replace(t,p)

def write_metadata(status:str,completed:int,total:int,last_case:str|None=None,error:str|None=None)->None:
    payload={"status":status,"updated_at_utc":datetime.now(timezone.utc).isoformat(),"judge_candidate":"semantic_relevance_v0.29.0-r5","query_set":"V5D01-V5D24","total_cases":total,"completed_cases":completed,"remaining_cases":total-completed,"last_completed_case_id":last_case,"error":error,"judging_input_sha256":boundary.EXPECTED["judging_input"],"historical_score_reuse_performed":False,"human_relevance_labels_used":False,"development_metrics_calculated":False,"development_selection_calculated":False,"release_metrics_calculated":False}
    METADATA.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")

def validate_existing(scores:pd.DataFrame,judging:pd.DataFrame)->pd.DataFrame:
    req(set(RESULT_COLUMNS)<=set(scores.columns),"Existing V5D score schema changed");req(not scores["case_id"].duplicated().any(),"Duplicate score case IDs");req(set(scores["case_id"])<=set(judging["case_id"]),"Score case outside frozen input")
    if len(scores):
        vals=pd.to_numeric(scores["judge_score"],errors="raise").astype(int);req(vals.between(0,4).all(),"Judge score outside 0..4")
        exp=judging.set_index("case_id")[["query_id","isbn13"]]
        for row in scores.itertuples(index=False):
            src=exp.loc[str(row.case_id)];req(str(row.query_id)==str(src["query_id"]),f"{row.case_id} query mismatch");req(boundary.canonical_isbn(row.isbn13)==boundary.canonical_isbn(src["isbn13"]),f"{row.case_id} ISBN mismatch")
    return scores[RESULT_COLUMNS].copy()

def choose_remaining(judging:pd.DataFrame,completed:set[str],smoke:bool,limit:int|None)->pd.DataFrame:
    r=judging[~judging["case_id"].isin(completed)].copy()
    if smoke:
        parts=[]
        for q in boundary.SMOKE_QUERY_IDS:
            g=r[r["query_id"]==q]
            if len(g): parts.append(g.iloc[[0]])
        return pd.concat(parts) if parts else r.head(0)
    return r.head(limit) if limit is not None else r

def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true");ap.add_argument("--limit",type=int,default=None);a=ap.parse_args()
    req(not(a.smoke and a.limit is not None),"--smoke and --limit are mutually exclusive")
    if a.limit is not None:req(a.limit>0,"--limit must be >0")
    print("Semantic relevance V5 - frozen v0.29.0-r5 judge execution")
    print("----------------------------------------------------------")
    print("Historical score reuse: NO");print("Human relevance labels: NOT USED");print("Retrieval/reranker calls: 0");print("Development metrics/selection: NOT CALCULATED");print("Persistence: after every completed case");print()
    req(not FINAL_LOCK.exists(),"Final V5D judge scoring lock already exists")
    judging,_=boundary.verify_input_boundary();instrument=boundary.verify_measurement_instrument();verify_verifier_unchanged()
    config,rubric,specs=instrument["config"],instrument["rubric"],instrument["facet_specs"]
    install_execution_patches();print("PASS  accepted non-semantic span-echo normalization installed");print("PASS  accepted duplicate candidate-ID normalization carried forward")
    if SCORES.exists():
        results=validate_existing(pd.read_csv(SCORES,dtype=str,keep_default_na=False,encoding="utf-8"),judging);print(f"PASS  resume-safe persisted scores verified: {len(results)}/{len(judging)}")
    else:
        results=pd.DataFrame(columns=RESULT_COLUMNS);print("INFO  no persisted V5D judge scores found; starting fresh")
    completed=set(results["case_id"].astype(str));remaining=choose_remaining(judging,completed,a.smoke,a.limit)
    judge_model=OllamaModel(model=config["model"],base_url=config["base_url"],temperature=config["temperature"])
    write_metadata("RUNNING",len(results),len(judging),None if not len(results) else str(results.iloc[-1]["case_id"]))
    try:
        for _,row in remaining.iterrows():
            cid=str(row["case_id"]);qid=str(row["query_id"]);spec=specs[qid]
            sem=facet_judge.generate_validated_semantic_verdict(judge_model=judge_model,row=row,spec=spec,rubric=rubric,config=config)
            score=compute_facet_score(spec=spec,verdict=sem.verdict)
            out={
             "case_id":cid,"query_id":qid,"query":str(row["query"]),"stratum":str(row["stratum"]),"isbn13":boundary.canonical_isbn(row["isbn13"]),"title":str(row["title"]),"authors":str(row["authors"]),
             "in_B":str(row["in_B"]),"B_rank":str(row["B_rank"]),"in_R":str(row["in_R"]),"R_rank":str(row["R_rank"]),"in_K":str(row["in_K"]),"K_rank":str(row["K_rank"]),"in_RK":str(row["in_RK"]),"RK_rank":str(row["RK_rank"]),"system_count":str(row["system_count"]),
             "judge_score":int(score.score),"match_level":str(score.label),"judge_reason":str(sem.verdict.overall_reason),"semantic_generation_mode":str(sem.generation_mode),"semantic_stage_retry_count":int(sem.stage_retry_count),
             "book_subject_summary":str(sem.book_subject_analysis.primary_subject_summary),"book_subject_span_ids_json":json.dumps(sem.book_subject_analysis.primary_subject_span_ids,ensure_ascii=False),"book_subject_reason":str(sem.book_subject_analysis.reason),"book_subject_analysis_retry_count":int(sem.subject_analysis_retry_count),
             "deterministic_direct_cue_count":int(sem.deterministic_direct_cue_count),"deterministic_cue_polarity_blocked_count":int(sem.deterministic_cue_polarity_blocked_count),"composite_verification_attempt_count":int(sem.composite_verification_attempt_count),"composite_verification_count":int(sem.composite_verification_count),"hard_exclusion_precheck_attempt_count":int(sem.hard_exclusion_precheck_attempt_count),"hard_exclusion_precheck_trigger_count":int(sem.hard_exclusion_precheck_trigger_count),
             "core_facet_count":int(score.core_facet_count),"incidental_core_count":int(score.incidental_core_count),"meaningful_core_count":int(score.meaningful_core_count),"meaningful_core_coverage":float(score.meaningful_core_coverage),"strong_core_count":int(score.strong_core_count),"direct_core_count":int(score.direct_core_count),"entailed_core_count":int(score.entailed_core_count),"adjacent_core_count":int(score.adjacent_core_count),"unsupported_core_count":int(score.unsupported_core_count),"qualifier_count":int(score.qualifier_count),"satisfied_qualifier_count":int(score.satisfied_qualifier_count),"qualifier_cap_applied":bool(score.qualifier_cap_applied),"clear_rule_applied":"" if score.clear_rule_applied is None else str(score.clear_rule_applied),"scoring_explanation":str(score.explanation),
             "facet_assessments_json":facet_judge.serialize_facet_assessments(spec=spec,verdict=sem.verdict),"facet_evidence_json":facet_judge.serialize_facet_evidence(sem.candidate_evidence_by_id),"component_evidence_ledger_json":json.dumps(sem.component_evidence_ledger_by_id,ensure_ascii=False)
            }
            results=pd.concat([results,pd.DataFrame([out])],ignore_index=True);results=validate_existing(results,judging);atomic_write_csv(results,SCORES);write_metadata("RUNNING",len(results),len(judging),cid)
            print(f"{len(results):3d}/{len(judging)} {cid}: score={score.score} label={score.label} retries={sem.stage_retry_count}")
    except Exception as exc:
        last=None if not len(results) else str(results.iloc[-1]["case_id"]);write_metadata("PARTIAL_FAILED",len(results),len(judging),last,repr(exc));print();print(f"PARTIAL  persisted judge scores: {len(results)}/{len(judging)}");print(f"FAILED   current case was not persisted: {repr(exc)}");print("Resume only after reviewing the failure.");raise
    if len(results)<len(judging):
        write_metadata("PARTIAL",len(results),len(judging),None if not len(results) else str(results.iloc[-1]["case_id"]));print();print(f"V5D JUDGE EXECUTION: PARTIAL ({len(results)}/{len(judging)} persisted)");print("Final scoring lock: NOT WRITTEN");print("Development metrics calculated: NO");print("Development selection calculated: NO");return
    results=validate_existing(results,judging);results=results.sort_values(["query_id","isbn13"],kind="stable").reset_index(drop=True);atomic_write_csv(results,SCORES)
    write_metadata("COMPLETE_FROZEN_BEFORE_DEVELOPMENT_ANALYSIS",len(results),len(judging),str(results.iloc[-1]["case_id"]))
    lock={"status":"V5D_JUDGE_SCORES_FROZEN_BEFORE_DEVELOPMENT_ANALYSIS","frozen_at_utc":datetime.now(timezone.utc).isoformat(),"query_set":"V5D01-V5D24","query_count":24,"case_count":boundary.EXPECTED_CASES,"judge_candidate":"semantic_relevance_v0.29.0-r5","judging_input_sha256":boundary.EXPECTED["judging_input"],"judging_input_lock_sha256":boundary.EXPECTED["judging_input_lock"],"score_inventory_sha256":boundary.EXPECTED["score_inventory"],"v5d_facet_spec_sha256":boundary.EXPECTED["facet"],"historical_measurement_instrument_lock_sha256":boundary.EXPECTED["historical_scoring_lock"],"span_echo_patch_sha256":boundary.EXPECTED["span_echo_patch"],"original_duplicate_id_patch_sha256":boundary.EXPECTED["original_dup_patch"],"v5d_duplicate_id_carryforward_sha256":boundary.EXPECTED["v5_carry_patch"],"judge_execution_contract_sha256":boundary.EXPECTED["exec_contract"],"judge_scores_sha256":sha256(SCORES),"judging_metadata_sha256":sha256(METADATA),"historical_score_reuse_performed":False,"fresh_judged_pairs":boundary.EXPECTED_CASES,"human_relevance_labels_used":False,"development_metrics_calculated":False,"development_selection_calculated":False,"release_metrics_calculated":False,"release_decision_produced":False,"next_stage":"CALCULATE_V5D_B_R_K_RK_DEVELOPMENT_METRICS_AND_PREREGISTERED_SELECTION"}
    FINAL_LOCK.write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")
    counts=pd.to_numeric(results["judge_score"],errors="raise").astype(int).value_counts().sort_index().to_dict()
    print();print("PASS  all 452 unique V5D displayed pairs freshly judged");print("PASS  complete V5D judge score set frozen before development analysis");print();print("V5D JUDGE EXECUTION: PASS");print(f"Judge scores SHA-256: {sha256(SCORES)}");print(f"Judge scoring lock SHA-256: {sha256(FINAL_LOCK)}");print(f"Score counts: {counts}");print("Development metrics calculated: NO");print("Development selection calculated: NO")

if __name__=="__main__":main()
