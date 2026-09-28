"""Check the v0.28.0 r3 targeted regression on a partial 180-case run."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
R=Path(__file__).resolve().parents[1]
M=R/'evals/datasets/semantic_relevance_v0.28_regression_manifest.v1.2.0.json'
D=R/'evals/datasets/semantic_relevance_v0.28_development.v4.0.0.csv'
def resolve(raw):
 p=Path(raw); return p if p.is_absolute() else R/p
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();run=resolve(a.run)
 results=pd.read_csv(run/'judge_results.csv',dtype={'case_id':str});gold=pd.read_csv(D,dtype={'case_id':str})
 manifest=json.loads(M.read_text(encoding='utf-8')); expectations=manifest['targeted_expectations']; diagnostics=manifest['diagnostic_only_cases']
 merged=gold[['case_id','human_score','title']].merge(results,on='case_id',how='right',validate='one_to_one')
 missing=sorted(set(expectations)-set(merged.case_id.astype(str)))
 if missing:raise ValueError(f'Targeted run is missing required cases: {missing}')
 print('v0.28.0 r3 targeted regression');print('-----------------------------');ok=True
 for cid,exp in expectations.items():
  row=merged.loc[merged.case_id==cid].iloc[0];score=int(row.judge_score);checks=[]
  if 'judge_min' in exp:checks.append(score>=int(exp['judge_min']))
  if 'judge_max' in exp:checks.append(score<=int(exp['judge_max']))
  passed=all(checks);ok &= passed
  print(f"{'PASS' if passed else 'FAIL'}  {cid}: human={int(row.human_score)} judge={score}  {row.title}")
 print();print('Adjudication-only diagnostics (not gate criteria)');print('------------------------------------------------')
 for cid,meta in diagnostics.items():
  rows=merged.loc[merged.case_id==cid]
  if rows.empty: print(f'INFO  {cid}: not executed');continue
  row=rows.iloc[0];print(f"INFO  {cid}: human={int(row.human_score)} judge={int(row.judge_score)}  {row.title} -- {meta['reason']}")
 print();print(f"Targeted regression: {'PASS' if ok else 'FAIL'}")
if __name__=='__main__':main()
