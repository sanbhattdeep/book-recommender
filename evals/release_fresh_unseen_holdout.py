"""Unlock final holdout only after a complete fresh validation run passes every pre-registered gate."""
from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
from fresh_unseen_contract import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); args=p.parse_args()
    validate_split_contract(require_release=False)
    run_dir=Path(args.run); run_dir=run_dir if run_dir.is_absolute() else REPO_ROOT/run_dir
    summary_file=run_dir/'fresh_validation_summary.json'
    if not summary_file.exists(): raise FileNotFoundError('Run analyze_judge_fresh_validation.py first.')
    summary=load_json(summary_file)
    if summary.get('cases_compared')!=30: raise ValueError('Validation summary is not complete 30/30.')
    if summary.get('all_reference_thresholds_met') is not True: raise ValueError('Validation did not pass every pre-registered gate; final holdout remains locked.')
    meta=summary['run_metadata']
    if meta.get('judge_config_version')!='0.18.0' or meta.get('evaluation_dataset_role')!='fresh_unseen_validation': raise ValueError('Validation summary provenance mismatch.')
    release={'version':'2.0.0','released_at_utc':datetime.now(timezone.utc).isoformat(),'validation_run_id':meta['run_id'],'validation_summary_sha256':sha256_file(summary_file),'validation_passed_all_pre_registered_gates':True,'judge_config_version':'0.18.0','facet_spec_version':'0.6.0','rubric_version':'0.1.0','split_manifest_sha256':sha256_file(SPLIT_MANIFEST_FILE),'validation_sha256':sha256_file(VALIDATION_FILE),'holdout_sha256':sha256_file(HOLDOUT_FILE),'holdout_single_evaluation':True,'no_tuning_on_holdout':True}
    RELEASE_FILE.write_text(json.dumps(release,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    validate_split_contract(require_release=True)
    print('Final holdout release: PASS')
    print(f'Release token: {RELEASE_FILE}')
    print('The frozen 30-case holdout may now be run exactly once.')
if __name__=='__main__': main()
