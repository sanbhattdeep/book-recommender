"""Build the 60-case v2 fresh-unseen human-labeling pool.

Run from repository root:
    uv run python evals/build_fresh_unseen_evaluation_pool.py

The script uses the app's raw Chroma similarity_search order, excludes every
ISBN frozen in semantic_relevance_exclusion_manifest.v2.0.0.json, chooses raw
target ranks 2/10/30/70 (scanning forward past excluded/duplicate books), and
adds one deterministic random candidate per query. It never calls the judge.
"""
from __future__ import annotations
import argparse, csv, importlib.util, json, random, re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT/'evals'; DATASETS_DIR=EVALS_DIR/'datasets'
POOL_VERSION='2.0.0'; RUBRIC_VERSION='0.1.0'
TARGET_RANKS=[2,10,30,70]; SEARCH_DEPTH=500; RANDOM_SEED=20260916
EXCLUSION_FILE=DATASETS_DIR/'semantic_relevance_exclusion_manifest.v2.0.0.json'
QUERY_METADATA_FILE=DATASETS_DIR/'semantic_relevance_query_metadata.v1.0.0.json'
OUTPUT_FILE=DATASETS_DIR/f'semantic_relevance_fresh_unseen_pool.v{POOL_VERSION}.csv'


def canonical_isbn(value):
    raw=str(value).strip()
    if re.fullmatch(r'\d+\.0', raw): raw=raw[:-2]
    if not re.fullmatch(r'\d{13}', raw): raise ValueError(f'Expected 13-digit ISBN, got {value!r}')
    return raw


def isbn_from_document(page_content: str) -> str:
    return canonical_isbn(page_content.strip().strip('"').split()[0])


def load_app_module():
    app_path=REPO_ROOT/'gradio-dashboard.py'
    if not app_path.exists(): raise FileNotFoundError(f'Application module not found: {app_path}')
    spec=importlib.util.spec_from_file_location('book_recommender_app', app_path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot load {app_path}')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    for attr in ['db_books','books']:
        if not hasattr(module, attr): raise AttributeError(f'gradio-dashboard.py does not expose {attr}')
    return module


def row_for_isbn(books, isbn: str):
    series=books['isbn13'].map(canonical_isbn); matches=books[series==isbn]
    return None if matches.empty else matches.iloc[0]


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=OUTPUT_FILE); args=parser.parse_args()
    exclusions=json.loads(EXCLUSION_FILE.read_text(encoding='utf-8'))
    seen={canonical_isbn(x) for x in exclusions['excluded_isbns']}
    query_payload=json.loads(QUERY_METADATA_FILE.read_text(encoding='utf-8'))
    queries=query_payload['queries']
    if len(queries)!=12: raise ValueError(f'Expected 12 frozen queries, got {len(queries)}')
    app=load_app_module(); rng=random.Random(RANDOM_SEED)
    records=[]; selected_global=set()
    for q in queries:
        qid=str(q['query_id']); query=str(q['query']); qslice=str(q['query_slice'])
        recs=app.db_books.similarity_search(query, k=SEARCH_DEPTH)
        retrieved=[isbn_from_document(rec.page_content) for rec in recs]
        local=set()
        for target in TARGET_RANKS:
            chosen=None
            for idx in range(target-1, len(retrieved)):
                isbn=retrieved[idx]
                if isbn in seen or isbn in selected_global or isbn in local: continue
                chosen=(idx+1,isbn); break
            if chosen is None: raise RuntimeError(f'{qid}: no eligible candidate at/after target {target}; increase SEARCH_DEPTH')
            actual_rank,isbn=chosen; row=row_for_isbn(app.books,isbn)
            if row is None: raise RuntimeError(f'{qid}: ISBN {isbn} missing from app.books')
            records.append({
                'case_id':f'U2_{qid}_T{target:02d}','query_id':qid,'query':query,'query_slice':qslice,
                'candidate_source':'semantic_retrieval','sampling_bucket':f'T{target:02d}',
                'sampling_target_rank':target,'retrieval_rank':actual_rank,'isbn13':isbn,
                'title':row['title'],'authors':row['authors'],'description':row['description'],
                'dataset_version':POOL_VERSION,'rubric_version':RUBRIC_VERSION,
                'human_score':'','human_reason':'','review_status':'UNLABELLED',
            })
            selected_global.add(isbn); local.add(isbn)
        excluded_random=seen|selected_global|set(retrieved)
        eligible=[idx for idx,v in app.books['isbn13'].items() if canonical_isbn(v) not in excluded_random]
        if not eligible: raise RuntimeError(f'{qid}: no eligible random candidate')
        ridx=rng.choice(eligible); row=app.books.loc[ridx]; isbn=canonical_isbn(row['isbn13'])
        records.append({
            'case_id':f'U2_{qid}_NEG','query_id':qid,'query':query,'query_slice':qslice,
            'candidate_source':'random_negative_candidate','sampling_bucket':'NEG',
            'sampling_target_rank':'','retrieval_rank':'','isbn13':isbn,
            'title':row['title'],'authors':row['authors'],'description':row['description'],
            'dataset_version':POOL_VERSION,'rubric_version':RUBRIC_VERSION,
            'human_score':'','human_reason':'','review_status':'UNLABELLED',
        })
        selected_global.add(isbn)
    if len(records)!=60: raise ValueError(f'Expected 60 rows, got {len(records)}')
    ids=[r['case_id'] for r in records]; isbns=[r['isbn13'] for r in records]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate case IDs')
    if len(isbns)!=len(set(isbns)): raise ValueError('Duplicate ISBNs')
    overlap=set(isbns)&seen
    if overlap: raise ValueError(f'Historical ISBN overlap: {sorted(overlap)}')
    output=args.output if args.output.is_absolute() else REPO_ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fields=list(records[0].keys())
    with output.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    print(f'Wrote {len(records)} fresh unseen cases: {output}')
    print(f'Historical ISBN exclusions: {len(seen)}')
    print('Duplicate ISBNs: 0')
    print('Historical ISBN overlap: 0')
    print('Judge calls: 0')

if __name__=='__main__': main()
