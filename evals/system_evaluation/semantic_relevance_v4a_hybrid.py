from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import math, re, unicodedata
from typing import Any, Sequence
import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"(?u)\b[^\W_]{2,}\b")
APOSTROPHES = str.maketrans({"'": "", "\u2019": ""})

def safe_text(value: Any) -> str:
    if value is None: return ""
    try:
        if pd.isna(value): return ""
    except Exception: pass
    return str(value).strip()

def canonical_isbn(value: Any) -> str:
    text=safe_text(value).strip('"').strip("'")
    if text.endswith('.0') and text[:-2].isdigit(): text=text[:-2]
    if not text: raise ValueError('ISBN is empty.')
    return text

def analyze_text(value: Any) -> list[str]:
    text=unicodedata.normalize('NFKC', safe_text(value)).casefold().translate(APOSTROPHES)
    return TOKEN_RE.findall(text)

def bm25_document_text(title: Any, description: Any) -> str:
    t,d=safe_text(title),safe_text(description)
    if not t and not d: raise ValueError('BM25 document cannot have both title and description empty.')
    return f"{t}\n{d}"

@dataclass(frozen=True)
class BM25Config:
    k1: float=1.5
    b: float=0.75
    candidate_depth: int=50
    def __post_init__(self):
        if float(self.k1)!=1.5: raise ValueError('Frozen v4a contract requires BM25 k1=1.5.')
        if float(self.b)!=0.75: raise ValueError('Frozen v4a contract requires BM25 b=0.75.')
        if int(self.candidate_depth)!=50: raise ValueError('Frozen v4a contract requires BM25 candidate_depth=50.')

class BM25Index:
    def __init__(self, *, isbn13: Sequence[Any], title: Sequence[Any], description: Sequence[Any], config: BM25Config|None=None):
        self.config=config or BM25Config()
        if not (len(isbn13)==len(title)==len(description)): raise ValueError('isbn13, title, and description must have equal length.')
        if not isbn13: raise ValueError('BM25 corpus is empty.')
        self.isbns=[canonical_isbn(x) for x in isbn13]
        if len(set(self.isbns))!=len(self.isbns): raise ValueError('Duplicate ISBN13 values are not allowed in the BM25 corpus.')
        self.doc_tf=[]; self.doc_len=[]; df=defaultdict(int)
        for t,d in zip(title,description):
            tokens=analyze_text(bm25_document_text(t,d))
            if not tokens: raise ValueError('BM25 document analyzed to zero tokens.')
            counts=Counter(tokens); self.doc_tf.append(counts); self.doc_len.append(len(tokens))
            for tok in counts: df[tok]+=1
        self.n_docs=len(self.isbns); self.avgdl=sum(self.doc_len)/self.n_docs; self.df=dict(df)
    def idf(self, token:str)->float:
        df=self.df.get(token,0); n=self.n_docs
        return math.log(1.0 + (n-df+0.5)/(df+0.5))
    def score_query(self, query:Any)->np.ndarray:
        scores=np.zeros(self.n_docs,dtype=float); k1=float(self.config.k1); b=float(self.config.b)
        for tok in analyze_text(query):
            idf=self.idf(tok)
            for i,tfmap in enumerate(self.doc_tf):
                tf=tfmap.get(tok,0)
                if not tf: continue
                dl=float(self.doc_len[i]); denom=tf+k1*(1-b+b*dl/self.avgdl)
                scores[i]+=idf*(tf*(k1+1))/denom
        return scores
    def rank_query(self, query:Any, *, top_k:int=50)->pd.DataFrame:
        if int(top_k)!=50: raise ValueError('Frozen v4a contract requires BM25 top_k=50.')
        rows=[{'isbn13':isbn,'bm25_score':float(score)} for isbn,score in zip(self.isbns,self.score_query(query)) if float(score)>0]
        rows.sort(key=lambda r:(-r['bm25_score'],r['isbn13'])); rows=rows[:top_k]
        for rank,row in enumerate(rows,1): row['bm25_rank']=rank
        return pd.DataFrame(rows,columns=['isbn13','bm25_score','bm25_rank'])

def _rank_or_inf(v:Any)->float:
    try:
        if pd.isna(v): return math.inf
    except Exception: pass
    return float(v)

def build_hybrid_union(dense:pd.DataFrame,bm25:pd.DataFrame,*,rrf_k:float=60.0)->pd.DataFrame:
    if float(rrf_k)!=60.0: raise ValueError('Frozen v4a contract requires hybrid RRF k=60.')
    for cols,frame,label in [({'isbn13','dense_rank'},dense,'dense'),({'isbn13','bm25_rank','bm25_score'},bm25,'BM25')]:
        miss=cols-set(frame.columns)
        if miss: raise ValueError(f'{label} frame missing: {sorted(miss)}')
    d=dense.copy(); b=bm25.copy(); d['isbn13']=d['isbn13'].map(canonical_isbn); b['isbn13']=b['isbn13'].map(canonical_isbn)
    if d['isbn13'].duplicated().any() or b['isbn13'].duplicated().any(): raise ValueError('Duplicate ISBN in retrieval branch.')
    m=d.merge(b[['isbn13','bm25_rank','bm25_score']],on='isbn13',how='outer',validate='one_to_one',indicator=True)
    m['candidate_source']=m['_merge'].map({'left_only':'dense_only','right_only':'bm25_only','both':'both'}); m=m.drop(columns=['_merge'])
    dense_c=np.where(m['dense_rank'].notna(),1/(rrf_k+pd.to_numeric(m['dense_rank'],errors='raise')),0.0)
    bm25_c=np.where(m['bm25_rank'].notna(),1/(rrf_k+pd.to_numeric(m['bm25_rank'],errors='raise')),0.0)
    m['hybrid_retrieval_rrf_score']=dense_c+bm25_c
    m['_dense']=m['dense_rank'].map(_rank_or_inf); m['_bm25']=m['bm25_rank'].map(_rank_or_inf); m['_isbn']=m['isbn13'].map(canonical_isbn)
    m=m.sort_values(['hybrid_retrieval_rrf_score','_dense','_bm25','_isbn'],ascending=[False,True,True,True],kind='stable').reset_index(drop=True)
    m['hybrid_retrieval_rank']=np.arange(1,len(m)+1,dtype=int)
    return m.drop(columns=['_dense','_bm25','_isbn'])

def assign_v4a_reranker_rank(frame:pd.DataFrame)->pd.DataFrame:
    req={'query_id','isbn13','hybrid_retrieval_rank','reranker_score'}; miss=req-set(frame.columns)
    if miss: raise ValueError(f'Missing reranker columns: {sorted(miss)}')
    out=[]
    for _,g in frame.groupby('query_id',sort=False):
        g=g.copy(); g['_isbn']=g['isbn13'].map(canonical_isbn)
        g=g.sort_values(['reranker_score','hybrid_retrieval_rank','_isbn'],ascending=[False,True,True],kind='stable')
        g['reranker_rank']=np.arange(1,len(g)+1,dtype=int); out.append(g.drop(columns=['_isbn']))
    return pd.concat(out,ignore_index=True)

def rank_v4a(frame:pd.DataFrame,*,rrf_k:float=60.0)->pd.DataFrame:
    if float(rrf_k)!=60.0: raise ValueError('Frozen v4a contract requires final RRF k=60.')
    r=assign_v4a_reranker_rank(frame)
    r['final_rrf_score']=1/(rrf_k+r['hybrid_retrieval_rank'].astype(float))+1/(rrf_k+r['reranker_rank'].astype(float))
    out=[]
    for _,g in r.groupby('query_id',sort=False):
        g=g.copy(); g['_isbn']=g['isbn13'].map(canonical_isbn)
        g=g.sort_values(['final_rrf_score','reranker_rank','hybrid_retrieval_rank','_isbn'],ascending=[False,True,True,True],kind='stable')
        g['v4a_rank']=np.arange(1,len(g)+1,dtype=int); out.append(g.drop(columns=['_isbn']))
    return pd.concat(out,ignore_index=True)
