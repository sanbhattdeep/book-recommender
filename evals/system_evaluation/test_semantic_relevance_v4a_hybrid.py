from __future__ import annotations
import math, unittest
import pandas as pd
from semantic_relevance_v4a_hybrid import BM25Config,BM25Index,analyze_text,build_hybrid_union,rank_v4a

class AnalyzerTests(unittest.TestCase):
    def test_contract(self):
        self.assertEqual(analyze_text("A Doctor’s Decision-Making, NOT politics."),['doctors','decision','making','not','politics'])
    def test_no_stop_word_removal(self):
        self.assertEqual(analyze_text('history of food but not recipes'),['history','of','food','but','not','recipes'])

class BM25Tests(unittest.TestCase):
    def test_hand_calculated_score(self):
        idx=BM25Index(isbn13=['1001','1002','1003'],title=['alpha beta beta','alpha gamma','delta'],description=['','',''])
        scores=idx.score_query('beta'); idf=math.log(1+(3-1+0.5)/(1+0.5)); avgdl=2.0; tf=2.0; dl=3.0; k1=1.5; b=0.75
        expected=idf*(tf*(k1+1))/(tf+k1*(1-b+b*dl/avgdl))
        self.assertAlmostEqual(float(scores[0]),expected,places=12); self.assertEqual(float(scores[1]),0.0)
    def test_repeated_query_term(self):
        idx=BM25Index(isbn13=['1','2'],title=['climate science','climate politics'],description=['',''])
        self.assertAlmostEqual(float(idx.score_query('climate climate')[0]),2*float(idx.score_query('climate')[0]),places=12)
    def test_tie_break_isbn(self):
        idx=BM25Index(isbn13=['2002','2001'],title=['climate','climate'],description=['science','science'])
        r=idx.rank_query('climate science'); self.assertEqual(r['isbn13'].tolist(),['2001','2002'])
    def test_zero_overlap(self):
        idx=BM25Index(isbn13=['1','2'],title=['climate science','food history'],description=['',''])
        self.assertEqual(len(idx.rank_query('quantum')),0)
    def test_parameter_tuning_rejected(self):
        with self.assertRaises(ValueError): BM25Config(k1=1.2)
        with self.assertRaises(ValueError): BM25Config(b=0.6)
        with self.assertRaises(ValueError): BM25Config(candidate_depth=25)

class HybridTests(unittest.TestCase):
    def test_union_sources(self):
        d=pd.DataFrame({'isbn13':['A','B','C'],'dense_rank':[1,2,3]}); b=pd.DataFrame({'isbn13':['B','D','E'],'bm25_score':[4.,3.,2.],'bm25_rank':[1,2,3]})
        u=build_hybrid_union(d,b); self.assertEqual(set(u['isbn13']),{'A','B','C','D','E'}); src=dict(zip(u['isbn13'],u['candidate_source'])); self.assertEqual(src['B'],'both'); self.assertEqual(src['D'],'bm25_only')
    def test_union_not_truncated(self):
        d=pd.DataFrame({'isbn13':[f'D{i}' for i in range(1,51)],'dense_rank':range(1,51)}); b=pd.DataFrame({'isbn13':[f'B{i}' for i in range(1,51)],'bm25_score':[100-i for i in range(50)],'bm25_rank':range(1,51)})
        self.assertEqual(len(build_hybrid_union(d,b)),100)
    def test_shared_dual_contribution(self):
        d=pd.DataFrame({'isbn13':['A','B'],'dense_rank':[1,2]}); b=pd.DataFrame({'isbn13':['B','C'],'bm25_score':[2.,1.],'bm25_rank':[1,2]}); u=build_hybrid_union(d,b)
        got=float(u.loc[u['isbn13']=='B','hybrid_retrieval_rrf_score'].iloc[0]); self.assertAlmostEqual(got,1/62+1/61,places=12)
    def test_final_rank_structure(self):
        f=pd.DataFrame({'query_id':['Q']*3,'isbn13':['A','B','C'],'hybrid_retrieval_rank':[1,2,3],'reranker_score':[.1,.9,.8]}); r=rank_v4a(f); self.assertEqual(sorted(r['v4a_rank'].tolist()),[1,2,3])
    def test_rrf_k_frozen(self):
        d=pd.DataFrame({'isbn13':['A'],'dense_rank':[1]}); b=pd.DataFrame({'isbn13':['A'],'bm25_score':[1.],'bm25_rank':[1]})
        with self.assertRaises(ValueError): build_hybrid_union(d,b,rrf_k=30)

if __name__=='__main__': unittest.main()
