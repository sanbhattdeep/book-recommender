"""Create a formatted Excel labeling workbook from the blind CSV.

Recommended command:
  uv run --with xlsxwriter python evals/prepare_fresh_unseen_labeling_workbook.py
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from fresh_unseen_contract import REPO_ROOT, RUBRIC_FILE

BLIND=REPO_ROOT/'evals'/'datasets'/'semantic_relevance_fresh_unseen_pool.v2.0.0.BLIND_LABELING.csv'
OUT=REPO_ROOT/'evals'/'datasets'/'semantic_relevance_fresh_unseen_pool.v2.0.0.BLIND_LABELING.xlsx'

def main():
    try:
        import xlsxwriter  # noqa: F401
    except ImportError as e:
        raise RuntimeError('xlsxwriter is required. Run with: uv run --with xlsxwriter python evals/prepare_fresh_unseen_labeling_workbook.py') from e
    df=pd.read_csv(BLIND,encoding='utf-8',keep_default_na=False)
    rubric=json.loads(RUBRIC_FILE.read_text(encoding='utf-8'))
    with pd.ExcelWriter(OUT,engine='xlsxwriter') as writer:
        df.to_excel(writer,sheet_name='Labeling',index=False)
        wb=writer.book; ws=writer.sheets['Labeling']
        header=wb.add_format({'bold':True,'bg_color':'#1F4E78','font_color':'white','border':1,'valign':'top'})
        wrap=wb.add_format({'text_wrap':True,'valign':'top','border':1})
        score_fmt=wb.add_format({'align':'center','valign':'top','border':1})
        for c,name in enumerate(df.columns): ws.write(0,c,name,header)
        widths={'blind_label_id':12,'query':42,'title':34,'authors':28,'description':80,'human_score':13,'human_reason':70,'review_status':16}
        for c,name in enumerate(df.columns):
            ws.set_column(c,c,widths.get(name,20), score_fmt if name=='human_score' else wrap)
        ws.freeze_panes(1,0); ws.autofilter(0,0,len(df),len(df.columns)-1)
        score_col=df.columns.get_loc('human_score'); status_col=df.columns.get_loc('review_status')
        ws.data_validation(1,score_col,len(df),score_col,{'validate':'list','source':[0,1,2,3,4]})
        ws.data_validation(1,status_col,len(df),status_col,{'validate':'list','source':['UNLABELLED','LABELLED']})
        ins=wb.add_worksheet('Instructions'); writer.sheets['Instructions']=ins
        title_fmt=wb.add_format({'bold':True,'font_size':16,'bg_color':'#D9EAF7'})
        sub=wb.add_format({'bold':True,'font_size':12}); txt=wb.add_format({'text_wrap':True,'valign':'top'})
        ins.set_column('A:A',24); ins.set_column('B:B',110)
        ins.write('A1','Blind semantic-relevance labeling',title_fmt)
        ins.merge_range('A1:B1','Blind semantic-relevance labeling',title_fmt)
        instructions=[
            ('Goal','Judge each query-book pair independently from the supplied description only.'),
            ('Evidence','Book description is primary evidence. Title/author may support but must not substitute for missing description evidence. No external knowledge.'),
            ('Tie-break','When genuinely between adjacent scores, choose the lower score unless the higher score is clearly supported.'),
            ('Blindness','Do not use retrieval rank, candidate source, ISBN, old judge outputs, or old human labels. Opaque B### IDs intentionally hide sampling identity.'),
            ('Completion','Every row requires human_score 0-4, a non-empty human_reason, and review_status=LABELLED.'),
        ]
        row=2
        for k,v in instructions: ins.write(row,0,k,sub); ins.write(row,1,v,txt); row+=2
        ins.write(row,0,'Score',sub); ins.write(row,1,'Definition',sub); row+=1
        for score in [4,3,2,1,0]:
            item=rubric['scores'][str(score)]
            ins.write(row,0,f"{score} — {item['label']}",sub); ins.write(row,1,item['definition'],txt); row+=1
    print(f'Wrote blind labeling workbook: {OUT}')

if __name__=='__main__': main()
