"""Check v2 fresh-unseen pool before human labeling."""
from __future__ import annotations
import pandas as pd
from fresh_unseen_contract import POOL_FILE, verify_behavior_hashes, validate_pool_dataframe

def main():
    verify_behavior_hashes()
    df=pd.read_csv(POOL_FILE, dtype={'isbn13':str}, encoding='utf-8')
    validate_pool_dataframe(df, require_labels=False)
    print('Fresh unseen pool integrity')
    print('---------------------------')
    print('PASS  frozen v0.18 behavior hashes')
    print('PASS  60 cases / 12 queries x 5')
    print('PASS  60 unique pool ISBNs')
    print('PASS  zero overlap with historical exclusion manifest')
    print('PASS  labels blank / review_status=UNLABELLED')

if __name__=='__main__': main()
