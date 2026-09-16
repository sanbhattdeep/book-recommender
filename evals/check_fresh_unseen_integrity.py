"""Integrity checker for fresh unseen split and holdout release."""
from __future__ import annotations
import argparse
from fresh_unseen_contract import *

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['split','validation','holdout'],default='split'); args=p.parse_args()
    manifest=validate_split_contract(require_release=(args.stage=='holdout'))
    print('Fresh unseen evaluation integrity')
    print('--------------------------------')
    print('PASS  exact frozen v0.18 behavior hashes')
    print('PASS  historical exclusion manifest')
    print('PASS  fully labelled 60-case pool')
    print('PASS  exact 30/30 disjoint split membership')
    print('PASS  split frozen before judge output')
    print('PASS  validation/holdout file hashes')
    if args.stage=='holdout': print('PASS  validation-gate release token')

if __name__=='__main__': main()
