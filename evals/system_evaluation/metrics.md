                         QUERY + BOOK
                              │
                              ▼
                semantic_relevance_facet_judge.py
                              │
                    semantic evidence
                              │
                              ▼
              semantic_relevance_facet_scoring.py
                              │
                         score 0–4
                              │
             ┌────────────────┴─────────────────┐
             │                                  │
             ▼                                  ▼
   JUDGE CALIBRATION                    SYSTEM EVALUATION
             │                                  │
human_score vs judge_score             ranked judge scores
             │                                  │
             ▼                                  ▼
analyze_judge_calibration.py    analyze_semantic_relevance_
                                system_eval_v2.py
             │                                  │
    exact agreement                     Mean@10
    within ±1                           Clear/strong
    weighted kappa                      Irrelevant
    precision                           Hit@5
    recall                              Top1
    F1                                  MRR
    confusion matrix                    nDCG
                                        catastrophic rate
                                                │
                                                ▼
                                   bootstrap_query_clusters()
                                                │
                                                ▼
                                         95% CIs
                                                │
                                                ▼
                                           gate_pass()

v1 query metrics ─┐
                  ├─> compare_semantic_relevance_system_eval_v1_v2.py
v2 query metrics ─┘
                         │
                         ├─ paired bootstrap CI
                         ├─ W/T/L
                         └─ gate transitions                                           