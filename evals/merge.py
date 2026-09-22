import pandas as pd

generated = pd.read_csv(
    "evals/datasets/semantic_relevance_calibration.v0.1.0.csv",
    dtype={"isbn13": str},
)

labelled = pd.read_csv(
    "evals/datasets/semantic_relevance_calibration.labelled-temp.csv",
    encoding="latin1",
)

labels = labelled[
    [
        "case_id",
        "human_score",
        "human_reason",
        "review_status",
    ]
]

final = generated.drop(
    columns=[
        "human_score",
        "human_reason",
        "review_status",
    ]
).merge(
    labels,
    on="case_id",
    how="left",
    validate="one_to_one",
)

# Important sanity check:
assert len(final) == 60
assert final["human_score"].notna().all()
assert final["human_reason"].notna().all()

final.to_csv(
    "evals/datasets/semantic_relevance_calibration.v0.1.0.csv",
    index=False,
    encoding="utf-8",
)