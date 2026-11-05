"""
export the headline results from one evaluate_surrogates.py results folder

keeps only models trained and scored within each benchmark, for the mean
baseline and the LLaMEA-SAGE settings in XGBoost, and writes them to headline.csv in
the same folder 
"""
import sys
from pathlib import Path

import pandas as pd

SPLITS = ["random", "by_run", "by_llm"]
MODELS = ["mean_baseline", "xgboost_sage"]
METRICS = ["r2", "spearman", "top_10pct_precision"]


def headline(summary):
    rows = summary[
        (summary["training_scope"] == "within_benchmark")
        & (summary["score_view"] == "within_benchmark")
        & summary["split"].isin(SPLITS)
        & summary["model"].isin(MODELS)
        & summary["metric"].isin(METRICS)
    ]
    if rows.empty:
        raise ValueError("No within-benchmark rows found in summary.csv")

    table = rows.pivot_table(index=["split", "model"], columns="metric", values="mean", dropna=False)
    # keep the order the splits and models are discussed in, not alphabetical
    order = pd.MultiIndex.from_product([SPLITS, MODELS], names=["split", "model"])
    table = table.reindex(order)[METRICS].round(3)

    # a blank cell means the score is undefined like spearman for a constant prediction
    return table.reset_index()


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python export_headline.py <results folder>")
    results_dir = Path(sys.argv[1])

    table = headline(pd.read_csv(results_dir / "summary.csv"))
    out = results_dir / "headline.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print("Saved", out)


if __name__ == "__main__":
    main()
