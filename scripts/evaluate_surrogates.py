"""
Evaluate surrogate models on one representation of the generated algorithms

Each repeat (seed) reshuffles the cross-validation folds, and every fold's
scores are saved. Two ways of training are compared:

- within_benchmark: a separate model per benchmark. This is the main result,
  because fitness is only comparable between programs on the same benchmark
- pooled: one model across all benchmarks, scored overall and per benchmark

To evaluate another representation, build a table with the same metadata
columns and pass it with --table. Every other column is treated as a feature.
"""
import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent

METADATA_COLS = ["id", "run", "llm", "benchmark", "repeat", "generation", "fitness", "parent_ids", "code"]
METRICS = ["r2", "rmse", "mae", "spearman", "kendall", "top_10pct_precision"]
CONFIG_COLS = ["representation", "model", "training_scope", "split"]

# scored_on value for scores across every benchmark in the test fold
ALL_BENCHMARKS = "all"
N_SPLITS = 5
TOP_FRACTION = 0.1
# a smaller test set gives scores too noisy to be worth averaging
MIN_TEST_ROWS = 10

# the column whose values must not appear in both train and test
SPLIT_GROUPS = {"random": None, "by_run": "run", "by_llm": "llm", "by_benchmark": "benchmark"}
POOLED_SPLITS = ["random", "by_run", "by_llm", "by_benchmark"]
WITHIN_SPLITS = ["random", "by_run", "by_llm"]

# XGBoost's random_state stays fixed, so seeds only change the folds
MODELS = {
    "mean_baseline": lambda: DummyRegressor(strategy="mean"),
    "xgboost_default": lambda: XGBRegressor(random_state=0, n_jobs=-1),
    # the settings LLaMEA-SAGE uses once it has 20 or more programs
    "xgboost_sage": lambda: XGBRegressor(
        max_depth=4, n_estimators=200, learning_rate=0.1, random_state=0, n_jobs=-1
    ),
}


def top_k_precision(y_true, y_pred, rng, fraction=TOP_FRACTION):
    """
    share of the truly best programs that are also in the predicted best

    ties are broken at random, so a model that predicts one constant value
    scores about `fraction` (chance level) rather than following row order
    """
    k = math.ceil(fraction * len(y_true)) # how many programs to count as top 10%

    def top_positions(values):
        # a fresh shuffle each call, so the two rankings' ties break independently
        order = rng.permutation(len(values))
        # sorts from highest to lowest in the random order
        # then converts back to the original positions and keeps the top-k
        # stable keeps tied values in sorted order
        return order[np.argsort(-values[order], kind="stable")[:k]]

    # get the precision
    return len(set(top_positions(y_true)) & set(top_positions(y_pred))) / k


def score_predictions(y_true, y_pred, rng):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # rank correlation is undefined if either side has no variation
    can_rank = np.ptp(y_true) > 0 and np.ptp(y_pred) > 0

    # get the score predictions
    return {
        "r2": r2_score(y_true, y_pred) if np.ptp(y_true) > 0 else np.nan,
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred)[0] if can_rank else np.nan,
        "kendall": kendalltau(y_true, y_pred)[0] if can_rank else np.nan,
        "top_10pct_precision": top_k_precision(y_true, y_pred, rng),
    }


def feature_columns(table):
    # stop if a metadata column is missing
    missing = [c for c in METADATA_COLS if c not in table.columns]
    
    if missing:
        raise ValueError(f"Table is missing metadata columns: {missing}")

    # stop if fitness is NaN or infitinite
    if not np.isfinite(table["fitness"]).all():
        raise ValueError("Table has non-finite fitness; failed programs belong in the failures file")

    # treats every other column as a feature, forward-compatible for embeddings coming later
    features = [c for c in table.columns if c not in METADATA_COLS]
    
    # drop features with only one value -> should take it from 28 down to 21
    return [c for c in features if table[c].nunique() > 1]


def make_folds(table, split, seed):
    # return (train positions, test positions) pairs for one split strategy
    group_col = SPLIT_GROUPS[split]

    # random split
    if group_col is None:
        return list(KFold(N_SPLITS, shuffle=True, random_state=seed).split(table))

    # else split by group
    groups = table[group_col]
    n_splits = min(N_SPLITS, groups.nunique()) # ceiling set to 5 by default
    splitter = GroupKFold(n_splits, shuffle=True, random_state=seed)

    return list(splitter.split(table, groups=groups))


def cross_validate(table, features, make_model, split, seed, labels):
    rows = []

    for fold, (train_pos, test_pos) in enumerate(make_folds(table, split, seed)):
        train, test = table.iloc[train_pos], table.iloc[test_pos]
        
        model = make_model()
        model.fit(train[features], train["fitness"])
        predictions = pd.Series(model.predict(test[features]), index=test.index)

        by_benchmark = list(test.groupby("benchmark"))
        if labels["training_scope"] == "pooled":
            by_benchmark.append((ALL_BENCHMARKS, test))

        for scored_on, subset in by_benchmark:
            if len(subset) < MIN_TEST_ROWS:
                continue
            rng = np.random.default_rng([seed, fold])
            scores = score_predictions(subset["fitness"], predictions.loc[subset.index], rng)
            rows.append({
                **labels,
                "seed": seed,
                "fold": fold,
                "scored_on": scored_on,
                "n_train": len(train),
                "n_test": len(subset),
                **scores,
            })

    return rows


def run_evaluation(table, features, seeds, models=None, representation="engineered"):

    models = MODELS if models is None else models
    table = table.reset_index(drop=True)    # renumber the rows from 0
    rows = []

    for seed in seeds:
        print(f"Seed {seed}", flush=True)

        # add in a loading bar**
        for model_name, make_model in models.items():
            base = {"representation": representation, "model": model_name}
            for split in POOLED_SPLITS:
                labels = {**base, "training_scope": "pooled", "split": split}
                rows += cross_validate(table, features, make_model, split, seed, labels)
            for _, subset in table.groupby("benchmark"):
                for split in WITHIN_SPLITS:
                    labels = {**base, "training_scope": "within_benchmark", "split": split}
                    rows += cross_validate(subset, features, make_model, split, seed, labels)

    return pd.DataFrame(rows)


def summarise(metrics):

    """
    Reduce fold scores to one value per seed, then a mean and SD over seeds

    within-benchmark scores average each benchmark first, so every benchmark
    counts equally however many folds or programs it appears in. The SD only
    shows how much the choice of folds matters, not how sure we are overall
    """

    long = metrics.melt(
        id_vars=CONFIG_COLS + ["seed", "scored_on"], value_vars=METRICS, var_name="metric"
    )
    keys = CONFIG_COLS + ["metric", "seed"]

    is_overall = long["scored_on"] == ALL_BENCHMARKS
    overall = long[is_overall].groupby(keys)["value"].mean().reset_index()
    overall["score_view"] = "overall"

    per_benchmark = long[~is_overall].groupby(keys + ["scored_on"])["value"].mean().reset_index()
    within = per_benchmark.groupby(keys)["value"].mean().reset_index()
    within["score_view"] = "within_benchmark"

    per_seed = pd.concat([overall, within])

    summary = (
        per_seed.groupby(CONFIG_COLS + ["score_view", "metric"])["value"]
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )

    return summary


def summarise_by_benchmark(metrics):
    long = metrics[metrics["scored_on"] != ALL_BENCHMARKS].melt(
        id_vars=CONFIG_COLS + ["seed", "scored_on"], value_vars=METRICS, var_name="metric"
    )

    keys = CONFIG_COLS + ["scored_on", "metric"]
    per_seed = long.groupby(keys + ["seed"])["value"].mean().reset_index()

    return per_seed.groupby(keys)["value"].agg(mean="mean", sd="std").reset_index()


def main():

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", type=Path, default=ROOT / "data" / "sbox_table.csv")
    parser.add_argument("--representation", default="engineered")
    parser.add_argument("--seeds", type=int, default=10, help="number of repeats with reshuffled folds")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "surrogates")
    args = parser.parse_args()

    table = pd.read_csv(args.table)
    features = feature_columns(table)
    print(f"{len(table)} programs, {len(features)} features, {args.seeds} seeds")

    seeds = list(range(args.seeds))
    metrics = run_evaluation(table, features, seeds, representation=args.representation)
    summary = summarise(metrics)

    # every run gets its own folder, so earlier results are never overwritten
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = args.out / f"{stamp}-{args.representation}"
    out_dir.mkdir(parents=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    summarise_by_benchmark(metrics).to_csv(out_dir / "summary_by_benchmark.csv", index=False)

    # record how these results were made, so a run can be checked or repeated:
    # the command, a hash of the input table (it changes if the data does),
    # the seeds, the features used and the software versions

    config = {
        "command": sys.argv,
        "table": str(args.table),
        "table_sha256": hashlib.sha256(args.table.read_bytes()).hexdigest(),
        "representation": args.representation,
        "seeds": seeds,
        "n_programs": len(table),
        "features": features,
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "xgboost"]},
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print("Saved results to", out_dir)

    headline = summary[summary["metric"].isin(["r2", "spearman", "top_10pct_precision"])]
    pd.set_option("display.width", 160, "display.max_rows", None)
    print(headline.pivot_table(
        index=["training_scope", "split", "score_view", "model"], columns="metric", values="mean"
    ).round(3))


if __name__ == "__main__":
    main()
