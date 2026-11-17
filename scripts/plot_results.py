import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SPLITS = ["random", "by_run", "by_llm"]
COLORS = {"mean_baseline": "#eb6834", "xgboost_sage": "#2a78d6"}
CHANCE = {"spearman": 0.0, "top_10pct_precision": 0.1}


def plot_headline(summary, out):
    rows = summary[(summary["training_scope"] == "within_benchmark") & summary["model"].isin(COLORS)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax, metric in zip(axes, CHANCE):
        for i, (model, color) in enumerate(COLORS.items()):
            m = rows[(rows["model"] == model) & (rows["metric"] == metric)].set_index("split").reindex(SPLITS)
            x = [j + (i - 0.5) * 0.38 for j in range(len(SPLITS))]
            ax.bar(x, m["mean"], width=0.36, yerr=m["sd"], color=color, label=model, capsize=3)
        # the score a model with no idea of the order would get
        ax.axhline(CHANCE[metric], color="#52514e", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(SPLITS)), SPLITS)
        ax.set_title(metric)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(frameon=False)
    fig.suptitle("Within-benchmark results (error bars: SD over seeds; dashed: chance)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_benchmarks(by_benchmark, out):
    rows = by_benchmark[
        (by_benchmark["training_scope"] == "within_benchmark")
        & (by_benchmark["model"] == "xgboost_sage")
        & (by_benchmark["metric"] == "spearman")
    ]
    grid = rows.pivot(index="scored_on", columns="split", values="mean")[SPLITS]

    # orange below 0 (worse than chance), grey at 0, blue above
    cmap = LinearSegmentedColormap.from_list("diverging", ["#eb6834", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(5, 6))
    image = ax.imshow(grid, cmap=cmap, vmin=-0.75, vmax=0.75)

    for (r, c), value in pd.DataFrame(grid.values).stack().items():
        ax.text(c, r, f"{value:.2f}", ha="center", va="center", color="white" if abs(value) > 0.5 else "#0b0b0b")
    ax.set_xticks(range(len(SPLITS)), SPLITS)
    ax.set_yticks(range(len(grid)), grid.index)
    ax.set_title("Spearman per benchmark (xgboost_sage)")
    fig.colorbar(image, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)

def plot_predictions(predictions, out):
    rows = predictions[
        (predictions["training_scope"] == "within_benchmark")
        & (predictions["split"] == "by_run")
        & (predictions["model"] == "xgboost_sage")
    ]
    benchmarks = sorted(rows["benchmark"].unique())
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))

    for ax, benchmark in zip(axes.flat, benchmarks):
        b = rows[rows["benchmark"] == benchmark]
        ax.scatter(b["fitness"], b["prediction"], s=8, alpha=0.3, color=COLORS["xgboost_sage"], linewidths=0)
        ax.set_title(benchmark)
        ax.spines[["top", "right"]].set_visible(False)

    fig.supxlabel("actual fitness")
    fig.supylabel("predicted fitness")
    fig.suptitle("Predicted vs actual fitness on unseen runs (xgboost_sage, first seed)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_folds(metrics, out):
    rows = metrics[
        (metrics["training_scope"] == "within_benchmark")
        & (metrics["model"] == "xgboost_sage")
    ]
    fig, ax = plt.subplots(figsize=(6, 4))

    for i, split in enumerate(SPLITS):
        scores = rows.loc[rows["split"] == split, "spearman"].dropna()
        # spread the dots sideways a little so they don't sit on top of each other
        x = i + np.random.default_rng(0).uniform(-0.2, 0.2, len(scores))
        ax.scatter(x, scores, s=8, alpha=0.25, color=COLORS["xgboost_sage"], linewidths=0)
        ax.plot([i - 0.3, i + 0.3], [scores.mean()] * 2, color="#0b0b0b", linewidth=2)

    ax.axhline(CHANCE["spearman"], color="#52514e", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(SPLITS)), SPLITS)
    ax.set_ylabel("Spearman")
    ax.set_title("Spearman for every benchmark, fold and seed (black: mean)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def plot_fitness(table, out):
    # left to right -> easiest benchmark to hardest
    order = table.groupby("benchmark")["fitness"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot([table.loc[table["benchmark"] == b, "fitness"] for b in order], showfliers=False)
    ax.set_xticks(range(1, len(order) + 1), order)
    ax.set_ylabel("fitness")
    ax.set_title("Fitness of every working program, by benchmark")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python plot_results.py <results folder>")
    results_dir = Path(sys.argv[1])
    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    plot_headline(pd.read_csv(results_dir / "summary.csv"), out_dir / "headline.png")
    plot_benchmarks(pd.read_csv(results_dir / "summary_by_benchmark.csv"), out_dir / "benchmarks.png")
    plot_folds(pd.read_csv(results_dir / "metrics.csv"), out_dir / "folds.png")
    
    # the table this run was evaluated on, as recorded in config.json
    table = pd.read_csv(json.loads((results_dir / "config.json").read_text())["table"])
    plot_fitness(table, out_dir / "fitness.png")
    if (results_dir / "predictions.csv").exists():  # older runs didn't save predictions
        plot_predictions(pd.read_csv(results_dir / "predictions.csv"), out_dir / "predictions.png")
    print("Saved figures to", out_dir)


if __name__ == "__main__":
    main()
