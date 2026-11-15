import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to files only, no window
import matplotlib.pyplot as plt
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


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python plot_results.py <results folder>")
    results_dir = Path(sys.argv[1])
    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    plot_headline(pd.read_csv(results_dir / "summary.csv"), out_dir / "headline.png")
    plot_benchmarks(pd.read_csv(results_dir / "summary_by_benchmark.csv"), out_dir / "benchmarks.png")
    print("Saved figures to", out_dir)


if __name__ == "__main__":
    main()
