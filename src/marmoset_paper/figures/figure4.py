"""Figure 4 from the metrics and identified test lesions of a single analysis run."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from marmoset_paper.helpers.PlottingFunctions import STRATEGIES, plot_performance_metrics

REGIMENS = ["BDQ+LIN+PRE", "EMB+MOX+PZA+RIF"]


def generate(analysis_dir: Path, output: Path):
    metrics = pd.read_csv(analysis_dir / "models/metrics.csv")
    predictions = pd.read_csv(analysis_dir / "models/predictions.csv")
    keys = ["strategy", "Compound", "MarmID", "Lesion", "timepoint"]
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate identified predictions")
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    plot_performance_metrics(metrics, axes[0])
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 0].text(-0.15, 1.1, "A", transform=axes[0, 0].transAxes, fontsize=18)
    selected = []
    for column, compound in enumerate(REGIMENS):
        rows = predictions[(predictions.Compound == compound) & (predictions.strategy == "m")]
        lesions = rows[["MarmID", "Lesion"]].drop_duplicates().sort_values(["MarmID", "Lesion"])
        if lesions.empty:
            raise ValueError(f"No held-out lesions for {compound}")
        indices = np.random.RandomState(5674).choice(
            len(lesions), size=min(3, len(lesions)), replace=False
        )
        lesions = lesions.iloc[sorted(indices)]
        selected.append(lesions.assign(Compound=compound))
        for row_index, (strategy, (label, _)) in enumerate(STRATEGIES.items(), start=1):
            ax = axes[row_index, column]
            for color, lesion in zip(["#E91E63", "#4CAF50", "#2196F3"], lesions.itertuples()):
                frame = predictions[
                    (predictions.Compound == compound)
                    & (predictions.strategy == strategy)
                    & (predictions.MarmID == lesion.MarmID)
                    & (predictions.Lesion == lesion.Lesion)
                ].sort_values("timepoint")
                if frame.timepoint.tolist() != [3, 4, 5, 6]:
                    raise ValueError(f"Incomplete prediction trajectory for {compound}, {lesion}")
                baseline = frame.baseline.iloc[0]
                weeks = [0, 2, 4, 6, 8]
                ax.plot(
                    weeks,
                    [baseline, *frame.actual],
                    "o-",
                    color=color,
                    label=f"{lesion.MarmID} / L{lesion.Lesion}",
                )
                ax.plot(weeks, [baseline, *frame.predicted], "o--", color=color)
            ax.set(
                title=f"{label}: {compound}",
                xlabel="weeks from treatment start",
                ylabel="HU",
                xticks=weeks,
            )
            ax.legend(fontsize=8)
            ax.spines[["right", "top"]].set_visible(False)
    axes[1, 0].text(-0.15, 1.1, "B", transform=axes[1, 0].transAxes, fontsize=18)
    fig.legend(
        handles=[
            Line2D([], [], color="black", linestyle=style, label=label)
            for style, label in [("-", "observed"), ("--", "predicted from observed history")]
        ],
        loc="lower center",
        ncol=2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output / "figure_4.svg", bbox_inches="tight")
    fig.savefig(output / "figure_4.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    pd.concat(selected).to_csv(output / "figure_4_selected_lesions.csv", index=False)
