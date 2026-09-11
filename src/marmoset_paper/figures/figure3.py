"""Figure 3A and the corresponding less-severe heatmap."""

from pathlib import Path

import pandas as pd

from marmoset_paper.analysis.correlations import PATHOLOGY
from marmoset_paper.helpers.CorrelationPlottingFunctions import (
    plot_correlation_heatmap,
    plot_correlation_legend,
)


def generate(analysis_dir: Path, output: Path):
    for severity in ["severe", "less_severe"]:
        tables = [
            pd.read_csv(
                analysis_dir / "correlations" / f"{severity}_{metric}_values_mean.csv",
                index_col="feature",
            )[PATHOLOGY]
            for metric in ["rho", "p"]
        ]
        plot_correlation_heatmap(
            *tables, output / f"figure_3a_{severity}.svg", severity.replace("_", " ")
        )
    plot_correlation_legend(output / "figure_3_legend.svg")
