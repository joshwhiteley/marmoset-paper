"""Figure 3 correlation heatmaps and the three manuscript scatterplots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from marmoset_paper.analysis.correlations import PATHOLOGY
from marmoset_paper.helpers.CorrelationPlottingFunctions import (
    plot_correlation_heatmap,
    plot_correlation_legend,
)

# These pairs reproduce the three annotated correlations in the manuscript after
# normalizing OPC/QBS identifiers. Both simple-condition examples use cellular PK.
SCATTER_PAIRS = [
    ("FxC50_cholesterolCellPK_Constant", "TP6_MeanHU"),
    ("FxC50_butyrateCellPK_Terminal", "delta_MeanHU"),
    ("FxC50_5NcellPK", "delta_MeanHU"),
]


def generate(analysis_dir: Path, output: Path):
    directory = analysis_dir / "correlations"
    for severity in ["severe", "less_severe"]:
        tables = [
            pd.read_csv(directory / f"{severity}_{metric}_values_mean.csv", index_col="feature")[
                PATHOLOGY
            ]
            for metric in ["rho", "p"]
        ]
        plot_correlation_heatmap(
            *tables, output / f"figure_3a_{severity}.svg", severity.replace("_", " ")
        )
    plot_correlation_legend(output / "figure_3_legend.svg")
    invitro = pd.read_csv(directory / "in_vitro_combinations.csv", index_col="Drug")
    means = pd.read_csv(directory / "severe_compound_means.csv", index_col="Compound")
    for index, (feature, outcome) in enumerate(SCATTER_PAIRS, 1):
        pair = pd.concat([invitro[feature], means[outcome]], axis=1, join="inner")
        pair = pair[np.isfinite(pair).all(axis=1)]
        rho, p = spearmanr(pair[feature], pair[outcome])
        pair.rename_axis("Compound").to_csv(output / f"figure_3b_{index}_source.csv")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(pair[feature], pair[outcome], color="black", s=20)
        ax.set(
            xlabel=feature,
            ylabel=outcome,
            title=f"ρ = {rho:.3f}; p = {p:.3f}; n = {len(pair)} regimens",
        )
        ax.spines[["right", "top"]].set_visible(False)
        fig.savefig(output / f"figure_3b_{index}.svg", bbox_inches="tight")
        plt.close(fig)
