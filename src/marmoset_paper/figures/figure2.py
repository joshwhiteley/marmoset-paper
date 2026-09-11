"""Figure 2B combination-response heatmaps (historically named figure_2a)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from marmoset_paper.helpers.constants import COMPOUND_NAMES, FXC50_COLS


def generate(data_dir: Path, output: Path):
    data = pd.read_csv(data_dir / "in_vitro_modeling.csv").set_index("Drug")
    if not data.index.is_unique:
        raise ValueError("Figure 2 requires one row per regimen")
    data = data.loc[data.index.isin(COMPOUND_NAMES)].sort_values("FxC50_7HcellPK")
    data.to_csv(output / "figure_2b_source.csv")
    for group, columns in FXC50_COLS.items():
        values = data[list(columns)].apply(pd.to_numeric, errors="raise")
        annotations = np.where(values.abs() > 2.5, "*", "")
        cmap = plt.get_cmap("RdBu_r").copy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_facecolor("#D3D3D3")
        sns.heatmap(
            values,
            cmap=cmap,
            vmin=-2.5,
            vmax=2.5,
            annot=annotations,
            fmt="s",
            linewidths=0.5,
            linecolor="grey",
            ax=ax,
            cbar_kws={"label": "FxC50"},
            xticklabels=list(columns.values()),
            yticklabels=[COMPOUND_NAMES[compound] for compound in values.index],
        )
        ax.set(title=f"FxC50 values — {group}", xlabel="", ylabel="")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.savefig(output / f"figure_2b_{group}.svg", transparent=True, bbox_inches="tight")
        plt.close(fig)
