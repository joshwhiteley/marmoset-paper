"""Baseline clustering panels, using the deposited coordinates and cluster labels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage

from marmoset_paper.helpers.constants import (
    LESS_SEVERE_CLUSTERS,
    LESS_SEVERE_COLOR_TUPLE,
    PALETTE_16,
    SEVERE_CLUSTERS,
    SEVERE_COLOR_TUPLE,
)

FEATURES = ["TP2_MeanSUV", "TP2_MeanHU", "TP2_HardVol", "TP2_SoftVol", "TP2_StandDevHU"]
BUCKETS = ["cavitary", "necrotic", "fibrotic", "resolved (scar)", "resolved"]
BUCKET_COLORS = ["0.0", "0.35", "0.6", "0.8", "0.9"]
LABEL_TO_BUCKET = {
    "Not Visible": "resolved",
    "Scar": "resolved (scar)",
    "Fibrotic": "fibrotic",
    "Fibrotic Scar": "fibrotic",
    "Fibrotic Consol": "fibrotic",
    "Fibrotic Cavity": "fibrotic",
    "Fibrotic HWC": "fibrotic",
    "Necrotic": "necrotic",
    "Necrotic Consol": "necrotic",
    "Cavity": "cavitary",
    "Cavity Consol": "cavitary",
    "Cavity Fibrotic": "cavitary",
}


def cluster_medians(data: pd.DataFrame) -> pd.DataFrame:
    """Median standardized baseline features; sample SD matches the original Polars code."""
    values = data[FEATURES]
    if not np.isfinite(values).all().all():
        raise ValueError("Cluster medians require finite baseline measurements")
    std = values.std(ddof=1)
    scaled = ((values - values.mean()) / std.replace(0, np.nan)).fillna(0)
    return scaled.groupby(data.cluster).median().sort_index().T.rename_axis("feature")


def plot_umap(data, colors, labels, output):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data.UMAP1, data.UMAP2, c=colors, s=100, edgecolor="k", linewidth=0.2)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=label)
        for label, color in labels
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set(xlabel="UMAP1", ylabel="UMAP2", xticks=[], yticks=[])
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def generate(data_dir: Path, output: Path):
    clustered = pd.read_csv(data_dir / "marm_data_wide_clustered.csv")
    classified = pd.read_csv(data_dir / "marm_data_wide_clustered_classif.csv")
    if set(clustered.cluster) != set(range(16)):
        raise ValueError("Figure 1 expects the deposited 16-cluster solution")
    if not set(classified.classif) <= {"cool", "hot"}:
        raise ValueError("Unrecognized severity classification")
    # Keep the deposited coordinates used by each original panel. The two tables
    # contain different embeddings; see docs/reproducibility.md before reconciling them.
    plot_umap(
        clustered,
        [PALETTE_16[c] for c in clustered.cluster],
        [(str(c), PALETTE_16[c]) for c in range(16)],
        output / "figure_1a.svg",
    )
    medians = cluster_medians(clustered)
    medians.to_csv(output / "figure_1b_cluster_medians.csv")
    grid = sns.clustermap(
        medians,
        col_linkage=linkage(medians.T, method="average", metric="cosine"),
        row_cluster=False,
        cmap="YlOrRd",
        figsize=(12, 8),
    )
    grid.ax_heatmap.set(xlabel="cluster", ylabel="feature")
    grid.savefig(output / "figure_1b.svg")
    plt.close(grid.fig)
    colors = {"cool": LESS_SEVERE_COLOR_TUPLE, "hot": SEVERE_COLOR_TUPLE}
    plot_umap(
        classified,
        [colors[label] for label in classified.classif],
        list(colors.items()),
        output / "figure_1c.svg",
    )

    classified["bucket"] = classified.TP6_LesionType.map(LABEL_TO_BUCKET)
    excluded = classified[classified.bucket.isna()]
    # Do not silently reclassify ambiguous labels during a code cleanup.
    excluded[["Compound", "MarmID", "Lesion", "TP6_LesionType"]].to_csv(
        output / "figure_1d_unmapped_labels.csv", index=False
    )
    if len(excluded):
        print(
            f"Figure 1D: {len(excluded)} unmapped lesion labels excluded; see the exported table."
        )
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)
    tables = []
    for ax, clusters, title in zip(
        axes, [LESS_SEVERE_CLUSTERS, SEVERE_CLUSTERS], ["less severe lesions", "severe lesions"]
    ):
        subset = classified[classified.cluster.isin(clusters)]
        counts = pd.crosstab(subset.cluster, subset.bucket).reindex(
            index=clusters, columns=BUCKETS, fill_value=0
        )
        percent = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
        tables.append(counts)
        bottom = np.zeros(len(clusters))
        for bucket, color in zip(BUCKETS, BUCKET_COLORS):
            ax.bar(
                range(len(clusters)),
                percent[bucket],
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.3,
                label=bucket,
            )
            bottom += percent[bucket].to_numpy()
        ax.set_xticks(range(len(clusters)), clusters)
        ax.set(
            title=title, xlabel="cluster number", ylabel="% of mapped lesions at TP6", ylim=(0, 100)
        )
        ax.spines[["right", "top"]].set_visible(False)
    pd.concat(tables).to_csv(output / "figure_1d_counts.csv")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="lesion type")
    fig.tight_layout()
    fig.savefig(output / "figure_1d.svg")
    plt.close(fig)
