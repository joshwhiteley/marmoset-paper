import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cm import get_cmap
from helpers.constants import LESS_SEVERE_CLUSTERS, SEVERE_CLUSTERS

# display names for legend (top to bottom)
BUCKET_ORDER = [
    "resolved",
    "resolved (scar)",
    "fibrotic",
    "necrotic",
    "cavitary",
]

BUCKET_ORDER.reverse() # whoops

# color scheme for plot
BUCKET_COLORS = {
    "resolved":         (0.9, 0.9, 0.9),
    "resolved (scar)":  (0.8, 0.8, 0.8),
    "fibrotic":         (0.6, 0.6, 0.6),
    "necrotic":         (0.35, 0.35, 0.35),
    "cavitary":         (0.0, 0.0, 0.0),
}

# label -> bucket
LABEL_TO_BUCKET = {
    # resolved
    "Not Visible":          "resolved",
    # resolved (scar)
    "Scar":                 "resolved (scar)",
    # fibrotic
    "Fibrotic":             "fibrotic",
    "Fibrotic Scar":        "fibrotic",
    "Fibrotic Consol":      "fibrotic",
    "Fibrotic Cavity":      "fibrotic",
    "Fibrotic HWC":         "fibrotic",
    # necrotic
    "Necrotic":             "necrotic",
    "Necrotic Consol":      "necrotic",
    # cavitary
    "Cavity":               "cavitary",
    "Cavity Consol":        "cavitary",
    "Cavity Fibrotic":      "cavitary",
}

# load data
df = pd.read_csv("data/marm_data_wide_clustered_classif.csv")

# toss in bucket
df["tp6_bucket"] = df["TP6_LesionType"].map(LABEL_TO_BUCKET)


def plot_split(ax, clusters, title, show_legend=True):
    # filter to requested clusters
    sub = df[df["cluster"].isin(clusters)].copy()
    # counters per (cluster, bucket)
    counts = (
        sub.groupby(["cluster", "tp6_bucket"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=BUCKET_ORDER, fill_value=0)
        .reindex(index=clusters, fill_value=0)
    )
    
    # percentage
    totals = counts.sum(axis=1).replace(0, np.nan)
    percent = counts.div(totals, axis=0).fillna(0.0) * 100
    
    # stacked bars
    bottom = np.zeros(len(percent), dtype=float)
    x = np.arange(len(percent))
    for bucket in BUCKET_ORDER:
        ax.bar(
            x,
            percent[bucket].to_numpy(),
            bottom=bottom,
            color=BUCKET_COLORS[bucket],
            edgecolor="white",
            linewidth=0.3,
            label=bucket if show_legend else None,
        )
        bottom += percent[bucket].to_numpy()

    ax.set_xticks(x, percent.index.astype(str))
    ax.set_ylim(0, 100)
    ax.set_xlabel("cluster number")
    ax.set_ylabel("% of lesions @ TP6")
    ax.set_title(title)
    ax.spines[["right", "top"]].set_visible(False)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharey=True)

plot_split(axes[0], LESS_SEVERE_CLUSTERS, "less severe lesions", show_legend=True)
plot_split(axes[1], SEVERE_CLUSTERS, "severe lesions", show_legend=False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="lesion type")
plt.tight_layout()
plt.savefig("manuscript-figures/1/figure_1d.svg", format='svg')
plt.show()