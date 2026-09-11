"""Figure 5 SHAP summaries and author-confirmed feature panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marmoset_paper.helpers.constants import COMPOUND_NAMES
from marmoset_paper.helpers.PlottingFunctions import plot_shap_summary

# The terminal dormancy phase was confirmed during publication cleanup.
HIGHLIGHTS = [
    ("tp6", "cellGRinf_dormancySimplePK_Termil"),
    ("tp4", "NeutralNequip_FBC90"),
]


def load_shap_rows(directory: Path, timepoint: str):
    values = np.load(directory / f"b_{timepoint}_values.npy", allow_pickle=False)
    features = pd.read_csv(directory / f"b_{timepoint}_features.csv")
    metadata = pd.read_csv(directory / f"b_{timepoint}_samples.csv")
    if values.shape != features.shape or len(features) != len(metadata):
        raise ValueError("SHAP values, features, and sample identifiers must align")
    return values, features, metadata


def plot_highlight(values, features, metadata, timepoint, feature, output):
    index = features.columns.get_loc(feature)
    shap_values = values[:, index]
    feature_values = features[feature].to_numpy()
    compounds = sorted(metadata.Compound.unique())
    jitter = np.random.default_rng(42).normal(0, 0.06, size=len(features))
    fig = plt.figure(figsize=(10, 10))
    grid = fig.add_gridspec(
        1 + (len(compounds) + 1) // 2, 2, height_ratios=[2] + [1] * ((len(compounds) + 1) // 2)
    )
    ax = fig.add_subplot(grid[0, :])
    ax.scatter(shap_values, jitter, color="0.8", s=15)
    finite = np.isfinite(feature_values)
    scatter = ax.scatter(
        shap_values[finite], jitter[finite], c=feature_values[finite], cmap="coolwarm", s=20
    )
    fig.colorbar(scatter, ax=ax, label="feature value")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set(
        title=f"{timepoint.upper()} MeanHU: {feature}", xlabel="SHAP contribution (HU)", yticks=[]
    )
    limits = ax.get_xlim()
    palette = plt.get_cmap("tab10")
    for i, compound in enumerate(compounds):
        panel = fig.add_subplot(grid[1 + i // 2, i % 2])
        mask = metadata.Compound == compound
        panel.scatter(shap_values, jitter, color="0.85", s=12, alpha=0.5)
        panel.scatter(
            shap_values[mask],
            jitter[mask],
            color=palette(i % 10),
            s=20,
            edgecolor="black",
            linewidth=0.4,
        )
        panel.axvline(0, color="black", linewidth=0.5)
        panel.set(title=COMPOUND_NAMES.get(compound, compound), xlim=limits, yticks=[])
    fig.tight_layout()
    stem = f"figure_5c_{timepoint}_{feature}"
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    metadata.assign(feature=feature, feature_value=feature_values, shap_value=shap_values).to_csv(
        output / f"{stem}_source.csv", index=False
    )


def generate(analysis_dir: Path, output: Path):
    directory = analysis_dir / "shap"
    values, features, metadata = load_shap_rows(directory, "tp6")
    plot_shap_summary(values, features, output / "figure_5a.svg", max_display=30)
    invitro = [i for i, name in enumerate(features) if not name.startswith("TP")]
    plot_shap_summary(
        values[:, invitro], features.iloc[:, invitro], output / "figure_5b.svg", max_display=25
    )
    contributions = metadata.assign(invitro_SHAP=values[:, invitro].sum(axis=1))
    contributions.to_csv(output / "invitro_contributions_by_lesion.csv", index=False)
    contributions.groupby("Compound")["invitro_SHAP"].agg(
        ["mean", "median", "std", "count"]
    ).sort_values("median").to_csv(output / "invitro_contributions_by_regimen.csv")
    for timepoint, feature in HIGHLIGHTS:
        values, features, metadata = load_shap_rows(directory, timepoint)
        plot_highlight(values, features, metadata, timepoint, feature, output)
