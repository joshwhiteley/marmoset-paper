"""Figure 5A/B SHAP summaries and explicitly selected feature panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marmoset_paper.helpers.PlottingFunctions import plot_shap_summary

# Features selected in the historical highlight script; not automatically ranked.
HIGHLIGHTS = [
    "casGRinf_dormancySimplePK_Constant",
    "cellAUC25_cholesterolSimplePK_Termil",
    "NeutralHcellPK_AUC50",
    "NeutralNequip_FBC50",
]


def generate(analysis_dir: Path, output: Path):
    directory = analysis_dir / "shap"
    values = np.load(directory / "b_tp6_values.npy", allow_pickle=False)
    features = pd.read_csv(directory / "b_tp6_features.csv")
    metadata = pd.read_csv(directory / "b_tp6_samples.csv")
    if len(features) != len(metadata):
        raise ValueError("SHAP sample identifiers do not align with the feature rows")
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
    for feature in HIGHLIGHTS:
        if feature not in features:
            raise ValueError(f"Missing selected SHAP feature: {feature}")
        index = features.columns.get_loc(feature)
        fig, ax = plt.subplots(figsize=(8, 5))
        for compound in sorted(metadata.Compound.unique()):
            mask = metadata.Compound == compound
            ax.scatter(
                features.loc[mask, feature], values[mask, index], s=15, label=compound, alpha=0.7
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set(xlabel=feature, ylabel="SHAP contribution to TP6 MeanHU (HU)")
        ax.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize=8)
        fig.savefig(output / f"shap_highlight_tp6_{feature}.svg", bbox_inches="tight")
        plt.close(fig)
