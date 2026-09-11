"""Exploratory standardized lesion changes; not a manuscript figure entry point."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from marmoset_paper.cli import ROOT
from marmoset_paper.provenance import recorded_run

FEATURES = ["MeanHU", "StandDevHU", "SoftVol", "HardVol", "MeanSUV"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "data/marm_data_wide_clustered_classif.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lesion-changes")
    args = parser.parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    with recorded_run(args.output_dir, [args.input], {"scaling": "global StandardScaler"}, ROOT):
        data = pd.read_csv(args.input)
        delta = pd.DataFrame({name: data[f"TP6_{name}"] - data[f"TP2_{name}"] for name in FEATURES})
        if not np.isfinite(delta).all().all():
            raise ValueError(
                "Lesion changes contain missing or infinite values; review them before scaling"
            )
        scaled = pd.DataFrame(StandardScaler().fit_transform(delta), columns=FEATURES)
        identified = pd.concat(
            [data[["Compound", "MarmID", "Lesion", "cluster", "classif"]], scaled], axis=1
        )
        identified.to_csv(args.output_dir / "scaled_changes.csv", index=False)
        for group in ["cluster", "Compound"]:
            medians = identified.groupby(group)[FEATURES].median()
            medians.to_csv(args.output_dir / f"{group}_medians.csv")
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(medians.T, center=0, cmap="RdBu_r", ax=ax)
            ax.set(title=f"Median standardized changes by {group}", ylabel="TP6 − TP2")
            fig.savefig(args.output_dir / f"{group}_medians.svg", bbox_inches="tight")
            plt.close(fig)
        grouped = identified.groupby(["Compound", "classif"])[FEATURES].median()
        grouped.to_csv(args.output_dir / "severity_regimen_medians.csv")
        limit = np.abs(grouped.to_numpy()).max()
        compounds = sorted(data.Compound.unique(), key=lambda value: (-value.count("+"), value))
        fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        for ax, label in zip(axes, ["cool", "hot"]):
            matrix = grouped.xs(label, level="classif").reindex(compounds).T
            sns.heatmap(matrix, center=0, vmin=-limit, vmax=limit, cmap="RdBu_r", ax=ax)
            ax.set(title=label, ylabel="TP6 − TP2")
        fig.savefig(args.output_dir / "severity_regimen_medians.svg", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
