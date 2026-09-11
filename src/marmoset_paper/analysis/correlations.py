"""Compound-mean Spearman correlations used by Figure 3."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from marmoset_paper.helpers.DataFunctions import normalize_regimen

PATHOLOGY = ["TP6_MeanHU", "delta_MeanHU", "TP6_TotalVol", "delta_TotalVol"]


def paired_correlations(invitro: pd.DataFrame, pathology: pd.DataFrame):
    """Correlate aligned compounds, recording the finite pair count for each test."""
    if not invitro.index.is_unique or not pathology.index.is_unique:
        raise ValueError("Correlation inputs require one row per regimen")
    shape = (len(invitro.columns), len(pathology.columns))
    rho, p = np.full(shape, np.nan), np.full(shape, np.nan)
    n = np.zeros(shape, dtype=int)
    for i, feature in enumerate(invitro):
        for j, outcome in enumerate(pathology):
            pair = pd.concat([invitro[feature], pathology[outcome]], axis=1, join="inner")
            pair = pair[np.isfinite(pair).all(axis=1)]
            n[i, j] = len(pair)
            if len(pair) >= 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                rho[i, j], p[i, j] = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return {
        name: pd.DataFrame(values, index=invitro.columns, columns=pathology.columns).rename_axis(
            "feature"
        )
        for name, values in [("rho_values", rho), ("p_values", p), ("n_pairs", n)]
    }


def run_correlations(data_dir: Path, output: Path) -> dict:
    invitro = pd.read_csv(data_dir / "in_vitro_modeling.csv")
    invitro = invitro[invitro.Drug.str.contains("+", regex=False)].copy()
    invitro["Drug"] = invitro.Drug.map(normalize_regimen)
    invitro = invitro.set_index("Drug")
    if not invitro.index.is_unique:
        raise ValueError("Duplicate in vitro regimen identifiers")
    lesions = pd.read_csv(data_dir / "marm_data_wide_clustered_classif.csv")
    lesions["Compound"] = lesions.Compound.map(normalize_regimen)
    invitro.to_csv(output / "in_vitro_combinations.csv")
    for tp in [2, 6]:
        expected = lesions[f"TP{tp}_SoftVol"] + lesions[f"TP{tp}_HardVol"]
        if not np.allclose(lesions[f"TP{tp}_TotalVol"], expected, equal_nan=True):
            raise ValueError(f"TP{tp}_TotalVol is not SoftVol + HardVol")
    for feature in ["MeanHU", "TotalVol"]:
        lesions[f"delta_{feature}"] = lesions[f"TP6_{feature}"] - lesions[f"TP2_{feature}"]
    counts = {}
    for name, label in [("less_severe", "cool"), ("severe", "hot")]:
        subset = lesions[lesions.classif == label]
        pathology = subset.groupby("Compound")[PATHOLOGY].mean()
        common = sorted(pathology.index.intersection(invitro.index))
        if len(common) < 3:
            raise ValueError(f"Fewer than three matched regimens for {name}")
        counts[name] = {
            "lesions": len(subset),
            "matched_regimens": common,
            "unmatched_regimens": sorted(set(pathology.index) - set(invitro.index)),
        }
        pathology.loc[common].to_csv(output / f"{name}_compound_means.csv")
        for metric, table in paired_correlations(
            invitro.loc[common], pathology.loc[common]
        ).items():
            table.to_csv(output / f"{name}_{metric}_mean.csv")
    return counts
