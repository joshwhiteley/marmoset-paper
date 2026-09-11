"""Pairwise FxC50 comparisons between dosing strategies."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def correlation_groups() -> dict[str, list[str]]:
    groups = {
        f"{condition}_comparison": [
            f"FxC50_{condition}{strategy}" for strategy in ["equip", "casPK", "cellPK"]
        ]
        for condition in ["5N", "7H", "7N"]
    }
    for condition in ["butyrate", "cholesterol", "dormancy"]:
        for phase in ["Constant", "Terminal"]:
            groups[f"{condition}_{phase.lower()}"] = [
                f"FxC50_{condition}{strategy}_{phase}" for strategy in ["", "CasPK", "CellPK"]
            ]
    return groups


def run_fxc(data_dir: Path, output: Path) -> dict:
    data = pd.read_csv(data_dir / "in_vitro_modeling.csv")
    rows = []
    for group, features in correlation_groups().items():
        for first, second in combinations(features, 2):
            pair = data[[first, second]]
            pair = pair[np.isfinite(pair).all(axis=1)]
            rho, p = np.nan, np.nan
            if len(pair) >= 3 and (pair.nunique() > 1).all():
                rho, p = spearmanr(pair[first], pair[second])
            rows.append(
                {
                    "group": group,
                    "feature_1": first,
                    "feature_2": second,
                    "rho": rho,
                    "p_value": p,
                    "n": len(pair),
                }
            )
    pd.DataFrame(rows).to_csv(output / "in_vitro_correlations_summary.csv", index=False)
    return {"input_rows": len(data), "comparisons": len(rows)}
