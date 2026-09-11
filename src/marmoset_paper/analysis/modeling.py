"""Observed-history random forests for TP3–TP6 MeanHU."""

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from marmoset_paper.helpers.FeatureAnalysisFunctions import (
    get_permutation_importance,
    get_rf_feature_importance,
)
from marmoset_paper.helpers.ModelingFunctions import impute_within_compound, split_by_compound

IDENTIFIERS = ["Compound", "MarmID", "Lesion"]
TARGETS = [f"TP{tp}_MeanHU" for tp in range(2, 7)]


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 10


def prepare_model_data(data_dir: Path) -> tuple[pl.DataFrame, list[str], dict]:
    lesions = pl.read_csv(
        data_dir / "marm_data_wide_clustered_classif.csv", infer_schema_length=None
    )
    severe = (
        lesions.filter(pl.col("classif") == "hot")
        .sort(["Compound", "Lesion"], maintain_order=True)
        .select(IDENTIFIERS + TARGETS)
    )
    invitro = (
        pl.read_csv(data_dir / "in_vitro_diamond_data.csv")
        .filter((pl.col("NumbDrugs") > 1) & ~pl.col("Drug").str.contains("QBS"))
        .drop("NumbDrugs")
    )
    if severe.select(IDENTIFIERS).is_duplicated().any():
        raise ValueError("Duplicate lesion identifiers in modeling data")
    features = [c for c in invitro.columns if c != "Drug"]
    merged = severe.join(
        invitro,
        left_on="Compound",
        right_on="Drug",
        how="inner",
        validate="m:1",
        maintain_order="left",
    )
    counts = {
        "all_lesions": len(lesions),
        "severe_lesions": len(severe),
        "modeled_lesions": len(merged),
        "modeled_regimens": sorted(merged["Compound"].unique().to_list()),
        "excluded_regimens": sorted(set(severe["Compound"]) - set(merged["Compound"])),
    }
    if merged.is_empty() or not np.isfinite(merged.select(TARGETS).to_numpy()).all():
        raise ValueError("Modeling requires finite radiodensity measurements at TP2–TP6")
    return merged, features, counts


def train_models(data_dir: Path, output: Path, config: ModelConfig) -> dict:
    """Fit both strategies on the same rows; retain observed prior HU as inputs.

    Compound-median imputation before splitting is retained from the source code.
    This command does not implement recursive baseline-only forecasts.
    """
    data, invitro_features, counts = prepare_model_data(data_dir)
    metrics, predictions, splits = [], [], []
    for strategy, baseline in {
        "m": ["TP2_MeanHU"],
        "b": ["TP2_MeanHU"] + invitro_features,
    }.items():
        imputed = impute_within_compound(data, baseline)
        train_pl, test_pl = split_by_compound(
            imputed, test_size=config.test_size, random_state=config.seed
        )
        train, test = train_pl.to_pandas(), test_pl.to_pandas()
        if train.empty or len(test) < 2:
            raise ValueError("Not enough rows to train and evaluate the models")
        counts[strategy] = {"train": len(train), "test": len(test)}
        for partition, frame in [("train", train), ("test", test)]:
            splits.append(frame[IDENTIFIERS].assign(strategy=strategy, partition=partition))
        for tp in range(3, 7):
            features = baseline + [f"TP{previous}_MeanHU" for previous in range(3, tp)]
            target = f"TP{tp}_MeanHU"
            model = RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.seed,
                n_jobs=1,
            )
            model.fit(train[features], train[target])
            predicted = model.predict(test[features])
            metrics.append(
                {
                    "strategy": strategy,
                    "timepoint": tp,
                    "output": target,
                    "mse": mean_squared_error(test[target], predicted),
                    "mae": mean_absolute_error(test[target], predicted),
                    "r2": r2_score(test[target], predicted),
                }
            )
            predictions.append(
                test[IDENTIFIERS].assign(
                    strategy=strategy,
                    timepoint=tp,
                    actual=test[target],
                    predicted=predicted,
                    baseline=test["TP2_MeanHU"],
                )
            )
            stem = f"{strategy}_tp{tp}"
            # The bundle stores the actual rows used for SHAP and evaluation. Never
            # reconstruct a historical split by globbing a model and rerunning RNG.
            joblib.dump(
                {
                    "schema_version": 1,
                    "model": model,
                    "features": features,
                    "target": target,
                    "strategy": strategy,
                    "config": asdict(config),
                    "train": train[IDENTIFIERS + features + [target]],
                    "test": test[IDENTIFIERS + features + [target]],
                },
                output / f"{stem}.joblib",
            )
            get_rf_feature_importance(model, features).to_csv(
                output / f"{stem}_mdi.csv", index=False
            )
            get_permutation_importance(
                model, test[features], test[target], features, random_state=config.seed
            ).to_csv(output / f"{stem}_permutation.csv", index=False)
    pd.DataFrame(metrics).to_csv(output / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.concat(splits, ignore_index=True).to_csv(output / "split_assignments.csv", index=False)
    return counts
