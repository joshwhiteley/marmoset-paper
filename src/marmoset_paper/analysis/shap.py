"""SHAP on stored training rows from the combined observed-history models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from marmoset_paper.analysis.modeling import IDENTIFIERS
from marmoset_paper.helpers.ShapFunctions import (
    calculate_shap_values,
    export_feature_data,
    select_output,
)


def load_combined_models(model_dir: Path) -> dict:
    """Validate bundle labels, fitted parameters, feature history, and shared split."""
    # Joblib inputs must come from a trusted source.
    bundles = {tp: joblib.load(model_dir / f"b_tp{tp}.joblib") for tp in range(3, 7)}
    reference = bundles[3]
    baseline = reference["features"]
    if len(baseline) < 2 or [name for name in baseline if name.startswith("TP")] != ["TP2_MeanHU"]:
        raise ValueError("The TP3 combined model must use baseline imaging only")
    for tp, bundle in bundles.items():
        if (
            bundle["schema_version"] != 1
            or bundle["target"] != f"TP{tp}_MeanHU"
            or bundle["strategy"] != "b"
        ):
            raise ValueError(f"Unexpected combined model bundle: b_tp{tp}")
        model = bundle["model"]
        if not isinstance(model, RandomForestRegressor):
            raise ValueError("Expected a fitted RandomForestRegressor")
        expected_features = baseline + [f"TP{previous}_MeanHU" for previous in range(3, tp)]
        if (
            bundle["features"] != expected_features
            or list(model.feature_names_in_) != expected_features
        ):
            raise ValueError(f"Inconsistent observed-history features for TP{tp}")
        config = bundle["config"]
        if (
            config != reference["config"]
            or model.n_estimators != config["n_estimators"]
            or model.max_depth != config["max_depth"]
            or model.random_state != config["seed"]
        ):
            raise ValueError(f"Inconsistent fitted model configuration for TP{tp}")
        for partition in ["train", "test"]:
            ids = bundle[partition][IDENTIFIERS].reset_index(drop=True)
            reference_ids = reference[partition][IDENTIFIERS].reset_index(drop=True)
            if ids.empty or ids.duplicated().any() or not ids.equals(reference_ids):
                raise ValueError(f"Inconsistent {partition} lesion identities for TP{tp}")
            shared = reference[partition].columns
            if (
                not bundle[partition][shared]
                .reset_index(drop=True)
                .equals(reference[partition].reset_index(drop=True))
            ):
                raise ValueError(f"Inconsistent shared {partition} measurements for TP{tp}")
        train_ids = pd.MultiIndex.from_frame(bundle["train"][IDENTIFIERS])
        test_ids = pd.MultiIndex.from_frame(bundle["test"][IDENTIFIERS])
        if not train_ids.intersection(test_ids).empty:
            raise ValueError(f"Train/test lesion overlap for TP{tp}")
    return bundles


def run_shap(model_dir: Path, output: Path, sample_size: int = 1000, seed: int = 42) -> dict:
    if sample_size < 1:
        raise ValueError("SHAP sample size must be positive")
    counts = {}
    for tp, bundle in load_combined_models(model_dir).items():
        stem = f"b_tp{tp}"
        train = bundle["train"]
        sample = train.sample(n=min(sample_size, len(train)), random_state=seed)
        features = sample[bundle["features"]]
        values, explainer = calculate_shap_values(bundle["model"], features)
        values = select_output(values)
        np.testing.assert_allclose(
            values.sum(axis=1) + np.asarray(explainer.expected_value).item(),
            bundle["model"].predict(features),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"SHAP does not reconstruct predictions for {stem}",
        )
        np.save(output / f"{stem}_values.npy", values)
        features.to_csv(output / f"{stem}_features.csv", index=False)
        sample[IDENTIFIERS].to_csv(output / f"{stem}_samples.csv", index=False)
        export_feature_data(
            values, features, sample[IDENTIFIERS], output / f"{stem}_feature_values.csv"
        )
        counts[stem] = {
            "samples": len(features),
            "features": len(features.columns),
            "expected_value": float(np.asarray(explainer.expected_value).item()),
        }
    return counts
