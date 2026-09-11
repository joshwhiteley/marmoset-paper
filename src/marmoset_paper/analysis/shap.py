"""SHAP on stored training rows from the combined observed-history models."""

from pathlib import Path

import joblib
import numpy as np

from marmoset_paper.analysis.modeling import IDENTIFIERS
from marmoset_paper.helpers.ShapFunctions import (
    calculate_shap_values,
    export_feature_data,
    select_output,
)


def run_shap(model_dir: Path, output: Path, sample_size: int = 1000, seed: int = 42) -> dict:
    if sample_size < 1:
        raise ValueError("SHAP sample size must be positive")
    counts = {}
    for tp in range(3, 7):
        stem = f"b_tp{tp}"
        # Only load bundles produced by this repository, or from a trusted source.
        bundle = joblib.load(model_dir / f"{stem}.joblib")
        if bundle["schema_version"] != 1 or bundle["target"] != f"TP{tp}_MeanHU":
            raise ValueError(f"Unexpected model bundle: {stem}")
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
