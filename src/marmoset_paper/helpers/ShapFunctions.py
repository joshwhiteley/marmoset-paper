"""Tree SHAP calculation and explicit output selection."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap


def calculate_shap_values(model, X_data: pd.DataFrame, approximate: bool = False):
    """Explain feature rows in training column order; propagate calculation failures."""
    if list(model.feature_names_in_) != list(X_data.columns):
        raise ValueError("SHAP features must match the model's training column order")
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X_data, approximate=approximate, check_additivity=True)
    return values, explainer


def select_output(values: np.ndarray, output_index: int = 0) -> np.ndarray:
    """Select a regression output or class from SHAP >=0.45 arrays.

    Single-output regression is two-dimensional. Multi-output regression and
    sklearn RF classification use (samples, features, outputs/classes).
    """
    values = np.asarray(values)
    if values.ndim == 2 and output_index == 0:
        return values
    if values.ndim == 3 and 0 <= output_index < values.shape[2]:
        return values[:, :, output_index]
    raise ValueError(f"Cannot select output {output_index} from SHAP shape {values.shape}")


def export_feature_data(
    values: np.ndarray, features: pd.DataFrame, metadata: pd.DataFrame, destination: Path
) -> None:
    """Export SHAP and feature values joined by row identity, never by measured value."""
    if values.shape != features.shape or not features.index.equals(metadata.index):
        raise ValueError("SHAP values, features, and sample identifiers are not aligned")
    rows = []
    for index, name in enumerate(features.columns):
        frame = metadata.copy()
        frame["feature"] = name
        frame["feature_value"] = features[name]
        frame["shap_value"] = values[:, index]
        rows.append(frame)
    pd.concat(rows, ignore_index=True).to_csv(destination, index=False)
