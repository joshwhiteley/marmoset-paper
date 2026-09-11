"""Shared plotting functions. Callers own input selection and output paths."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from marmoset_paper.helpers.constants import MARMOSET_IN_VITRO_TUPLE, MARMOSET_ONLY_TUPLE

STRATEGIES = {
    "m": ("marmoset only", MARMOSET_ONLY_TUPLE),
    "b": ("marmoset + in vitro", MARMOSET_IN_VITRO_TUPLE),
}


def plot_performance_metrics(metrics: pd.DataFrame, axes):
    """Plot one score per strategy and timepoint; reject missing or duplicate scores."""
    expected = pd.MultiIndex.from_product([["m", "b"], range(3, 7)])
    indexed = metrics.set_index(["strategy", "timepoint"])
    if not indexed.index.is_unique or set(indexed.index) != set(expected):
        raise ValueError("Expected exactly one metric row per strategy and TP3–TP6")
    if not np.isfinite(indexed[["mse", "mae", "r2"]]).all().all():
        raise ValueError("Performance metrics must be finite")
    for ax, metric, title, unit in zip(
        axes, ["mse", "r2"], ["mean squared error", "coefficient of determination"], ["HU²", "R²"]
    ):
        for index, (strategy, (label, color)) in enumerate(STRATEGIES.items()):
            values = indexed.loc[strategy].sort_index()[metric]
            ax.bar(
                np.arange(4) + (index - 0.5) * 0.35, values, width=0.35, label=label, color=color
            )
        ax.set_xticks(range(4), [2, 4, 6, 8])
        ax.set(title=title, ylabel=unit, xlabel="weeks from treatment start")
        ax.spines[["right", "top"]].set_visible(False)


def plot_shap_summary(values: np.ndarray, features: pd.DataFrame, output: Path, max_display: int):
    """Draw a beeswarm without changing the selected feature set or SHAP values."""
    if values.shape != features.shape or features.empty:
        raise ValueError("SHAP plot requires aligned, nonempty samples and features")
    shap.summary_plot(values, features, max_display=max_display, show=False, rng=42)
    fig = plt.gcf()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
