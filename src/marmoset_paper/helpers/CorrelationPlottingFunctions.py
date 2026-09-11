"""Figure 3 feature ordering, phase selection, and correlation heatmaps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

METRIC_COLORS = {"AUC": "#FF8C00", "Einf": "#2E8B57", "FxC": "#FF1493", "GRmax": "#8A2BE2"}
CATEGORY_ORDER = ["simple equipotent", "simple PK", "LIDS"]


def metric_type(feature: str) -> str:
    name = feature.upper()
    for metric, patterns in {
        "AUC": ["AUC"],
        "Einf": ["EINF", "E_INF"],
        "FxC": ["FIC", "FBC", "F50", "F90", "FXC50", "FXC90"],
        "GRmax": ["GRMAX", "GR_MAX", "GR MAX"],
    }.items():
        if any(pattern in name for pattern in patterns):
            return metric
    return "other"


def feature_category(feature: str) -> str:
    name = feature.lower()
    if any(marker in name for marker in ["5n", "7n", "7h"]):
        return "LIDS"
    if "cellpk" in name or "caspk" in name:
        return "simple PK"
    return "simple equipotent"


def select_phases(rho: pd.DataFrame, p: pd.DataFrame, alpha: float = 0.05) -> list[str]:
    """Keep the phase with largest mean |rho| among nominally significant outcomes.

    This is a display selection from the original Figure 3 script, not an adjusted
    hypothesis test. If neither phase is significant, retain the first input phase.
    """
    if not rho.index.equals(p.index) or not rho.columns.equals(p.columns):
        raise ValueError("Rho and p-value tables must have identical feature and outcome labels")
    if not rho.index.is_unique:
        raise ValueError("Duplicate correlation features")
    groups = {}
    for feature in rho.index:
        base = feature
        for suffix in [
            "_Constant",
            "_Terminal",
            "_C",
            "_T",
            "(constant)",
            "(terminal)",
            "C",
            "T",
            "CT",
        ]:
            if base.endswith(suffix):
                base = base[: -len(suffix)].strip()
                break
        groups.setdefault(base, []).append(feature)
    selected = []
    for features in groups.values():
        scores = rho.loc[features].abs().where(p.loc[features] < alpha).mean(axis=1)
        selected.append(scores.idxmax() if scores.notna().any() else features[0])
    return sorted(selected, key=lambda f: (CATEGORY_ORDER.index(feature_category(f)), f))


def plot_correlation_heatmap(rho: pd.DataFrame, p: pd.DataFrame, output: Path, title: str):
    """Plot selected phases; flip Einf signs for display only, as in the source figure."""
    selected = select_phases(rho, p)
    display = rho.loc[selected].copy()
    for feature in selected:
        if metric_type(feature) == "Einf":
            display.loc[feature] *= -1
    display = display.where(p.loc[selected] < 0.05)
    cmap = LinearSegmentedColormap.from_list(
        "correlation", ["#d73027", "#fee0d2", "white", "#deebf7", "#4575b4"]
    )
    cmap.set_bad("#D3D3D3")
    fig, ax = plt.subplots(figsize=(8, 20))
    image = ax.imshow(display.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(
        range(len(display.columns)),
        display.columns.str.replace("TP6_", "").str.replace("delta_", "Δ "),
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(selected)), selected)
    for tick, feature in zip(ax.get_yticklabels(), selected):
        tick.set_color(METRIC_COLORS.get(metric_type(feature), "black"))
        tick.set_weight("bold")
    categories = [feature_category(feature) for feature in selected]
    for category in CATEGORY_ORDER:
        positions = [i for i, value in enumerate(categories) if value == category]
        if positions:
            if positions[0]:
                ax.axhline(positions[0] - 0.5, color="black", linewidth=2)
            ax.text(
                len(display.columns) - 0.4,
                np.mean(positions),
                category,
                va="center",
                rotation=90,
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, shrink=0.8, pad=0.12, label="Displayed ρ (Einf sign reversed)")
    ax.set(title=title, xlabel="Pathology measure", ylabel="In vitro feature")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    # Record exactly what was displayed, including the sign convention and phase.
    display.rename_axis("feature").to_csv(output.with_suffix(".csv"))
    return selected


def plot_correlation_legend(output: Path):
    fig, ax = plt.subplots(figsize=(10, 2))
    handles = [Patch(color=color, label=metric) for metric, color in METRIC_COLORS.items()]
    handles.append(Patch(color="#D3D3D3", label="p ≥ 0.05 or unavailable"))
    ax.legend(handles=handles, loc="center", ncol=5)
    ax.axis("off")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
