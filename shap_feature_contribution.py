"""shap_feature_contribution.py
-------------------------------------------------
Command-line utility to deep-dive into SHAP feature contributions
for a selected time-point model.

Usage (run via uv):
    uv run shap_feature_contribution.py \
        --tp tp6 \
        --output TP6_MeanSUV \
        --features AcidicN,AcidicH \
        --plot box \
        --heatmap            # optional

Key functions
-------------
1. Computes SHAP values for the chosen model/time-point on a
   sub-sample of training data (if not already cached).
2. Extracts contributions of user-specified feature(s)
   (exact names or substring patterns).
3. Produces:
   • Box / violin / swarm plots of contribution distributions
   • Optional heat-map of median contribution per compound × feature
4. Saves tidy per-sample SHAP dataframe and summary statistics CSVs.

All heavy data-wrangling is done with polars. Plots use seaborn.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import joblib
import polars as pl
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Import shap only when needed for SHAP calculations
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. SHAP calculations will be skipped.")

from data.MarmosetData import MarmosetData
from data.InVitroData import DiamondData
from helpers.ModelingFunctions import impute_within_compound, split_by_compound
from helpers.FeatureAnalysisFunctions import calculate_shap_values

# -----------------------------------------------------------------------------
# CONSTANTS & DEFAULTS
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")
MODEL_DIR = Path("modeling/models")
DEFAULT_SAVE_DIR = Path("shap_feature_outputs")
SHAP_SAMPLE_SIZE = 1_000  # subsample for SHAP calc
RND_STATE = 42

# Define consistent colors for each compound
# Sorted by: 1. Number of '+' (ascending), 2. Alphabetical
COMPOUND_COLORS = {
    # 2-drug combinations
    'BDQ+DEL': '#4E79A7',  # blue
    'BDQ+LIN': '#F28E2B',  # orange
    'BDQ+PRE': '#E15759',  # red
    'INH+PZA': '#76B7B2',  # teal
    'LIN+PRE': '#59A14F',  # green
    'MOX+RIF': '#EDC948',  # yellow
    'PZA+RIF': '#B07AA1',  # purple
    # 3-drug combinations
    'BDQ+LIN+PRE': '#FF9DA7',  # light red
    # 4-drug combinations
    'EMB+INH+PZA+RIF': '#9C755F',  # brown
    'EMB+MOX+PZA+RIF': '#BAB0AC',  # gray
}

# Function to get color for a compound, with fallback for unknown compounds
def get_compound_color(compound: str) -> str:
    """Return consistent color for a compound, with fallback for unknown compounds."""
    return COMPOUND_COLORS.get(compound, '#D3D3D3')  # light gray for unknown

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def _discover_model(tp_key: str) -> Path:
    """Return first model file matching *_{tp_key}_*.joblib."""
    files = sorted(MODEL_DIR.glob(f"*_{tp_key}_*.joblib"))
    if not files:
        raise FileNotFoundError(f"No model file found in {MODEL_DIR} for {tp_key}.")
    return files[0]


def _build_feature_lists(df_sev: pl.DataFrame, diamond_cols: list[str]) -> dict[str, list[str]]:
    """Return mapping from tp key to lesion feature columns."""
    mapping: dict[str, list[str]] = {}
    for tp in range(2, 7):
        mapping[f"tp{tp}"] = [c for c in df_sev.columns if f"TP{tp}" in c]
    # Add base list (tp2 & in-vitro) – callers can extend.
    mapping["base"] = mapping["tp2"] + diamond_cols
    return mapping


def _parse_feature_patterns(all_feats: Sequence[str], patterns: Sequence[str]) -> dict[str, list[int]]:
    """Map each pattern to list of feature indices in *all_feats*."""
    idx_dict: dict[str, list[int]] = {}
    for pattern in patterns:
        idx = [i for i, feat in enumerate(all_feats) if (feat == pattern) or (pattern in feat)]
        if not idx:
            print(f"[warn] pattern '{pattern}' not found in feature list – skipped.")
            continue
        idx_dict[pattern] = idx
    if not idx_dict:
        raise ValueError(
            f"None of the provided patterns matched any feature names. "
            f"Available features: {all_feats[:10]}... (truncated, {len(all_feats)} total)"
        )
    return idx_dict


def _plot_distribution(df: pl.DataFrame, feature_label: str, plot_type: str, save_path: Path) -> None:
    """Create box/violin/swarm plot using seaborn and save PNG.
    
    Uses consistent colors for each compound across all plots.
    Sorts compounds by descending median SHAP value.
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate median SHAP values for each compound
    median_shap = (
        df.group_by("Compound")
        .agg(pl.col("SHAP").median().alias("median_shap"))
        .sort("median_shap", descending=False)
    )
    
    # Get compounds ordered by median SHAP (descending)
    compounds = median_shap["Compound"].to_list()
    
    # Create a color palette using our consistent colors
    palette = [get_compound_color(c) for c in compounds]
    
    if plot_type == "box":
        sns.boxplot(
            data=df.to_pandas(), 
            x="Compound", 
            y="SHAP", 
            order=compounds,  # Order by descending median SHAP
            palette=palette,
            boxprops=dict(alpha=0.7)
        )
    elif plot_type == "violin":
        sns.violinplot(
            data=df.to_pandas(), 
            x="Compound", 
            y="SHAP", 
            order=compounds,
            palette=palette,
            cut=0,
            inner="quartile"
        )
    elif plot_type == "swarm":
        sns.swarmplot(
            data=df.to_pandas(), 
            x="Compound", 
            y="SHAP", 
            order=compounds,
            palette=palette,
            size=4,
            alpha=0.7
        )
    else:
        raise ValueError("plot_type must be one of: box, violin, swarm")
    
    # Add zero line and format
    plt.axhline(0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{feature_label} SHAP by Compound")
    plt.ylabel("SHAP Value")
    plt.xlabel("")
    
    # Add legend for compound colors
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=get_compound_color(c), 
                  markersize=10, label=c)
        for c in compounds
    ]
    plt.legend(
        handles=handles, 
        title="Compounds",
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_heatmap(med_df: pl.DataFrame, save_path: Path) -> None:
    """Plot heat-map of median SHAP per compound × feature.
    
    Rows (compounds) are sorted by descending median SHAP value.
    """
    # Pivot with updated API and ensure numeric data
    pivot_df = med_df.to_pandas().pivot(
        index="Compound", 
        columns="Feature", 
        values="median"
    )
    
    # Sort compounds by ascending median SHAP value
    compounds = pivot_df.mean(axis=1).sort_values(ascending=True).index
    pivot_df = pivot_df.loc[compounds]
    
    # Set up the figure with appropriate size
    plt.figure(figsize=(
        max(10, 0.6 * len(pivot_df.columns)),  # Width
        max(8, 0.5 * len(pivot_df.index))     # Height
    ))
    
    # Create a custom colormap that matches our compound colors
    # Using a diverging colormap for SHAP values (red/blue)
    cmap = sns.diverging_palette(220, 20, as_cmap=True, sep=10)
    
    # Create the heatmap
    ax = sns.heatmap(
        pivot_df,
        cmap=cmap,
        center=0,
        cbar_kws={
            'label': 'Median SHAP',
            'shrink': 0.8,
            'aspect': 10
        },
        square=False,  # Make cells rectangular
        linewidths=0.5,  # Add grid lines
        linecolor='white',
        xticklabels=True,
        yticklabels=True,
        vmin=-2,  # Set fixed scale for better comparison
        vmax=2,
    )
    
    # Add colored y-tick labels based on compound colors
    # This is a workaround since row_colors isn't supported
    for i, label in enumerate(ax.get_yticklabels()):
        compound = label.get_text()
        if compound in COMPOUND_COLORS:
            label.set_color(COMPOUND_COLORS[compound])
    
    # Add color legend for compounds
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=compound)
        for compound, color in COMPOUND_COLORS.items()
        if compound in compounds
    ]
    
    # Improve the appearance
    plt.title(
        "Median SHAP Contribution per Compound × Feature\n"
        "(Red: Positive, Blue: Negative)",
        pad=20,
        fontsize=12
    )
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Add legend for compound colors
    if legend_elements:  # Only add legend if we have elements to show
        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            borderaxespad=0.,
            title="Compounds",
            frameon=False
        )
    
    plt.tight_layout()
    
    # Save with higher resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------------------------------

import datetime as dt

def run(tp_key: str, output_name: str, feature_patterns: list[str], plot_type: str, save_dir: Path, heatmap: bool) -> None:
    now = dt.datetime.now().strftime("%Y%m%d")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & prep data ------------------------------------------------------
    marm = MarmosetData(str(DATA_DIR / "marm_data_wide_clustered_classif.csv"))
    diamond = DiamondData(str(DATA_DIR / "in_vitro_diamond_data.csv"))

    sev_df = marm.get_severe_lesions().data  # polars
    lesion_feats = [c for c in sev_df.columns if any(t in c for t in ("MeanHU", "MeanSUV"))]
    meta_cols = ["Compound", "Lesion"]

    # keep only lesion + meta columns
    sev_df = sev_df.select(lesion_feats + meta_cols)

    # prepare diamond / in-vitro features (remove single-drug rows)
    diamond_df = (
        diamond.data.filter(pl.col("NumbDrugs") > 1).drop("NumbDrugs")
    )
    invitro_cols = [c for c in diamond_df.columns if "Drug" not in c]

    # merge on compound
    merged = sev_df.join(diamond_df, left_on="Compound", right_on="Drug", how="inner")

    # impute within compound, then train-test split (we only need train)
    base_cols = _build_feature_lists(sev_df, invitro_cols)["base"]
    merged_imp = impute_within_compound(merged, base_cols, grouping_column="Compound")
    train_pl, _ = split_by_compound(merged_imp, grouping_column="Compound", test_size=0.2, random_state=RND_STATE)
    train_df = train_pl.to_pandas().reset_index(drop=True)

    # construct feature list for SHAP – include previous time-points
    tp_num = int(tp_key.replace("tp", ""))
    feature_map = _build_feature_lists(sev_df, invitro_cols)
    feature_list: list[str] = feature_map["base"].copy()
    for prev in range(3, tp_num):
        feature_list += feature_map[f"tp{prev}"]

    # 2. Load model -----------------------------------------------------------
    model_path = _discover_model(tp_key)
    model = joblib.load(model_path)

    # 3. SHAP calculation -----------------------------------------------------
    sample_df = train_df.sample(n=min(SHAP_SAMPLE_SIZE, len(train_df)), random_state=RND_STATE)
    X_shap = sample_df[feature_list]

    print("Calculating SHAP values …")
    shap_vals, _ = calculate_shap_values(model, X_shap, approximate=False)
    if shap_vals is None:
        raise RuntimeError("SHAP calculation failed.")

    # shap_vals shape: (n_samp, n_feat, n_outputs)
    outs = feature_map[tp_key]
    if output_name not in outs:
        raise ValueError(f"output_name '{output_name}' not found among {outs}")
    out_idx = outs.index(output_name)
    shap_2d = shap_vals[:, :, out_idx]  # (n_samp, n_feat)

    # 4. Extract contributions -----------------------------------------------
    idx_dict = _parse_feature_patterns(feature_list, feature_patterns)

    # accumulate median table if heatmap requested
    med_rows: list[pl.DataFrame] = []

    for pat, idxs in idx_dict.items():
        contrib = shap_2d[:, idxs].sum(axis=1)  # (n_samp,)
        df_pat = pl.DataFrame({
            "Compound": sample_df["Compound"].values,
            "SHAP": contrib,
        })

        # save per-sample full list
        csv_file = save_dir / f"{output_name}/shap_{pat}_{tp_key}_{output_name}_{now}.csv"
        df_pat.write_csv(csv_file)

        # summary stats
        summary = (
            df_pat.group_by("Compound")
            .agg([
                pl.col("SHAP").median().alias("median"),
                pl.col("SHAP").mean().alias("mean"),
                pl.col("SHAP").std().alias("std"),
                pl.len().alias("n")
            ])
            .sort("median")
        )
        summary_file = save_dir / f"{output_name}/summary_{pat}_{tp_key}_{output_name}_{now}.csv"
        summary.write_csv(summary_file)

        # plot distribution
        img_file = save_dir / f"{output_name}/{pat}_{plot_type}_{tp_key}_{output_name}_{now}.svg"
        _plot_distribution(df_pat, pat, plot_type, img_file)

        # collect for heatmap
        if heatmap:
            med_rows.append(summary.select(["Compound", pl.lit(pat).alias("Feature"), "median"]))

    # 5. Heat-map across patterns --------------------------------------------
    if heatmap and med_rows:
        med_df = pl.concat(med_rows)
        heat_file = save_dir / f"heatmap_{tp_key}_{output_name}_{now}.png"
        _plot_heatmap(med_df, heat_file)

    print(f"✔︎ Outputs saved to → {save_dir.absolute()}")

# -----------------------------------------------------------------------------
# ENTRY-POINT
# -----------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyse SHAP feature contributions by compound.")
    p.add_argument("--tp", required=True, help="Time-point key (e.g. tp6)")
    p.add_argument("--output", required=True, help="Output name (e.g. TP6_MeanSUV)")
    p.add_argument("--features", required=True, help="Comma-separated list of feature patterns")
    p.add_argument("--plot", default="box", choices=["box", "violin", "swarm"], help="Plot type")
    p.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR), help="Directory to save figures/CSVs")
    p.add_argument("--heatmap", action="store_true", help="Also create heat-map across patterns")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    feature_patterns = [f.strip() for f in args.features.split(",") if f.strip()]
    run(
        tp_key=args.tp.lower(),
        output_name=args.output,
        feature_patterns=feature_patterns,
        plot_type=args.plot,
        save_dir=Path(args.save_dir),
        heatmap=args.heatmap,
    )


if __name__ == "__main__":
    main()
