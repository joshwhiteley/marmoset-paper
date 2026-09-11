"""
Script for creating single-feature SHAP highlight plots.

This script generates detailed SHAP visualizations for individual features,
including both a combined colored view and compound-specific breakdowns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import warnings

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    "in_vitro_data": "data/in_vitro_diamond_data.csv",
    "shap_base_dir": "radiodensity-only/modeling/feature_analysis",
    "output_dir": "manuscript-figures/5/shap_highlights",
    "random_state": 42,
    "figure_width": 10,
    "figure_height": 2.5,
    "dpi": 300,
    "save_format": "svg",
}

# Compound styling
COMPOUND_COLORS = {
    'BDQ+DEL': '#4E79A7',
    'BDQ+LIN': '#F28E2B',
    'BDQ+PRE': '#E15759',
    'INH+PZA': '#76B7B2',
    'LIN+PRE': '#59A14F',
    'MOX+RIF': '#EDC948',
    'PZA+RIF': '#B07AA1',
    'BDQ+LIN+PRE': '#FF9DA7',
    'EMB+INH+PZA+RIF': '#9C755F',
    'EMB+MOX+PZA+RIF': '#BAB0AC',
}

COMPOUND_NAMES = {
    'BDQ+DEL': "BD",
    'BDQ+LIN': "BL",
    'BDQ+PRE': "BP",
    'INH+PZA': "HZ",
    'LIN+PRE': "PaL",
    'MOX+RIF': "MR",
    'PZA+RIF': "RZ",
    'BDQ+LIN+PRE': "BPaL",
    'EMB+INH+PZA+RIF': "HRZE",
    'EMB+MOX+PZA+RIF': "MRZE",
}

# Define Blue-Purple-Red colormap
BPR_COLORMAP = LinearSegmentedColormap.from_list(
    'blue_purple_red',
    [(0.0, '#1E3A8A'), (0.5, '#8C4199'), (1.0, '#991B1B')]
)


def load_shap_data_for_features(
    timepoint: str,
    output_name: str,
    feature_names: List[str],
    model_prefix: str = "b"
) -> pd.DataFrame:
    """
    Load SHAP values CSV and extract data for specific features.

    Returns a DataFrame with columns: sample_id, feature_name, feature_value, shap_value
    """

    # Load the full SHAP values CSV
    shap_csv = Path(CONFIG["shap_base_dir"]) / f"{model_prefix}_{timepoint}_shap_values_{output_name}.csv"

    if not shap_csv.exists():
        raise FileNotFoundError(f"Could not find SHAP CSV at {shap_csv}")

    print(f"  Loading SHAP values from: {shap_csv.name}")
    shap_df = pd.read_csv(shap_csv)

    # Also need the feature values from training data
    # We'll reconstruct this from the sequential regression setup
    from marmoset_paper.data.MarmosetData import MarmosetData
    from marmoset_paper.data.InVitroData import DiamondData
    from marmoset_paper.helpers.ModelingFunctions import impute_within_compound, split_by_compound
    import polars as pl

    # Load data
    marmoset_data = MarmosetData("data/marm_data_wide_clustered_classif.csv")
    diamond_data = DiamondData("data/in_vitro_diamond_data.csv")
    severe_lesions = marmoset_data.get_severe_lesions().data

    # Feature selection
    y_features = ["MeanHU"]
    metadata_features = ["Compound", "Lesion"]
    y_cols = [col for col in severe_lesions.columns if any(f in col for f in y_features)]
    severe_data = severe_lesions.select(y_cols + metadata_features)

    # Timepoint organization
    timepoints = {
        f"tp{tp}": [col for col in severe_data.columns if f"TP{tp}" in col and "MeanHU" in col]
        for tp in range(2, 7)
    }

    # In vitro features
    diamond_df = diamond_data.data.filter(
        pl.col('NumbDrugs') > 1
    ).filter(
        ~pl.col('Drug').str.contains('QBS')
    ).drop('NumbDrugs')
    combo_features = [col for col in diamond_df.columns if "Drug" not in col]

    # Merge
    X = severe_data.join(other=diamond_df, left_on="Compound", right_on="Drug", how="inner")

    # Define features based on timepoint
    base_input_features = timepoints["tp2"] + combo_features
    all_features = base_input_features.copy()

    # Add previous timepoints for the specified timepoint
    tp_num = int(timepoint.replace('tp', ''))
    for prev_tp_num in range(3, tp_num):
        all_features.extend(timepoints[f"tp{prev_tp_num}"])

    # Imputation
    X_imputed = impute_within_compound(X, columns=all_features, grouping_column='Compound')

    # Train/test split
    train, test = split_by_compound(
        X_imputed,
        grouping_column='Compound',
        test_size=0.2,
        random_state=42
    )

    # Sample training data (same as used for SHAP calculation)
    n_samples = min(1000, len(train))
    X_train_sample = (
        train.sample(n=n_samples, with_replacement=False, seed=42)
        .select(all_features)
        .to_pandas()
    )

    # Now create the long-form dataset for requested features
    result_rows = []
    for feat in feature_names:
        if feat not in X_train_sample.columns:
            print(f"  Warning: Feature {feat} not found in training data")
            continue

        if feat not in shap_df.columns:
            print(f"  Warning: Feature {feat} not found in SHAP values")
            continue

        for idx in range(len(X_train_sample)):
            result_rows.append({
                'sample_id': idx,
                'feature_name': feat,
                'feature_value': X_train_sample[feat].iloc[idx],
                'shap_value': shap_df[feat].iloc[idx]
            })

    return pd.DataFrame(result_rows)


def merge_with_invitro(shap_df: pd.DataFrame, in_vitro_path: str) -> pd.DataFrame:
    """Merge SHAP data with in vitro data to get compound labels."""

    # Load in vitro data
    df_invitro = pd.read_csv(in_vitro_path)

    # Filter out compounds containing OPD or QBS
    df_invitro = df_invitro[~df_invitro['Drug'].str.contains('OPD|QBS', na=False, regex=True)]

    # Melt in vitro data to long format
    melt_df = df_invitro.melt(
        id_vars=['Drug'],
        var_name='feature_name',
        value_name='feature_value'
    )

    # Merge on feature name and value
    merged = shap_df.merge(melt_df, on=['feature_name', 'feature_value'], how='left')

    # Drop rows without drug labels (these are compounds not in the in vitro dataset)
    merged = merged.dropna(subset=['Drug'])

    return merged


def add_jitter(df: pd.DataFrame, seed: int = 42, scale: float = 0.1) -> pd.DataFrame:
    """Add consistent jitter for y-axis positioning."""
    np.random.seed(seed)
    df['jitter'] = np.random.normal(loc=0.0, scale=scale, size=len(df))
    return df


def plot_shap_colored(
    df: pd.DataFrame,
    feature_name: str,
    output_path: Path,
    title: Optional[str] = None
) -> None:
    """Create SHAP plot colored by feature value."""

    plot_df = df[df['feature_name'] == feature_name].copy()

    if len(plot_df) == 0:
        print(f"  Warning: No data found for feature {feature_name}")
        return

    fig, ax = plt.subplots(figsize=(CONFIG["figure_width"], CONFIG["figure_height"]))

    # Normalize feature values around median
    vmin = plot_df['feature_value'].min()
    vmax = plot_df['feature_value'].max()
    vcenter = np.median(plot_df['feature_value'])

    # Handle constant or near-constant feature values
    range_val = vmax - vmin
    if range_val < 1e-10:  # Essentially constant
        # Use simple coloring for constant features
        sc = ax.scatter(
            plot_df['shap_value'],
            plot_df['jitter'],
            c='#8C4199',  # Use middle color from colormap
            alpha=0.85,
            s=45,
            edgecolors='none'
        )
        # Create a simple label instead of colorbar
        ax.text(0.98, 0.98, f'Feature value: {vmin:.4f}',
               transform=ax.transAxes,
               ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'),
               fontsize=9)
    else:
        # Ensure vcenter is strictly between vmin and vmax
        epsilon = range_val * 0.05  # 5% of range for safety
        if vcenter <= vmin:
            vcenter = vmin + range_val * 0.5
        elif vcenter >= vmax:
            vcenter = vmin + range_val * 0.5
        elif vcenter - vmin < epsilon:
            vcenter = vmin + range_val * 0.5
        elif vmax - vcenter < epsilon:
            vcenter = vmin + range_val * 0.5

        # Double check the values are valid
        if not (vmin < vcenter < vmax):
            # Fall back to middle
            vcenter = vmin + range_val * 0.5

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        # Scatter plot
        sc = ax.scatter(
            plot_df['shap_value'],
            plot_df['jitter'],
            c=plot_df['feature_value'],
            cmap=BPR_COLORMAP,
            norm=norm,
            alpha=0.85,
            s=45,
            edgecolors='none'
        )

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label('Feature Value', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=9)

    # Styling
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.axvline(0, color='#666666', linestyle='--', linewidth=1.5, zorder=1)

    # Clean spines
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')

    # Grid
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight', format=CONFIG["save_format"])
    plt.close()
    print(f"  ✓ Saved colored plot: {output_path.name}")


def plot_shap_by_compound(
    df: pd.DataFrame,
    feature_name: str,
    compound: str,
    output_path: Path
) -> None:
    """Create SHAP plot highlighting a specific compound."""

    plot_df = df[df['feature_name'] == feature_name].copy()

    if len(plot_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(CONFIG["figure_width"], CONFIG["figure_height"]))

    # Background: all points in light grey
    ax.scatter(
        plot_df['shap_value'],
        plot_df['jitter'],
        color='#D3D3D3',
        alpha=0.4,
        s=40,
        edgecolors='none'
    )

    # Foreground: highlight compound
    compound_data = plot_df[plot_df['Drug'] == compound]

    if len(compound_data) > 0:
        color = COMPOUND_COLORS.get(compound, '#333333')
        name = COMPOUND_NAMES.get(compound, compound)

        ax.scatter(
            compound_data['shap_value'],
            compound_data['jitter'],
            facecolors=color,
            edgecolors='black',
            s=50,
            linewidths=1.2,
            alpha=0.9,
            label=name,
            zorder=3
        )

        # Legend
        ax.legend(loc='upper right', frameon=True, edgecolor='#CCCCCC',
                 fontsize=10, framealpha=0.95)

    # Styling
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=11, fontweight='bold')
    ax.set_yticks([])
    ax.set_ylabel('')
    ax.axvline(0, color='#666666', linestyle='--', linewidth=1.5, zorder=1)

    # Clean spines
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')

    # Grid
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight', format=CONFIG["save_format"])
    plt.close()


def plot_shap_stacked(
    df: pd.DataFrame,
    feature_name: str,
    output_path: Path,
    title: Optional[str] = None
) -> None:
    """Create a stacked plot with SHAP distribution on top and compounds below."""

    plot_df = df[df['feature_name'] == feature_name].copy()

    if len(plot_df) == 0:
        print(f"  Warning: No data found for feature {feature_name}")
        return

    # Get unique compounds sorted
    compounds = sorted(plot_df['Drug'].unique())
    n_compounds = len(compounds)

    # Create figure with subplots: 1 for overall SHAP + n for compounds
    fig, axes = plt.subplots(
        nrows=n_compounds + 1,
        figsize=(CONFIG["figure_width"], 2.5 + n_compounds * 1.2),
        sharex=True,
        gridspec_kw={'height_ratios': [2.5] + [1.2] * n_compounds}
    )

    # Determine x-limits
    xmin = plot_df['shap_value'].min()
    xmax = plot_df['shap_value'].max()
    xrange = xmax - xmin
    xmin_plot = xmin - xrange * 0.05
    xmax_plot = xmax + xrange * 0.05

    # === Top plot: Colored by feature value ===
    ax_top = axes[0]

    # Normalize feature values
    vmin = plot_df['feature_value'].min()
    vmax = plot_df['feature_value'].max()
    vcenter = np.median(plot_df['feature_value'])
    range_val = vmax - vmin

    if range_val < 1e-10:  # Constant feature
        sc = ax_top.scatter(
            plot_df['shap_value'],
            plot_df['jitter'],
            c='#8C4199',
            alpha=0.85,
            s=45,
            edgecolors='none'
        )
        ax_top.text(0.98, 0.98, f'Feature value: {vmin:.4f}',
                   transform=ax_top.transAxes,
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#CCCCCC'),
                   fontsize=9)
    else:
        # Ensure vcenter is valid
        epsilon = range_val * 0.05
        if vcenter <= vmin or vcenter >= vmax or \
           vcenter - vmin < epsilon or vmax - vcenter < epsilon:
            vcenter = vmin + range_val * 0.5

        if not (vmin < vcenter < vmax):
            vcenter = vmin + range_val * 0.5

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        sc = ax_top.scatter(
            plot_df['shap_value'],
            plot_df['jitter'],
            c=plot_df['feature_value'],
            cmap=BPR_COLORMAP,
            norm=norm,
            alpha=0.85,
            s=45,
            edgecolors='none'
        )

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax_top, pad=0.02)
        cbar.set_label('Feature Value', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

    # Style top plot
    ax_top.set_ylabel('All\nCompounds', fontsize=9, fontweight='bold', rotation=0,
                     ha='right', va='center', labelpad=10)
    ax_top.set_yticks([])
    ax_top.axvline(0, color='#666666', linestyle='--', linewidth=1.5, zorder=1)
    ax_top.spines['left'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_color('#CCCCCC')
    ax_top.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5)
    ax_top.set_axisbelow(True)
    ax_top.set_xlim(xmin_plot, xmax_plot)

    if title:
        ax_top.set_title(title, fontsize=12, fontweight='bold', pad=10)

    # === Compound-specific plots ===
    for idx, (ax, compound) in enumerate(zip(axes[1:], compounds)):
        # Background: all points in grey
        ax.scatter(
            plot_df['shap_value'],
            plot_df['jitter'],
            color='#D3D3D3',
            alpha=0.3,
            s=30,
            edgecolors='none'
        )

        # Foreground: highlight compound
        compound_data = plot_df[plot_df['Drug'] == compound]
        color = COMPOUND_COLORS.get(compound, '#333333')
        name = COMPOUND_NAMES.get(compound, compound)

        ax.scatter(
            compound_data['shap_value'],
            compound_data['jitter'],
            facecolors=color,
            edgecolors='black',
            s=35,
            linewidths=0.8,
            alpha=0.9,
            zorder=3
        )

        # Style
        ax.set_ylabel(name, fontsize=9, fontweight='bold', rotation=0,
                     ha='right', va='center', labelpad=10)
        ax.set_yticks([])
        ax.axvline(0, color='#666666', linestyle='--', linewidth=1.0, zorder=1)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.grid(True, axis='x', alpha=0.15, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlim(xmin_plot, xmax_plot)

        # Only show x-axis for last subplot
        if idx < len(compounds) - 1:
            ax.spines['bottom'].set_visible(False)

    # X-axis label on bottom plot
    axes[-1].set_xlabel('SHAP Value (impact on model output)',
                        fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight',
               format=CONFIG["save_format"])
    plt.close()
    print(f"  ✓ Saved stacked plot: {output_path.name}")


def create_shap_highlights(
    features: List[str],
    timepoint: str,
    output_name: str = "MeanHU",
    model_prefix: str = "b"
) -> None:
    """
    Create SHAP highlight plots for a list of features.

    Parameters:
    -----------
    features : List[str]
        List of feature names to plot
    timepoint : str
        Timepoint (e.g., 'tp6')
    output_name : str
        Output variable name (e.g., 'TP6_MeanHU')
    model_prefix : str
        Model prefix ('b', 'm', or 'd')
    """

    print(f"\n{'='*70}")
    print(f"Creating SHAP Highlights for {timepoint.upper()} - {output_name}")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load SHAP data for all features at once
        print("Loading SHAP data and feature values...")
        shap_df = load_shap_data_for_features(timepoint, output_name, features, model_prefix)

        # Merge with in vitro data
        print("Merging with in vitro data...")
        merged = merge_with_invitro(shap_df, CONFIG["in_vitro_data"])

        # Add jitter
        merged = add_jitter(merged, seed=CONFIG["random_state"])

        print(f"\nTotal data points: {len(merged)}")
        print(f"Unique compounds: {merged['Drug'].nunique()}")
        print(f"Compounds: {sorted(merged['Drug'].unique())}\n")

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Process each feature
    for feature_name in features:
        print(f"\nProcessing feature: {feature_name}")
        print("-" * 70)

        try:
            feature_data = merged[merged['feature_name'] == feature_name]

            if len(feature_data) == 0:
                print(f"  ✗ No data found for feature {feature_name}")
                continue

            print(f"  Data points: {len(feature_data)}")
            print(f"  Unique compounds: {feature_data['Drug'].nunique()}")

            # Create stacked plot
            stacked_path = output_dir / f"{timepoint}_{output_name}_{feature_name}_stacked.{CONFIG['save_format']}"
            plot_shap_stacked(merged, feature_name, stacked_path, title=feature_name)

            print(f"  ✓ Completed {feature_name}")

        except Exception as e:
            print(f"  ✗ Error processing {feature_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    """Main function to generate SHAP highlight plots."""

    # Test cases for TP6
    features_to_plot = [
        "casGRinf_dormancySimplePK_Constant",
        "cellAUC25_cholesterolSimplePK_Termil",
        "NeutralHcellPK_AUC50",
        "NeutralNequip_FBC50",
    ]

    create_shap_highlights(
        features=features_to_plot,
        timepoint="tp6",
        output_name="TP6_MeanHU",
        model_prefix="b"
    )

    print(f"\n{'='*70}")
    print("✓ All SHAP highlight plots created successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
