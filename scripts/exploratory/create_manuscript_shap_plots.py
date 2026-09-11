"""
Standalone script for creating publication-quality SHAP plots from saved models.

This script loads pre-trained models and generates both regular and zoomed SHAP plots
for manuscript figure 5.
"""

import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import necessary functions
from marmoset_paper.data.MarmosetData import MarmosetData
from marmoset_paper.data.InVitroData import DiamondData
from marmoset_paper.helpers.ModelingFunctions import impute_within_compound, split_by_compound
from marmoset_paper.helpers.FeatureAnalysisFunctions import categorize_features
from marmoset_paper.helpers.constants import FEATURE_CATEGORY_COLORS

# Configuration
CONFIG = {
    "model_path": "radiodensity-only/modeling/models/model_tp6_b_seed42_20250930.joblib",
    "shap_csv_path": "radiodensity-only/modeling/feature_analysis/b_tp6_shap_values_TP6_MeanHU.csv",
    "output_dir": "manuscript-figures/5",
    "tp_key": "tp6",
    "output_name": "TP6_MeanHU",
    "random_state": 42,
    "shap_sample_size": 1000,
    "n_top_features_to_drop": 8,  # for zoomed plot
    "figsize": (10, 12),
    "dpi": 300,
}


def load_and_prepare_data():
    """Load and prepare the training data matching the sequential_regression setup."""

    # Load data
    marmoset_data = MarmosetData("data/marm_data_wide_clustered_classif.csv")
    diamond_data = DiamondData("data/in_vitro_diamond_data.csv")

    # Get severe lesions
    severe_lesions = marmoset_data.get_severe_lesions().data

    # Feature selection
    y_features = ["MeanHU"]
    metadata_features = ["Compound", "Lesion"]

    # Get columns
    y_cols = [col for col in severe_lesions.columns if any(f in col for f in y_features)]
    severe_data = severe_lesions.select(y_cols + metadata_features)

    # Timepoint organization
    timepoints = {
        f"tp{tp}": [col for col in severe_data.columns if f"TP{tp}" in col and "MeanHU" in col]
        for tp in range(2, 7)
    }

    # In vitro feature preparation
    diamond_df = diamond_data.data.filter(
        pl.col('NumbDrugs') > 1
    ).filter(
        ~pl.col('Drug').str.contains('QBS')
    ).drop('NumbDrugs')
    combo_features = [col for col in diamond_df.columns if "Drug" not in col]

    # Merge
    X = severe_data.join(
        other=diamond_df,
        left_on="Compound",
        right_on="Drug",
        how="inner"
    )

    # Define base features (TP2 + combo features)
    base_input_features = timepoints["tp2"] + combo_features

    # For tp6, also include tp3, tp4, tp5
    all_features = base_input_features.copy()
    for prev_tp_num in range(3, 6):
        all_features.extend(timepoints[f"tp{prev_tp_num}"])

    # Imputation
    X_imputed = impute_within_compound(
        X,
        columns=all_features,
        grouping_column='Compound'
    )

    # Train/test split
    train, test = split_by_compound(
        X_imputed,
        grouping_column='Compound',
        test_size=0.2,
        random_state=CONFIG["random_state"]
    )

    return train, test, all_features


def create_improved_shap_plot(shap_values, X_data, output_path, title, max_display=None):
    """Create a clean, publication-ready SHAP summary plot."""

    # Add feature categories
    feature_categories = [categorize_features(feat) for feat in X_data.columns]

    # Calculate mean absolute SHAP values for sorting
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Limit features if max_display is set
    if max_display:
        top_indices = np.argsort(mean_abs_shap)[-max_display:]
        shap_values_plot = shap_values[:, top_indices]
        X_data_plot = X_data.iloc[:, top_indices]
        feature_categories_plot = [feature_categories[i] for i in top_indices]
    else:
        shap_values_plot = shap_values
        X_data_plot = X_data
        feature_categories_plot = feature_categories

    # Sort features by mean absolute SHAP
    mean_abs_local = np.mean(np.abs(shap_values_plot), axis=0)
    order = np.argsort(mean_abs_local)

    shap_values_sorted = shap_values_plot[:, order]
    X_data_sorted = X_data_plot.iloc[:, order]
    feature_categories_sorted = [feature_categories_plot[i] for i in order]
    feature_names = X_data_sorted.columns.tolist()

    # Normalize feature values for color mapping (0 to 1)
    X_normalized = X_data_sorted.copy()
    for col in X_normalized.columns:
        col_min = X_normalized[col].min()
        col_max = X_normalized[col].max()
        if col_max > col_min:
            X_normalized[col] = (X_normalized[col] - col_min) / (col_max - col_min)
        else:
            X_normalized[col] = 0.5

    # Create figure with custom styling
    fig, ax = plt.subplots(figsize=CONFIG["figsize"])

    # Set background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Plot parameters
    dot_size = 25  # Larger dots for better visibility
    alpha = 0.8

    # Use a better colormap (red to blue) with stronger contrast
    from matplotlib.colors import LinearSegmentedColormap
    colors_map = ['#1E3A8A', '#3B82F6', '#93C5FD', '#D1D5DB', '#FCA5A5', '#EF4444', '#991B1B']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('shap', colors_map, N=n_bins)

    # Plot each feature
    for i, feature in enumerate(feature_names):
        shap_vals = shap_values_sorted[:, i]
        feature_vals = X_normalized.iloc[:, i].values

        # Add slight jitter to y-position for better visibility
        y_pos = np.full(len(shap_vals), i) + np.random.randn(len(shap_vals)) * 0.15

        scatter = ax.scatter(
            shap_vals,
            y_pos,
            c=feature_vals,
            cmap=cmap,
            s=dot_size,
            alpha=alpha,
            edgecolors='none',
            vmin=0,
            vmax=1,
            rasterized=False  # For better rendering in PDFs
        )

    # Set y-axis
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_ylim(-1, len(feature_names))

    # Color y-tick labels by category
    for i, label in enumerate(ax.get_yticklabels()):
        cat = feature_categories_sorted[i]
        color = FEATURE_CATEGORY_COLORS.get(cat, FEATURE_CATEGORY_COLORS["unknown"])
        label.set_color(color)
        label.set_fontweight('medium')

    # Set x-axis
    ax.set_xlabel('SHAP value (impact on model output)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10)

    # Add vertical line at x=0
    ax.axvline(x=0, color='#999999', linestyle='-', linewidth=1.0, zorder=1)

    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

    # Add subtle grid
    ax.grid(True, axis='x', alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)

    # Title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
    cbar.set_label('Feature value', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_color('#CCCCCC')

    # Customize colorbar ticks
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])

    # Add legend for feature categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=FEATURE_CATEGORY_COLORS['marmoset'],
              edgecolor='black', linewidth=0.5, label='Marmoset'),
        Patch(facecolor=FEATURE_CATEGORY_COLORS['simple PK'],
              edgecolor='black', linewidth=0.5, label='Simple PK'),
        Patch(facecolor=FEATURE_CATEGORY_COLORS['LIDS'],
              edgecolor='black', linewidth=0.5, label='LIDS'),
        Patch(facecolor=FEATURE_CATEGORY_COLORS['simple equipotent'],
              edgecolor='black', linewidth=0.5, label='Simple Equipotent'),
    ]
    legend = ax.legend(
        handles=legend_elements,
        title='Feature Category',
        bbox_to_anchor=(1.15, 0.5),
        loc='center left',
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        edgecolor='#CCCCCC',
        fancybox=False
    )
    legend.get_frame().set_alpha(0.95)

    plt.tight_layout()
    plt.savefig(output_path, dpi=CONFIG["dpi"], bbox_inches='tight', format='svg')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main function to generate manuscript SHAP plots."""

    print("=" * 70)
    print("Creating Manuscript SHAP Plots")
    print("=" * 70)

    # Set random seed
    np.random.seed(CONFIG["random_state"])

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")

    # Load model
    print(f"\n✓ Loading model from: {CONFIG['model_path']}")
    model = joblib.load(CONFIG["model_path"])

    # Load and prepare data
    print("\n✓ Loading and preparing data...")
    train, test, all_features = load_and_prepare_data()

    # Sample training data for SHAP
    n_samples = min(CONFIG["shap_sample_size"], len(train))
    print(f"\n✓ Sampling {n_samples} training instances for SHAP analysis...")

    X_train_sample = (
        train.sample(
            n=n_samples,
            with_replacement=False,
            seed=CONFIG["random_state"]
        )
        .select(all_features)
        .to_pandas()
    )

    # Calculate SHAP values
    print("\n✓ Calculating SHAP values (this may take a few minutes)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_sample)

    print(f"  - SHAP values shape: {shap_values.shape}")

    # Create full SHAP plot
    print("\n✓ Creating full SHAP summary plot...")
    full_output_path = output_dir / f"figure5_shap_summary_{CONFIG['output_name']}_full.svg"
    create_improved_shap_plot(
        shap_values,
        X_train_sample,
        full_output_path,
        f"SHAP Feature Importance for {CONFIG['output_name']} Prediction",
        max_display=30  # Show top 30 features
    )

    # Create zoomed plot (dropping MeanHU features only)
    print("\n✓ Creating zoomed SHAP summary plot...")

    # Identify MeanHU feature indices
    meanhu_indices = [i for i, col in enumerate(X_train_sample.columns) if 'MeanHU' in col]
    non_meanhu_indices = [i for i in range(X_train_sample.shape[1]) if i not in meanhu_indices]

    print(f"  - Dropping {len(meanhu_indices)} MeanHU features: {[X_train_sample.columns[i] for i in meanhu_indices]}")

    zoom_shap = shap_values[:, non_meanhu_indices]
    zoom_X = X_train_sample.iloc[:, non_meanhu_indices]

    zoom_output_path = output_dir / f"figure5_shap_summary_{CONFIG['output_name']}_zoomed.svg"
    create_improved_shap_plot(
        zoom_shap,
        zoom_X,
        zoom_output_path,
        f"SHAP Feature Importance (Lower-Ranked Features) for {CONFIG['output_name']}",
        max_display=25  # Show top 25 of remaining features
    )

    # Print summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total features: {len(all_features)}")
    print(f"Training samples used: {n_samples}")
    print(f"\nTop 10 features by mean |SHAP|:")

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_order = np.argsort(mean_abs_shap)[::-1]
    top_10_indices = feature_order[:10]
    for i, idx in enumerate(top_10_indices, 1):
        feat_name = X_train_sample.columns[idx]
        mean_shap = mean_abs_shap[idx]
        print(f"  {i:2d}. {feat_name:60s} {mean_shap:.4f}")

    print("\n" + "=" * 70)
    print("✓ All plots created successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
