import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific warnings if needed
# warnings.filterwarnings("ignore", category=UserWarning)
# Suppress the specific FutureWarning about the median callable in pivot_table
warnings.filterwarnings("ignore", message="The provided callable <function median", category=FutureWarning)


def analyze_features_stacked_heatmap_sorted_no_annot(data_path: str, output_dir: str = "figures/stacked_comparison_sorted_no_annot"):
    """
    Analyzes scaled delta features, creating vertically stacked heatmaps of median
    changes for 'cool' (top) and 'hot' (bottom) classifications, grouped by Compound,
    WITHOUT numerical annotations on the cells.

    Compounds are sorted first by the number of drugs in combination (descending,
    based on '+' count) and then alphabetically.

    Calculates delta features (TP6-TP2) and scales them. Ensures consistent
    ordering of compounds and features for direct comparison.

    Args:
        data_path: Path to the data file (must contain TP2/TP6 features,
                   'classif', and 'Compound' columns).
        output_dir: Directory to save the generated plots.
    """
    print(f"Starting stacked heatmap analysis (no annotations) with custom sorting using data: {data_path}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Read Data ---
    try:
        df = pl.read_csv(data_path)
        print(f"Successfully read data. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # --- Define Features for Delta Calculation ---
    features_to_delta = [
        "TP2_MeanHU", "TP2_StandDevHU", "TP2_SoftVol",
        "TP2_HardVol", "TP2_MeanSUV"
    ]
    print(f"Defining features for delta calculation: {features_to_delta}")

    # --- Calculate Delta Features (TP6 - TP2) ---
    print("Calculating DeltaTp2_6 features...")
    delta_features = []
    missing_base_cols = []
    for feat_tp2 in features_to_delta:
        if feat_tp2 not in df.columns:
            missing_base_cols.append(feat_tp2)
            continue

        base_feat_name = feat_tp2.split('_', 1)[1]
        feat_tp6 = f"TP6_{base_feat_name}"
        delta_feat_name = f"DeltaTp2_6_{base_feat_name}"

        if feat_tp6 in df.columns:
            try:
                df = df.with_columns(
                    (pl.col(feat_tp6).cast(pl.Float64, strict=False) -
                     pl.col(feat_tp2).cast(pl.Float64, strict=False))
                    .alias(delta_feat_name)
                )
                delta_features.append(delta_feat_name)
            except Exception as e:
                print(f"  Error calculating delta for {base_feat_name}: {e}. Skipping.")
        else:
            print(f"  Warning: TP6 counterpart '{feat_tp6}' not found for '{feat_tp2}'. Skipping delta.")

    if missing_base_cols:
         print(f"  Warning: Base TP2 features missing: {missing_base_cols}")
    if not delta_features:
        print("Error: No delta features calculated. Check TP2/TP6 columns.")
        return
    print(f"Calculated {len(delta_features)} delta features.")
    feature_order = delta_features

    # --- Scale Delta Features ---
    print("Scaling delta features...")
    fill_null_exprs = [pl.col(f).fill_null(pl.median(f)).alias(f) for f in delta_features]
    df = df.with_columns(fill_null_exprs)

    delta_data_np = df.select(delta_features).to_numpy()
    if np.any(np.isnan(delta_data_np)) or np.any(np.isinf(delta_data_np)):
        print("  Warning: NaNs/Infs detected before scaling. Filling with 0.")
        delta_data_np = np.nan_to_num(delta_data_np, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    try:
        scaled_data = scaler.fit_transform(delta_data_np)
        scaled_delta_features = [f"{feat}_scaled" for feat in delta_features]
        scaled_df = pl.DataFrame(scaled_data, schema=scaled_delta_features)
        df = pl.concat([df, scaled_df], how='horizontal')
        print(f"Scaled features: {scaled_delta_features}")
        feature_order_scaled = scaled_delta_features
    except ValueError as e:
        print(f"Error scaling: {e}. Check for zero variance.")
        return
    except Exception as e:
        print(f"Unexpected scaling error: {e}")
        return

    # --- Ensure Required Columns Exist ---
    required_cols = ['classif', 'Compound'] + scaled_delta_features
    if 'Compound' not in df.columns:
        found_compound = next((col for col in df.columns if col.lower() == 'compound'), None)
        if found_compound:
            print(f"Found '{found_compound}', renaming to 'Compound'.")
            df = df.rename({found_compound: 'Compound'})
        else:
             print("Error: 'Compound' column not found (case-insensitive).")
             return

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns after calculations: {missing_cols}")
        print("Available columns:", df.columns)
        return

    # --- Prepare Data for Heatmaps ---
    print("Preparing data for heatmaps...")
    df_pd = df.select(['Compound', 'classif'] + feature_order_scaled).to_pandas()
    median_agg_data = df_pd.groupby(['Compound', 'classif'])[feature_order_scaled].median()

    try:
        pivot_median = pd.pivot_table(
            df_pd,
            values=feature_order_scaled,
            index=['Compound'],
            columns=['classif'],
            aggfunc="median" # Use string "median"
        )

        # --- Custom Compound Sorting ---
        unique_compounds = pivot_median.index.unique().tolist()
        compound_order = sorted(unique_compounds, key=lambda s: (-s.count('+'), s))
        print(f"Applying compound order: {compound_order}")
        # --- End Custom Sorting ---

        if 'cool' in pivot_median.columns.get_level_values(1):
            cool_data = pivot_median.xs('cool', level=1, axis=1)
            cool_data = cool_data.reindex(index=compound_order, columns=feature_order_scaled)
        else:
            print("Warning: No data found for 'cool' classification.")
            cool_data = pd.DataFrame(np.nan, index=compound_order, columns=feature_order_scaled)

        if 'hot' in pivot_median.columns.get_level_values(1):
            hot_data = pivot_median.xs('hot', level=1, axis=1)
            hot_data = hot_data.reindex(index=compound_order, columns=feature_order_scaled)
        else:
            print("Warning: No data found for 'hot' classification.")
            hot_data = pd.DataFrame(np.nan, index=compound_order, columns=feature_order_scaled)

        cool_data_hm = cool_data.T
        hot_data_hm = hot_data.T

        clean_feature_labels_map = {
            feat: feat.replace('_scaled', '').replace('DeltaTp2_6_', 'Change in ')
            for feat in feature_order_scaled
        }
        cool_data_hm = cool_data_hm.rename(index=clean_feature_labels_map)
        hot_data_hm = hot_data_hm.rename(index=clean_feature_labels_map)

    except KeyError as e:
         print(f"Error processing classifications in pivot table: {e}. Ensure both 'hot' and 'cool' exist or handle missing classifications.")
         return
    except Exception as e:
        print(f"Error pivoting or preparing heatmap data: {e}")
        return

    if cool_data_hm.empty and hot_data_hm.empty:
        print("Error: No data available for either 'cool' or 'hot' groups after processing.")
        return

    # --- Generate Stacked Heatmaps ---
    print("Generating stacked heatmaps (no annotations)...")

    # Determine global min/max for consistent color scaling
    all_values = pd.concat([cool_data_hm.stack(), hot_data_hm.stack()]).dropna()
    if all_values.empty:
        print("Warning: No numeric data found for heatmap color scaling. Using default.")
        vmin, vmax, center = None, None, None
    else:
        vmin = all_values.min()
        vmax = all_values.max()
        center = 0
        if vmin < center < vmax:
             abs_max = max(abs(vmin - center), abs(vmax - center))
             vmin = center - abs_max
             vmax = center + abs_max

    # Create figure with two subplots, stacked vertically
    n_features = len(cool_data_hm.index)
    n_compounds = len(cool_data_hm.columns)
    fig_width = 6 + n_compounds * 0.6
    fig_height = 4 + n_features * 0.3 * 2 + 1
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
    fig.suptitle('Median Scaled Feature Changes: Cool vs. Hot Classification (Sorted Compounds)', fontsize=16, y=0.98)

    common_heatmap_kws = dict(
        cmap='RdBu_r',
        center=center,
        annot=False,  # <--- Set annot to False to remove numbers
        fmt=".2f",    # fmt is ignored when annot=False, but kept for consistency
        linewidths=.5,
        cbar=False,
        vmin=vmin,
        vmax=vmax
    )

    # Heatmap 1: Cool (Top)
    sns.heatmap(cool_data_hm, ax=axes[0], **common_heatmap_kws)
    axes[0].set_title('Cool Classification')
    axes[0].set_ylabel('Feature Change')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', bottom=False, labelbottom=False)
    axes[0].tick_params(axis='y', rotation=0)


    # Heatmap 2: Hot (Bottom)
    sns.heatmap(hot_data_hm, ax=axes[1], **common_heatmap_kws)
    axes[1].set_title('Hot Classification')
    axes[1].set_ylabel('Feature Change')
    axes[1].set_xlabel('Compound')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')


    # Add a single shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])

    if vmin is not None and vmax is not None:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=common_heatmap_kws['cmap'], norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label='Median Scaled Change')
    else:
        cbar_ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    # Save the figure
    try:
        # Update filename to reflect no annotations
        heatmap_path = output_path / "stacked_median_heatmap_sorted_compounds_no_annot.png"
        heatmap_path_svg = output_path / "stacked_median_heatmap_sorted_compounds_no_annot.svg"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.savefig(heatmap_path_svg, format='svg')
        plt.close(fig)
        print(f"  Saved stacked heatmap (no annotations): {heatmap_path.name}")
    except Exception as e:
        print(f"  Error saving heatmap: {e}")
        plt.close(fig)


    print(f"\nAnalysis complete. Plot saved in '{output_path}'.")


def main():
    """Main function"""
    input_data_path = "data/marm_data_wide_clustered_classif.csv"
    # Update function call and output directory name
    analyze_features_stacked_heatmap_sorted_no_annot(input_data_path, output_dir="figures/stacked_comparison_sorted_no_annot")


if __name__ == "__main__":
    main()
