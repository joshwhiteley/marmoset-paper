#!/usr/bin/env python3
"""
Create final radiodensity figure with Panel A (performance metrics) and Panel B (realistic trajectories).
Uses actual modeling performance and real lesion prediction data.
"""

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns
import joblib
from datetime import datetime

# Import your data classes and functions
from data.MarmosetData import MarmosetData
from data.InVitroData import DiamondData
from helpers.ModelingFunctions import impute_within_compound, split_by_compound
from helpers.constants import MARMOSET_ONLY_TUPLE, MARMOSET_IN_VITRO_TUPLE

# Set style for publication-ready figures
plt.style.use('default')
sns.set_palette("Set2")

def load_radiodensity_metrics() -> pl.DataFrame:
    """Load and process radiodensity modeling metrics data."""

    # Use the actual files that exist
    results_dir = Path("modeling/results")
    m_file = results_dir / "20250611_m_42_performance.csv"
    b_file = results_dir / "20250612_b_42_performance.csv"

    # Load the data
    m_df = (
        pl.read_csv(m_file)
        .with_columns(pl.lit("marmoset only").alias("model_type"))
    )
    b_df = (
        pl.read_csv(b_file)
        .with_columns(pl.lit("marmoset + in vitro").alias("model_type"))
    )

    # Combine datasets
    df = pl.concat([m_df, b_df])

    # Filter for HU only (radiodensity)
    df = df.filter(pl.col("output").str.contains("HU"))

    # Add week mapping: timepoint 3->2, 4->4, 5->6, 6->8
    timepoint_to_week = {
        3: 2,
        4: 4,
        5: 6,
        6: 8
    }

    df = df.with_columns(
        pl.col("timepoint").replace(timepoint_to_week).alias("week")
    )

    return df

def load_sample_lesion_data(n_compounds: int = 2, n_lesions_per_compound: int = 3, specific_compounds: List[str] = None):
    """Load sample lesion trajectory data from actual models."""

    print("Loading sample lesion data...")

    # Load data
    marmoset_data = MarmosetData("data/marm_data_wide_clustered_classif.csv")
    diamond_data = DiamondData("data/in_vitro_diamond_data.csv")
    severe_lesions = marmoset_data.get_severe_lesions().data

    # Feature preparation
    y_features = ["MeanHU"]
    metadata_features = ["Compound", "Lesion"]
    y_cols = [col for col in severe_lesions.columns if any(f in col for f in y_features)]
    severe_data = severe_lesions.select(y_cols + metadata_features)

    timepoints = {
        f"tp{tp}": [col for col in severe_data.columns if f"TP{tp}" in col and "MeanHU" in col]
        for tp in range(2, 7)
    }

    # In vitro features
    diamond_df = diamond_data.data.filter(pl.col('NumbDrugs') > 1).drop('NumbDrugs')
    combo_features = [col for col in diamond_df.columns if "Drug" not in col]

    # Merge data
    X = severe_data.join(other=diamond_df, left_on="Compound", right_on="Drug", how="inner")

    # Get sample compounds
    compounds = X['Compound'].unique().to_list()
    print(f"Available compounds: {compounds}")

    # Look for MRZE combination (MOX, RIF, PZA, EMB/ETH)
    mrze_candidates = []
    for compound in compounds:
        comp_upper = compound.upper()
        # Check for combinations that might be MRZE
        if (('MOX' in comp_upper or 'MXF' in comp_upper) and
            ('RIF' in comp_upper or 'RFP' in comp_upper) and
            ('PZA' in comp_upper) and
            ('EMB' in comp_upper or 'ETH' in comp_upper)):
            mrze_candidates.append(compound)

    print(f"MRZE candidates: {mrze_candidates}")

    if specific_compounds:
        # Filter for specific compounds requested
        available_specific = [c for c in specific_compounds if c in compounds]
        selected_compounds = available_specific[:n_compounds]
        print(f"Using specific compounds: {selected_compounds}")
    else:
        selected_compounds = compounds[:n_compounds]

    results = {}

    for strategy in ['marmoset_only', 'combined']:
        if strategy == 'marmoset_only':
            base_input_features = timepoints["tp2"]
            model_prefix = "m"
        else:
            base_input_features = timepoints["tp2"] + combo_features
            model_prefix = "b"

        # Process data
        X_imputed = impute_within_compound(X, columns=base_input_features, grouping_column='Compound')
        train, test = split_by_compound(X_imputed, grouping_column='Compound', test_size=0.2, random_state=42)

        # Filter for selected compounds
        test_sample = test.filter(pl.col('Compound').is_in(selected_compounds))

        # Set seed for reproducible lesion selection
        np.random.seed(5674)

        predictions = test_sample.select(["Compound", "Lesion"] + timepoints["tp2"]).clone()

        # Load models and predict
        model_dir = Path("radiodensity-only/modeling/models")

        for tp_num in range(3, 7):
            tp_key = f"tp{tp_num}"
            model_file = model_dir / f"model_{tp_key}_{model_prefix}_seed42_20250918.joblib"

            if model_file.exists():
                model = joblib.load(model_file)
                features_for_prediction = base_input_features.copy()
                for prev_tp_num in range(3, tp_num):
                    prev_tp_key = f"tp{prev_tp_num}"
                    features_for_prediction.extend(timepoints[prev_tp_key])

                X_pred = test_sample.select(features_for_prediction)
                y_pred = model.predict(X_pred)

                target_col = timepoints[tp_key][0]
                predictions = predictions.with_columns(
                    pl.Series(f"{target_col}_pred", y_pred)
                )

        results[strategy] = {
            'test_data': test_sample,
            'predictions': predictions,
            'timepoints': timepoints,
            'compounds': selected_compounds
        }

    return results

def create_final_radiodensity_figure(figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
    """Create the complete radiodensity figure with real Panel A and B."""

    # Load actual metrics data
    metrics_df = load_radiodensity_metrics()

    # Load sample lesion data for BDQ+LIN+PRE and MRZE (EMB+MOX+PZA+RIF)
    target_compounds = ['BDQ+LIN+PRE', 'EMB+MOX+PZA+RIF']
    lesion_results = load_sample_lesion_data(n_compounds=2, n_lesions_per_compound=3, specific_compounds=target_compounds)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # Colors from your constants
    colors = {
        'marmoset only': MARMOSET_ONLY_TUPLE,
        'marmoset + in vitro': MARMOSET_IN_VITRO_TUPLE
    }

    weeks = [2, 4, 6, 8]
    x_pos = np.arange(len(weeks))
    width = 0.35

    # PANEL A: MSE and R² plots
    # MSE subplot
    ax_mse = plt.subplot2grid((3, 4), (0, 0), colspan=2)

    mse_marmoset = []
    mse_combined = []
    for week in weeks:
        # Marmoset only
        marm_data = metrics_df.filter(
            (pl.col('model_type') == 'marmoset only') &
            (pl.col('week') == week)
        )
        mse_marmoset.append(marm_data['mse'][0] if len(marm_data) > 0 else 0)

        # Marmoset + in vitro
        comb_data = metrics_df.filter(
            (pl.col('model_type') == 'marmoset + in vitro') &
            (pl.col('week') == week)
        )
        mse_combined.append(comb_data['mse'][0] if len(comb_data) > 0 else 0)

    bars1 = ax_mse.bar(x_pos - width/2, mse_marmoset, width,
                       label='marmoset only', color=colors['marmoset only'], alpha=0.8)
    bars2 = ax_mse.bar(x_pos + width/2, mse_combined, width,
                       label='marmoset + in vitro', color=colors['marmoset + in vitro'], alpha=0.8)

    ax_mse.set_xlabel('weeks from treatment start')
    ax_mse.set_ylabel('HU²')
    ax_mse.set_title('mean squared error')
    ax_mse.set_xticks(x_pos)
    ax_mse.set_xticklabels(weeks)
    ax_mse.set_ylim(0, 2750)
    ax_mse.grid(True, alpha=0.3)

    # Add "best" arrow pointing downward
    ax_mse.annotate('best', xy=(-0.15, 2600), xytext=(-0.15, 2400),
                    fontsize=12, ha='center', va='top', rotation=90,
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # R² subplot
    ax_r2 = plt.subplot2grid((3, 4), (0, 2), colspan=2)

    r2_marmoset = []
    r2_combined = []
    for week in weeks:
        # Marmoset only
        marm_data = metrics_df.filter(
            (pl.col('model_type') == 'marmoset only') &
            (pl.col('week') == week)
        )
        r2_marmoset.append(marm_data['r2'][0] if len(marm_data) > 0 else 0)

        # Marmoset + in vitro
        comb_data = metrics_df.filter(
            (pl.col('model_type') == 'marmoset + in vitro') &
            (pl.col('week') == week)
        )
        r2_combined.append(comb_data['r2'][0] if len(comb_data) > 0 else 0)

    bars3 = ax_r2.bar(x_pos - width/2, r2_marmoset, width,
                      label='marmoset only', color=colors['marmoset only'], alpha=0.8)
    bars4 = ax_r2.bar(x_pos + width/2, r2_combined, width,
                      label='marmoset + in vitro', color=colors['marmoset + in vitro'], alpha=0.8)

    ax_r2.set_xlabel('weeks from treatment start')
    ax_r2.set_ylabel('R²')
    ax_r2.set_title('coef. of determination')
    ax_r2.set_xticks(x_pos)
    ax_r2.set_xticklabels(weeks)
    ax_r2.set_ylim(0, 0.9)
    ax_r2.grid(True, alpha=0.3)

    # Add "best" arrow pointing upward
    ax_r2.annotate('best', xy=(-0.15, 0.1), xytext=(-0.15, 0.25),
                   fontsize=12, ha='center', va='bottom', rotation=90,
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Add Panel A label
    ax_mse.text(-0.15, 1.1, 'A', transform=ax_mse.transAxes, fontsize=18,
                fontweight='bold', va='bottom', ha='right')

    # PANEL B: Real lesion trajectory plots
    weeks_traj = np.array([0, 2, 4, 6, 8])

    # Create subplots for lesion trajectories
    compounds = lesion_results['marmoset_only']['compounds']

    for strategy_idx, (strategy, strategy_data) in enumerate(lesion_results.items()):
        for comp_idx, compound in enumerate(compounds):

            # Calculate subplot position
            row = strategy_idx + 1  # Rows 1 and 2
            col = comp_idx * 2      # Columns 0-1, 2-3

            print(f"Creating subplot for {strategy} - {compound} at row={row}, col={col}")
            ax = plt.subplot2grid((3, 4), (row, col), colspan=2)

            # Filter data for this compound
            test_data = strategy_data['test_data'].filter(pl.col('Compound') == compound)
            pred_data = strategy_data['predictions'].filter(pl.col('Compound') == compound)

            print(f"Test data rows for {compound}: {len(test_data)}")
            print(f"Pred data rows for {compound}: {len(pred_data)}")

            # Get lesions for this compound and select with seed
            lesions = test_data['Lesion'].unique().to_list()
            np.random.seed(5674)  # Use specified seed for lesion selection
            if len(lesions) > 3:
                selected_lesions = sorted(np.random.choice(lesions, size=3, replace=False))
            else:
                selected_lesions = lesions[:3]

            print(f"Available lesions for {compound}: {lesions}")
            print(f"Selected lesions: {selected_lesions}")

            # Define compound-specific color schemes (darker for better visibility)
            compound_colors = {
                'BDQ+LIN+PRE': ['#E91E63', '#4CAF50', '#2196F3'],  # Darker pink, green, blue
                'EMB+MOX+PZA+RIF': ['#FF9800', '#FF5722', '#9C27B0']  # Darker orange, red-orange, purple
            }

            # Get colors for this compound
            if compound in compound_colors:
                lesion_colors = compound_colors[compound]
            else:
                # Fallback to default darker colors
                lesion_colors = ['#E91E63', '#4CAF50', '#2196F3']

            all_hu_values = []  # For auto-scaling Y-axis

            for lesion_idx, lesion in enumerate(selected_lesions):
                lesion_test = test_data.filter(pl.col('Lesion') == lesion)
                lesion_pred = pred_data.filter(pl.col('Lesion') == lesion)

                if len(lesion_test) == 0 or len(lesion_pred) == 0:
                    print(f"Warning: No data for lesion {lesion} in compound {compound}")
                    continue

                # Extract HU values
                actual_hu = []
                predicted_hu = []

                for tp_num in range(2, 7):
                    col_name = f"TP{tp_num}_MeanHU"
                    if col_name in lesion_test.columns:
                        val = lesion_test[col_name][0]
                        if val is not None and not np.isnan(val):
                            actual_hu.append(val)
                            all_hu_values.append(val)
                        else:
                            actual_hu.append(None)

                    if tp_num > 2:  # Predictions start from TP3
                        pred_col = f"{col_name}_pred"
                        if pred_col in lesion_pred.columns:
                            val = lesion_pred[pred_col][0]
                            if val is not None and not np.isnan(val):
                                predicted_hu.append(val)
                                all_hu_values.append(val)
                            else:
                                predicted_hu.append(None)

                print(f"Lesion {lesion}: actual_hu={actual_hu}, predicted_hu={predicted_hu}")

                # Plot actual (full trajectory) - handle None values
                if len(actual_hu) == 5 and any(x is not None for x in actual_hu):
                    # Filter out None values for plotting
                    plot_weeks = []
                    plot_hu = []
                    for i, hu_val in enumerate(actual_hu):
                        if hu_val is not None:
                            plot_weeks.append(weeks_traj[i])
                            plot_hu.append(hu_val)

                    if len(plot_hu) > 0:
                        ax.plot(plot_weeks, plot_hu, 'o-', color=lesion_colors[lesion_idx],
                               linewidth=2, markersize=5, alpha=0.8, label=f'L{lesion} actual')

                # Plot predicted (TP3-TP6)
                if len(predicted_hu) == 4 and len(actual_hu) > 0 and actual_hu[0] is not None:
                    pred_trajectory = [actual_hu[0]] + predicted_hu
                    # Filter out None values
                    plot_weeks_pred = []
                    plot_hu_pred = []
                    for i, hu_val in enumerate(pred_trajectory):
                        if hu_val is not None:
                            plot_weeks_pred.append(weeks_traj[i])
                            plot_hu_pred.append(hu_val)

                    if len(plot_hu_pred) > 0:
                        ax.plot(plot_weeks_pred, plot_hu_pred, 'o--', color=lesion_colors[lesion_idx],
                               linewidth=2, markersize=5, alpha=0.7, label=f'L{lesion} pred')

            # Styling
            strategy_label = 'marmoset only' if strategy == 'marmoset_only' else 'marmoset + in vitro'

            # Add colored background for title
            ax.add_patch(patches.Rectangle((0, 1.02), 1, 0.08,
                                          transform=ax.transAxes,
                                          facecolor=colors[strategy_label], alpha=0.7,
                                          clip_on=False))

            ax.text(0.5, 1.05, strategy_label, transform=ax.transAxes,
                    fontsize=10, fontweight='bold', ha='center', va='center', color='white')
            ax.text(0.5, 1.025, f'radiodensity - {compound}', transform=ax.transAxes,
                    fontsize=8, ha='center', va='center', color='white')

            ax.set_ylabel('HU')
            ax.set_xlabel('weeks from treatment start')
            ax.set_xticks(weeks_traj)
            ax.set_xlim(-0.5, 8.5)

            # Set consistent Y-axis ranges for each compound
            compound_y_ranges = {
                'BDQ+LIN+PRE': (-200, 0),  # BPaL (Bedaquiline + Pretomanid + Linezolid)
                'EMB+MOX+PZA+RIF': (-400, -100)  # MRZE
            }

            if compound in compound_y_ranges:
                y_min, y_max = compound_y_ranges[compound]
                ax.set_ylim(y_min, y_max)
                print(f"Y-axis for {compound}: [{y_min}, {y_max}] (fixed range)")
            else:
                # Auto-scale for unknown compounds
                if all_hu_values:
                    y_min = min(all_hu_values)
                    y_max = max(all_hu_values)
                    y_range = y_max - y_min
                    y_margin = y_range * 0.1  # 10% margin
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)
                    print(f"Y-axis for {compound}: [{y_min - y_margin:.0f}, {y_max + y_margin:.0f}] (auto-scaled)")
                else:
                    ax.set_ylim(-400, -50)  # Default if no data
                    print(f"No data found for {compound}, using default Y-axis")

            ax.grid(True, alpha=0.3)

            # Add Panel B label to the first subplot only
            if strategy_idx == 0 and comp_idx == 0:
                ax.text(-0.1, 1.1, 'B', transform=ax.transAxes, fontsize=18,
                       fontweight='bold', va='bottom', ha='right')

    # Add legend at the bottom
    fig.legend(['marmoset only', 'marmoset + in vitro'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02),
               title='modeling strategy:', title_fontsize=12, fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.95)

    return fig

def main():
    """Main function to create the final radiodensity figure."""

    # Create output directory
    output_dir = Path("figures_v2/radiodensity_only")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating final radiodensity figure with realistic trajectories...")

    try:
        fig = create_final_radiodensity_figure()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d")
        fig.savefig(output_dir / f"final_radiodensity_figure_{timestamp}.svg",
                    dpi=300, bbox_inches='tight', format='svg')
        fig.savefig(output_dir / f"final_radiodensity_figure_{timestamp}.png",
                    dpi=300, bbox_inches='tight', format='png')

        print(f"Figure saved to: {output_dir}")
        print(f"Files created: final_radiodensity_figure_{timestamp}.svg/png")

        # Close figures to avoid hanging
        plt.close('all')

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()