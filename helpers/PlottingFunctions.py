
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Union, List, Optional

from helpers.ModelingFunctions import get_compound_abbreviation

def plot_lesion_progressions(
    compound: str,
    num_lesions: int,
    test_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    debug: bool = False,
    save_dir: str = "figures/predictions",
    save_bool: bool = True,
    random_state: int = 42
) -> None:
    """Plot lesion progressions for a given compound.
    
    Args:
        compound: Name of the compound to plot
        num_lesions: Number of random lesions to plot
        test_df: DataFrame containing true values
        pred_df: DataFrame containing predicted values
        debug: Whether to print debug information
        save_dir: Directory to save figures
        save_bool: Whether to save the figure
        random_state: Random seed for reproducibility
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Get abbreviated compound name
    abbrev = get_compound_abbreviation(compound)
    
    # Filter data for the given compound
    compound_test = test_df.filter(pl.col("Compound") == compound)
    compound_pred = pred_df.filter(pl.col("Compound") == compound)
    
    if debug:
        lesion_counts = compound_test.group_by("Lesion").agg(
            pl.len().alias("count")
        )
        print(f"\nNumber of data points per lesion for {compound}:")
        print(lesion_counts)
    
    # Randomly select lesions with fixed seed
    available_lesions = compound_test["Lesion"].unique().to_list()
    #np.random.seed(random_state)
    selected_lesions = np.random.choice(
        available_lesions,
        size=min(num_lesions, len(available_lesions)),
        replace=False
    )
    
    # Create figure with adjusted width to accommodate legends
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Lesion Progressions for {compound}", fontsize=16)
    
    # Timepoints for x-axis
    timepoints = ['TP2', 'TP3', 'TP4', 'TP5', 'TP6']
    x = np.arange(len(timepoints))
    
    # Create a color palette for the lesions
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_lesions)))
    
    # Plot each lesion
    for idx, lesion in enumerate(selected_lesions):
        # Get data for this lesion
        lesion_test = compound_test.filter(pl.col("Lesion") == lesion)
        lesion_pred = compound_pred.filter(pl.col("Lesion") == lesion)
        
        # Extract values - take the first value if multiple exist
        hu_true = [lesion_test[f"TP{i}_MeanHU"].to_list()[0] 
                  for i in range(2, 7)]
        hu_pred = [lesion_pred[f"TP{i}_MeanHU"].to_list()[0] 
                  for i in range(3, 7)]
        hu_pred = [hu_true[0]] + hu_pred  # Add TP2 value
        
        suv_true = [lesion_test[f"TP{i}_MeanSUV"].to_list()[0] 
                   for i in range(2, 7)]
        suv_pred = [lesion_pred[f"TP{i}_MeanSUV"].to_list()[0] 
                   for i in range(3, 7)]
        suv_pred = [suv_true[0]] + suv_pred  # Add TP2 value
        
        # Plot MeanHU with matching colors
        ax1.plot(x, hu_true, 'o-', color=colors[idx], 
                label=f'Lesion {lesion}', alpha=0.7)
        ax1.plot(x, hu_pred, 'o--', color=colors[idx], alpha=0.7)
        
        # Plot MeanSUV with matching colors
        ax2.plot(x, suv_true, 'o-', color=colors[idx], 
                label=f'Lesion {lesion}', alpha=0.7)
        ax2.plot(x, suv_pred, 'o--', color=colors[idx], alpha=0.7)
    
    # Customize MeanHU plot
    ax1.set_title('MeanHU')
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('MeanHU')
    # figure
    ax1.set_ylim(-375, -25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(timepoints)
    ax1.grid(True, alpha=0.3)
    
    # Customize MeanSUV plot
    ax2.set_title('MeanSUV')
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('MeanSUV')
    ax2.set_xticks(x)
    ax2.set_xticklabels(timepoints)
    ax2.set_ylim(0.3, 5.2)
    ax2.grid(True, alpha=0.3)
    
    # Create custom legend for line styles
    line_style_legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label='True'),
        Line2D([0], [0], color='black', linestyle='--', label='Predicted')
    ]
    
    # Add legends
    legend1 = ax1.legend(bbox_to_anchor=(1.05, 1), 
                        loc='upper left', title="Lesions")
    legend2 = ax1.legend(
        handles=line_style_legend_elements,
        bbox_to_anchor=(1.05, 0.5),
        loc='center left',
        title="Line Style"
    )
    ax1.add_artist(legend1)  # Add the first legend back after it was removed
    
    plt.tight_layout()
    
    if save_bool:
        save_file = save_path / f"{abbrev}_lesion_progressions.svg"
        plt.savefig(save_file, bbox_inches='tight', dpi=300, format='svg')
    plt.close()  # Close the figure to free memory

def plot_ground_truth_lesion_trajectory(
    data_df: pl.DataFrame,
    compound: str,
    num_lesions_to_plot: int = 5,
    specific_lesions: Optional[List[int]] = None,
    timepoint_style: str = "week",
    save_dir: str = "figures/ground_truth",
    save_bool: bool = True,
    random_state: int = 42,
    output_format: str = "png"
) -> None:
    """Plots the ground truth lesion trajectories for a given compound.

    Args:
        data_df: DataFrame containing the ground truth lesion data. 
                 Expected columns: 'Compound', 'Lesion', 'TP{i}_MeanHU', 'TP{i}_MeanSUV' (for i=2..6).
        compound: Name of the compound to plot.
        num_lesions_to_plot: Number of random lesions to plot if specific_lesions is None.
        specific_lesions: A list of specific lesion IDs to plot. If provided, num_lesions_to_plot is ignored.
        save_dir: Directory to save the figure.
        save_bool: Whether to save the figure.
        random_state: Random seed for reproducibility when selecting random lesions.
    """
    # Create save directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Get abbreviated compound name
    try:
        abbrev = get_compound_abbreviation(compound) # Assumes this function is accessible
    except Exception as e:
        print(f"Warning: Could not get abbreviation for {compound}. Using full name. Error: {e}")
        abbrev = compound.replace("+", "_") # Basic fallback for filename

    # Filter data for the given compound
    compound_data = data_df.filter(pl.col("Compound") == compound)

    if compound_data.is_empty():
        print(f"No data found for compound: {compound}")
        return

    available_lesions = compound_data["Lesion"].unique().to_list()
    
    if not available_lesions:
        print(f"No lesions found for compound: {compound}")
        return

    # Select lesions to plot
    if specific_lesions:
        selected_lesions = [l for l in specific_lesions if l in available_lesions]
        if not selected_lesions:
             print(f"None of the specified lesions {specific_lesions} found for compound {compound}.")
             return
        print(f"Plotting specified lesions for {compound}: {selected_lesions}")
    else:
        np.random.seed(random_state)
        selected_lesions = np.random.choice(
            available_lesions,
            size=min(num_lesions_to_plot, len(available_lesions)),
            replace=False
        )
        print(f"Plotting {len(selected_lesions)} random lesions for {compound}: {selected_lesions.tolist()}")


    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) # Adjusted figsize slightly
    fig.suptitle(f"Ground Truth Lesion Trajectories for {compound}", fontsize=16)

    # timepoints
    if timepoint_style == "week":
        timepoints = ['week 0', 'week 2', 'week 4', 'week 6', 'week 8']
    elif timepoint_style == 'tp': 
        timepoints = ['TP2', 'TP3', 'TP4', 'TP5', 'TP6']
    else:
        return KeyError(f"Invalid timepoint_style argument: {timepoint_style}. Choose from 'week' or 'tp'.")
    
    x = np.arange(len(timepoints))

    # Create a color palette for the lesions
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_lesions)))

    # Plot each selected lesion
    for idx, lesion in enumerate(selected_lesions):
        # Get data for this specific lesion
        lesion_data = compound_data.filter(pl.col("Lesion") == lesion)

        if lesion_data.is_empty():
            print(f"Warning: No data found for lesion {lesion} in compound {compound} after filtering.")
            continue
            
        # Extract ground truth values - take the first row if multiple exist for a lesion
        # This assumes the input df might have duplicates per lesion, takes the first. Adjust if needed.
        try:
            hu_true = [lesion_data[f"TP{i}_MeanHU"].to_list()[0] for i in range(2, 7)]
            suv_true = [lesion_data[f"TP{i}_MeanSUV"].to_list()[0] for i in range(2, 7)]
        except IndexError:
            print(f"Warning: Could not extract all timepoints for lesion {lesion}. Skipping.")
            continue
        except pl.ColumnNotFoundError as e:
             print(f"Warning: Missing expected column for lesion {lesion}. Error: {e}. Skipping.")
             continue


        # Plot MeanHU
        ax1.plot(x, hu_true, 'o-', color=colors[idx], label=f'Lesion {lesion}', alpha=0.8)

        # Plot MeanSUV
        ax2.plot(x, suv_true, 'o-', color=colors[idx], label=f'Lesion {lesion}', alpha=0.8)

    # Customize MeanHU plot
    ax1.set_title('MeanHU')
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('MeanHU')
    ax1.set_xticks(x)
    ax1.set_xticklabels(timepoints)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Lesions")

    # Customize MeanSUV plot
    ax2.set_title('MeanSUV')
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('MeanSUV')
    ax2.set_xticks(x)
    ax2.set_xticklabels(timepoints)
    ax2.grid(True, alpha=0.3)
    # Optional: Add legend to second plot if needed, or rely on the first plot's legend
    # ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Lesions")


    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to prevent legend overlap

    if save_bool:
        save_file = save_path / f"{abbrev}_ground_truth_lesions.{output_format}"
        try:
            plt.savefig(save_file, bbox_inches='tight', dpi=300, format=output_format)
            print(f"Plot saved to: {save_file}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.close(fig) 


def plot_mse_results(
    mse_df: pl.DataFrame,
    save_path: Union[str, Path],
    plot_title: Optional[str] = "Model Mean Squared Error (MSE) by Timepoint",
    save_prefix: Optional[str] = "",
    save_format: Optional[str] = "png",
    save_bool: bool = True
) -> None:
    """
    Generates and saves a bar chart visualizing MSE results by timepoint.

    Args:
        mse_df: Polars DataFrame with 'Timepoint' and 'MSE' columns.
        save_path: Full path (including filename, e.g., 'results/errors/mse_plot.png')
                   to save the plot image.
        plot_title: The main title for the plot.
        feature_set_name: Optional name describing the feature set used
                          (e.g., 'Marmoset Only', 'Both') to add to the title.
    """
    if mse_df is None or mse_df.is_empty():
        print("MSE DataFrame is empty or None. Skipping plot generation.")
        return

    if 'Timepoint' not in mse_df.columns or 'MSE' not in mse_df.columns:
        print(f"Error: mse_df must contain 'Timepoint' and 'MSE' columns. Found: {mse_df.columns}")
        return

    # Ensure data is sorted by timepoint for consistent plotting
    try:
        # Extract numeric part of timepoint for sorting (e.g., 'TP3' -> 3)
        mse_df = mse_df.with_columns(
            pl.col('Timepoint').str.extract(r'(\d+)', 1).cast(pl.Int64).alias('tp_num')
        ).sort('tp_num').drop('tp_num')
    except Exception as e:
        print(f"Warning: Could not sort timepoints numerically ({e}). Plotting in original order.")
        # Fallback sort alphabetically if numeric fails
        mse_df = mse_df.sort('Timepoint')


    timepoints = mse_df['Timepoint'].to_list()
    mse_values = mse_df['MSE'].to_list()

    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(timepoints, mse_values, color='skyblue')

    # Add labels on top of bars
    ax.bar_label(bars, fmt='%.4f', padding=3) # Format MSE to 4 decimal places

    ax.set_xlabel("Timepoint")
    ax.set_ylabel("Mean Squared Error (MSE)")

    ax.set_title(plot_title)

    # Adjust y-axis limits for better visualization (add some padding)
    if mse_values:
        max_mse = max(mse_values)
        ax.set_ylim(0, max_mse * 1.15) # Add 15% padding above the max bar
    else:
        ax.set_ylim(0, 1) # Default if no data


    plt.xticks(rotation=45, ha='right') # Rotate x-labels if they overlap
    plt.tight_layout() # Adjust layout

    if save_bool:
        now = datetime.datetime.now().strftime("%Y%m%d")
        save_path = Path(save_path)
        save_file = save_path / f"{save_prefix}_mse_{now}.{save_format}"
        
        try:
            plt.savefig(save_file, bbox_inches='tight', dpi=300, format=save_format)
            print(f"Plot saved to: {save_file}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.close(fig)
    