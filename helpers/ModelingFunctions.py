"""Helper functions for data modeling and preprocessing.

This module provides functions for handling compound-specific data operations,
including imputation and train-test splitting while maintaining compound integrity.
"""

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def impute_within_compound(
    df: pl.DataFrame,
    columns: list[str],
    grouping_column: str = "Compound"
) -> pl.DataFrame:
    """Impute missing values within each compound group using median imputation.

    Args:
        df: Input DataFrame containing the data to be imputed
        columns: List of column names to perform imputation on
        grouping_column: Name of the column used for grouping
            (default: "Compound")

    Returns:
        DataFrame with missing values imputed using compound-specific medians
    """
    for col in columns:
        # median for each compound group
        medians = (
            df.group_by(grouping_column)
            .agg(pl.col(col).median().alias('median'))
        )

        df = df.join(
            other=medians,
            on=grouping_column,
            how='left'
        )

        # replace null w/ compound-specific median
        df = df.with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.col('median'))
            .otherwise(pl.col(col))
            .alias(col)
        ).drop('median')

    return df


def split_by_compound(
    df: pl.DataFrame,
    grouping_column: str = "Compound",
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split data into train and test sets while maintaining compound integrity.

    This function ensures that each compound's data is split proportionally
    between train and test sets, preventing data leakage.

    Args:
        df: Input DataFrame to split
        grouping_column: Column used for grouping compounds
            (default: "Compound")
        test_size: Proportion of data to include in test set
            (default: 0.2)
        random_state: Random seed for reproducibility
            (default: 42)

    Returns:
        Tuple containing (train_data, test_data) DataFrames
    """
    np.random.seed(random_state)
    train_data = pl.DataFrame()
    test_data = pl.DataFrame()

    compounds = df.select(grouping_column).unique().to_series().to_list()

    for compound in compounds:

        compound_data = df.filter(pl.col(grouping_column) == compound)

        # calculate # of test samples
        n_test = max(1, int(len(compound_data) * test_size))

        # random split
        all_indices = np.arange(len(compound_data))
        test_indices = np.random.choice(
            all_indices,
            size=n_test,
            replace=False
        )

        test_mask = np.zeros(len(compound_data), dtype=bool)
        test_mask[test_indices] = True

        compound_test = compound_data.filter(test_mask)
        compound_train = compound_data.filter(~test_mask)

        train_data = pl.concat([train_data, compound_train])
        test_data = pl.concat([test_data, compound_test])

    return train_data, test_data


def get_compound_abbreviation(compound: str) -> str:
    """Get abbreviated name for a compound combination.
    
    Args:
        compound: Full compound name (e.g., "MOX+RIF")
        
    Returns:
        Abbreviated name (e.g., "MR")
    """
    abbrev_map = {
        "MOX": "M",
        "RIF": "R",
        "EMB": "E",
        "PZA": "Z",
        "BDQ": "B",
        "PRE": "Pa",
        "LIN": "L",
        "DEL": "D",
        "INH": "H"
    }
    
    # Split compound into individual drugs
    drugs = compound.split("+")
    
    # Get abbreviations and sort them
    abbrevs = [abbrev_map[drug] for drug in drugs]
    abbrevs.sort()
    
    # Join abbreviations
    return "".join(abbrevs)


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
    ax1.set_xticks(x)
    ax1.set_xticklabels(timepoints)
    ax1.grid(True, alpha=0.3)
    
    # Customize MeanSUV plot
    ax2.set_title('MeanSUV')
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('MeanSUV')
    ax2.set_xticks(x)
    ax2.set_xticklabels(timepoints)
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
        save_file = save_path / f"{abbrev}_lesion_progressions.png"
        plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory
