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
