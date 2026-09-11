"""Feature preparation and the manuscript's within-treatment lesion split."""

import numpy as np
import polars as pl


def impute_within_compound(
    df: pl.DataFrame, columns: list[str], grouping_column: str = "Compound"
) -> pl.DataFrame:
    """Fill nulls with each treatment's median without changing row order.

    Entirely missing treatment-feature groups remain null. The historical analysis
    applies this before splitting; it is not a train-fitted imputer.
    """
    return df.with_columns(
        pl.col(column).fill_null(pl.col(column).median().over(grouping_column))
        for column in columns
    )


def split_by_compound(
    df: pl.DataFrame,
    grouping_column: str = "Compound",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sample test lesions within each treatment, not whole treatments or animals.

    Treatments are sorted before consuming the random stream. Historical code used
    unordered Polars groups, so its exact split cannot be recovered from a seed.
    Single-row treatments go to the test set, as in the original implementation.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if df.is_empty() or df[grouping_column].null_count():
        raise ValueError("The split requires rows with non-null treatment identifiers")
    rng = np.random.RandomState(random_state)
    train, test = [], []
    for compound in df[grouping_column].unique().sort():
        group = df.filter(pl.col(grouping_column) == compound)
        n_test = max(1, int(len(group) * test_size))
        mask = np.zeros(len(group), dtype=bool)
        mask[rng.choice(len(group), size=n_test, replace=False)] = True
        train.append(group.filter(~mask))
        test.append(group.filter(mask))
    return pl.concat(train), pl.concat(test)


def get_compound_abbreviation(compound: str) -> str:
    """Return the historical alphabetical drug abbreviation."""
    abbreviations = {
        "MOX": "M",
        "RIF": "R",
        "EMB": "E",
        "PZA": "Z",
        "BDQ": "B",
        "PRE": "Pa",
        "LIN": "L",
        "DEL": "D",
        "INH": "H",
        "QBS": "Q",
    }
    return "".join(sorted(abbreviations[drug] for drug in compound.split("+")))
