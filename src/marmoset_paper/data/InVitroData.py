import os
import re

import polars as pl


class DiamondData:
    def __init__(self, filepath: str):
        file_extension = os.path.splitext(filepath)[1].lower()

        if file_extension == ".xlsx":
            self.data = pl.read_excel(filepath)
        elif file_extension == ".csv":
            self.data = pl.read_csv(filepath, infer_schema_length=10000)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        self.treatments = set(self.data["Drug"].unique())
        self.treatment_count = len(self.data["Drug"].drop_nulls().unique())

    def _create_new_instance(self, filtered_data: pl.DataFrame) -> "DiamondData":
        """Create a new instance of DiamondData from a filtered DataFrame."""
        new_instance = DiamondData.__new__(DiamondData)
        new_instance.data = filtered_data
        new_instance.treatments = set(filtered_data["Drug"].unique())
        new_instance.treatment_count = len(filtered_data["Drug"].drop_nulls().unique())
        return new_instance

    def filter_by_condition(self, condition: str) -> "DiamondData":
        pattern = "|".join([f"_{condition}_"])

        filtered_columns = [
            col for col in self.data.columns if re.search(pattern, col) or col == "Drug"
        ]

        return self._create_new_instance(self.data.select(filtered_columns))

    def filter_by_conditions(self, conditions: list) -> "DiamondData":
        pattern = "|".join([f"_{condition}_" for condition in conditions])

        filtered_columns = [
            col for col in self.data.columns if re.search(pattern, col) or col == "Drug"
        ]

        return self._create_new_instance(self.data.select(filtered_columns))

    def filter_by_treatment(self, treatment: str) -> "DiamondData":
        filtered_data = self.data.filter(pl.col("Drug") == treatment)
        return self._create_new_instance(filtered_data)

    def filter_by_treatments(self, treatments: list) -> "DiamondData":
        filtered_data = self.data.filter(pl.col("Drug").is_in(treatments))
        return self._create_new_instance(filtered_data)

    def drop_na(self) -> "DiamondData":
        """Keep rows with complete measurements outside the Drug column."""
        non_drug_columns = [col for col in self.data.columns if col != "Drug"]

        filtered_data = self.data.drop_nulls(subset=non_drug_columns)

        return self._create_new_instance(filtered_data)

    def drop_na_cols(self) -> "DiamondData":
        """Remove columns where all values are NA."""
        columns = [c for c in self.data.columns if self.data[c].null_count() < self.data.height]
        return self._create_new_instance(self.data.select(columns))

    def get_data(self) -> pl.DataFrame:
        """Return the data as a DataFrame."""
        meta_feats = ["Drug", "NumbDrugs", "Dataset", "Score"]
        return self.data.drop([c for c in meta_feats if c in self.data.columns])

    def drop_extra_cols(self) -> "DiamondData":
        """
        Drops columns in the dataset that contain patterns like 'Stdev' or 'Averaged' in their names.
        """
        # Create a copy of the data to perform the drop
        dropped_data = self.data.clone()

        # Define a regex pattern to match columns containing 'Stdev' or 'Averaged'
        pattern = re.compile(r"Stdev|Averaged", re.IGNORECASE)

        # Identify columns to drop based on the pattern
        columns_to_drop = [col for col in dropped_data.columns if pattern.search(col)]

        # Drop the identified columns
        dropped_data = dropped_data.drop(columns_to_drop)

        return self._create_new_instance(dropped_data)
