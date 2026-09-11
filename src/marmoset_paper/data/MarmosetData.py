import os

import polars as pl


class MarmosetData:
    def __init__(self, filepath: str):
        file_extension = os.path.splitext(filepath)[1].lower()

        if file_extension == ".xlsx":
            self.data = pl.read_excel(filepath)
        elif file_extension == ".csv":
            self.data = pl.read_csv(filepath, infer_schema_length=10000)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        self.treatments = set(self.data["Compound"].unique())
        self.treatment_count = len(self.data["Compound"].drop_nulls().unique())
        self.marmosets = set(self.data["MarmID"].unique())
        self.marmoset_count = len(self.data["MarmID"].drop_nulls().unique())

    def _create_new_instance(self, filtered_data: pl.DataFrame) -> "MarmosetData":
        """Creates a new instance of Marmoset Data."""
        new_instance = MarmosetData.__new__(MarmosetData)
        new_instance.data = filtered_data
        new_instance.treatments = set(new_instance.data["Compound"].unique())
        new_instance.treatment_count = len(new_instance.data["Compound"].drop_nulls().unique())
        new_instance.marmosets = set(new_instance.data["MarmID"].unique())
        new_instance.marmoset_count = len(new_instance.data["MarmID"].drop_nulls().unique())
        return new_instance

    def get_treatment(self, treatment: str) -> "MarmosetData":
        """Get a MarmosetData instance for a specific treatment."""
        filtered_data = self.data[self.data["Compound"] == treatment]
        return self._create_new_instance(filtered_data)

    def get_lesion(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> "MarmosetData":
        """Get a MarmosetData instance for a specific lesion."""
        filtered_data = self.data[
            (self.data["MarmID"] == marmoset_id) & (self.data["Lesion"] == lesion_id)
        ]
        return self._create_new_instance(filtered_data)

    def get_lesion_data(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> pl.DataFrame:
        """Returns a dataframe of a specific lesion."""
        lesion_data = self.data[
            (self.data["MarmID"] == marmoset_id) & (self.data["Lesion"] == lesion_id)
        ]

        if not lesion_data.is_empty():
            return lesion_data
        else:
            raise ValueError("Lesion not found.")

    def get_severe_lesions(self) -> "MarmosetData":
        """Returns a MarmosetData instance with only severe lesions."""
        if "classif" in self.data.columns:
            filtered_data = self.data.filter(pl.col("classif").is_in(["hot"])).sort(
                ["Compound", "Lesion"]
            )
            return self._create_new_instance(filtered_data)
        else:
            raise ValueError("Lesion class not found.")

    def get_less_severe_lesions(self) -> "MarmosetData":
        """Returns a MarmosetData instance with only less severe lesions."""
        if "classif" in self.data.columns:
            filtered_data = self.data.filter(pl.col("classif").is_in(["cool"])).sort(
                ["Compound", "Lesion"]
            )
            return self._create_new_instance(filtered_data)
        else:
            raise ValueError("Lesion class not found.")
