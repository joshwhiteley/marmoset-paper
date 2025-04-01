import polars as pl
import os


class MarmosetData:
    def __init__(self, filepath: str):
        file_extension = os.path.splitext(filepath)[1].lower()

        if file_extension == '.xlsx':
            self.data = pl.read_excel(filepath)
        elif file_extension == '.csv':
            self.data = pl.read_csv(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        self.treatments = set(self.data['Compound'].unique())
        self.treatment_count = len(self.data['Compound'].drop_nulls().unique())
        self.marmosets = set(self.data['MarmID'].unique())
        self.marmoset_count = len(self.data['MarmID'].drop_nulls().unique())
        self.timepoints = (
            set(self.data['Timepoint'].unique())
            if 'Timepoint' in self.data else None
        )
        self.lesion_count = (
            len(self.data['LesionID'].drop_nulls().unique())
            if 'Lesion' in self.data else 
            len(self.data)
        )
        self.lesion_types = (
            set(self.data['LesionType'].unique())
            if 'LesionType' in self.data else None
        )
        self.final_lesion_types = (
            set(self.data['FinalLesionType'].unique())
            if 'FinalLesionType' in self.data else None
        )

    def _create_new_instance(self, filtered_data: pl.DataFrame) -> 'MarmosetData':
        """Creates a new instance of Marmoset Data."""
        new_instance = MarmosetData.__new__(MarmosetData)
        new_instance.data = filtered_data
        new_instance.treatments = set(new_instance.data['Compound'].unique())
        new_instance.treatment_count = len(new_instance.data['Compound'].drop_nulls().unique())
        new_instance.marmosets = set(new_instance.data['MarmID'].unique())
        new_instance.marmoset_count = len(new_instance.data['MarmID'].drop_nulls().unique())
        new_instance.timepoints = (
            set(new_instance.data['Timepoint'].unique())
            if 'Timepoint' in new_instance.data else None
        )
        return new_instance

    def get_treatment(self, treatment: str) -> 'MarmosetData':
        """Get a MarmosetData instance for a specific treatment."""
        filtered_data = self.data[self.data['Compound'] == treatment]
        return self._create_new_instance(filtered_data)

    def get_timepoint(self, timepoint: int) -> 'MarmosetData':
        """Get a MarmosetData instance for a specific timepoint."""
        filtered_data = self.data[self.data['Timepoint'] == timepoint]
        return self._create_new_instance(filtered_data)

    def get_lesion(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> 'MarmosetData':
        """Get a MarmosetData instance for a specific lesion."""
        filtered_data = self.data[
            (self.data['MarmID'] == marmoset_id) &
            (self.data['Timepoint'] == timepoint_id) &
            (self.data['Lesion'] == lesion_id)
        ]
        return self._create_new_instance(filtered_data)

    def get_lesion_data(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> pl.DataFrame:
        """Returns a dataframe of a specific lesion."""
        lesion_data = self.data[
            (self.data['MarmID'] == marmoset_id) &
            (self.data['Timepoint'] == timepoint_id) &
            (self.data['Lesion'] == lesion_id)
        ]

        if not lesion_data.is_empty():
            return lesion_data
        else:
            raise ValueError("Lesion not found.")

    def get_lesion_type(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> str:
        lesion_data = self.get_lesion_data(marmoset_id, timepoint_id, lesion_id)

        if 'LesionType' in lesion_data:
            return lesion_data['LesionType'].drop_nulls().unique()[0]
        else:
            raise ValueError("Lesion type not found.")

    def get_consolidate_status(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> bool:
        """Returns true if the lesion is consolidated, false otherwise."""
        lesion_data = self.get_lesion_data(marmoset_id, timepoint_id, lesion_id)

        if 'Consolidate' in lesion_data:
            return lesion_data['Consolidate'][0]
        else:
            raise ValueError("Consolidate status not found.")

    def get_contour(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> str:
        """Returns the contour of a specific lesion."""
        lesion_data = self.get_lesion_data(marmoset_id, timepoint_id, lesion_id)

        if 'Contour' in lesion_data:
            return lesion_data['Contour'][0]
        else:
            raise ValueError("Contour not found.")

    def get_scanner(self, marmoset_id: str, timepoint_id: int, lesion_id: int) -> str:
        """Returns the scanner used for a specific lesion."""
        lesion_data = self.get_lesion_data(marmoset_id, timepoint_id, lesion_id)

        if 'Scanner' in lesion_data:
            return lesion_data['Scanner'][0]
        else:
            raise ValueError("Scanner not found.")

    def get_severe_lesions(self) -> 'MarmosetData':
        """Returns a MarmosetData instance with only severe lesions."""
        if 'lesion_class' in self.data.columns:
            filtered_data = self.data[self.data['lesion_class'] == 'severe']
            return self._create_new_instance(filtered_data)
        else: 
            raise ValueError("Lesion class not found.")

    def get_less_severe_lesions(self) -> 'MarmosetData':
        """Returns a MarmosetData instance with only less severe lesions."""
        if 'lesion_class' in self.data.columns:
            filtered_data = self.data[self.data['lesion_class'] == 'less']
            return self._create_new_instance(filtered_data)
        else:
            raise ValueError("Lesion class not found.")


if __name__ == '__main__':
    lesion_data = MarmosetData("./data/Marm-TP2-Clustered.xlsx")
    print(f"Total lesion count: {lesion_data.lesion_count}")
    print(f"Total treatment count: {lesion_data.treatment_count}")
    print(f"Total marmoset count: {lesion_data.marmoset_count}")
    
    less = lesion_data.get_less_severe_lesions()
    more = lesion_data.get_severe_lesions()

    print(f"Less severe lesion count: {less.lesion_count}")
    print(f"More severe lesion count: {more.lesion_count}")