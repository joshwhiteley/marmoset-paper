import polars as pl
from helpers.PlottingFunctions import (
    plot_ground_truth_lesion_trajectory
)

PLOT_CONFIG = {
    "data_df": "data/marm_data_wide_clustered_classif.csv",
    "num_lesions_to_plot": 5,
    "specific_lesions": None,
    "timepoint_style": "week",
    "save_dir": "figures/ground_truth",
    "save_bool": True,
    "random_state": 5674,
    "output_format": "svg"
}

marm_df = pl.read_csv(PLOT_CONFIG["data_df"])
compounds = marm_df["Compound"].unique().to_list()

for c in compounds:

    print(f"Plotting ground truth for compound: {c}")
    plot_ground_truth_lesion_trajectory(
        data_df=marm_df,
        compound=c,
        num_lesions_to_plot=PLOT_CONFIG["num_lesions_to_plot"],
        specific_lesions=PLOT_CONFIG["specific_lesions"],
        timepoint_style=PLOT_CONFIG["timepoint_style"],
        save_dir=PLOT_CONFIG["save_dir"],
        save_bool=PLOT_CONFIG["save_bool"],
        random_state=PLOT_CONFIG["random_state"],
        output_format=PLOT_CONFIG["output_format"]
    )

if PLOT_CONFIG["save_bool"]:
    print(f'Ground truth plots generated in: {PLOT_CONFIG["save_dir"]}')


# HRZE figure
plot_ground_truth_lesion_trajectory(
    data_df=marm_df,
    compound="EMB+INH+PZA+RIF",
    num_lesions_to_plot=3,
    specific_lesions=[6, 8, 3],
    timepoint_style="week",
    save_dir="figures/predictions/HRZE-fig4",
    save_bool=True,
    random_state=5674,
    output_format="svg"
)