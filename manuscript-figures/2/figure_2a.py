import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches

from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from data.InVitroData import DiamondData
from data.MarmosetData import MarmosetData
from helpers.constants import COMPOUND_NAMES, FXC50_COLS

# Load in vitro data
diamond_data = pl.read_csv("data/in_vitro_modeling.csv")

# Filter to combos only
compounds = list(COMPOUND_NAMES.keys())
diamond_data = diamond_data.filter(
    pl.col("Drug").is_in(compounds)
)

# Identify FxC50 related columns
all_cols = []
for group in FXC50_COLS.values():
    all_cols.extend(group.keys())

# Filter to only FxC50 + drug, add colloquial name to df
FxC50_df = (
    diamond_data
    .select(["Drug"] + all_cols) # change df to Drug+FXC50 cols only
    .with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in all_cols])
    .with_columns(pl.col("Drug").replace(COMPOUND_NAMES).alias("abbrev"))
)

# Reorganize table to be in the format of:
# | Drug    | abbrev | feature  | value | group      | condition |
# | BDQ+DEL | BD     | butyrate | -0.39 | equipotent | butyrate  |
# ...

FxC50_melted = pl.DataFrame()
for group_name, cols in FXC50_COLS.items():
    temp_df = (
        FxC50_df
        .select(["Drug", "abbrev"] + list(cols.keys()))
        .melt(
            id_vars=["Drug", "abbrev"],
            value_vars=list(cols.keys()),
            variable_name="feature",
            value_name="value"
        )
        .with_columns(
            pl.lit(group_name).alias("group"),
            pl.col("feature").replace_strict(cols).alias("condition")
        )
    )
    FxC50_melted = pl.concat([FxC50_melted, temp_df])

# Pivot the data for heatmap generation
# Then convert to pandas
FxC50_heatmap_data = (
    FxC50_melted
    .pivot(
        values="value",
        index="abbrev",
        on=["group", "condition"],
        aggregate_function="first"  # shouldn't happen, but just in case
    ).to_pandas()
    .sort_values(by='{"cell","7H"}', ascending=True)
)

# Flatten multi index column ('equipotent', 'butyrate') -> 'butyrate'
FxC50_heatmap_data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in FxC50_heatmap_data.columns]

# Remove abbrev column from heatmap data for plotting
heatmap_df = FxC50_heatmap_data.drop(columns=['abbrev']).values

# Labels for plot
row_labels = FxC50_heatmap_data['abbrev'].tolist()
col_labels = FxC50_heatmap_data.drop(columns=['abbrev']).columns.tolist()


# Plotting parameters
vmin, vmax = -2.5, 2.5
cmap = plt.cm.get_cmap("RdBu_r")  # Red-Blue reversed, white center
nan_color = "#D3D3D3"  # theme grey

# Create three separate heatmaps using the working logic
group_labels_map = {
    "equipotent": "equipotent",
    "cas": "cas PK",
    "cell": "cell PK"
}

# Get column indices for each group
columns_start_indices = {}
current_idx = 0
for group in ['equipotent', 'cas', 'cell']:
    group_len = len(FXC50_COLS[group])
    columns_start_indices[group] = current_idx
    current_idx += group_len

# Create each heatmap
for group_key, group_display_name in group_labels_map.items():
    # Get the column slice for this group
    start_col = columns_start_indices[group_key]
    end_col = start_col + len(FXC50_COLS[group_key])

    # Slice the data for this group
    group_heatmap_data = heatmap_df[:, start_col:end_col]
    group_col_labels = col_labels[start_col:end_col]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))

    # Use the same working logic from before
    nan_mask = np.isnan(group_heatmap_data)

    # Data prep
    clamped_data = np.where(group_heatmap_data < vmin, vmin*1.01, group_heatmap_data)
    clamped_data = np.where(group_heatmap_data > vmax, vmax*1.01, clamped_data)
    clamped_data[nan_mask] = np.nan

    # Annotation array
    annot = np.full(group_heatmap_data.shape, "", dtype=object)
    annot[(group_heatmap_data < vmin) | (group_heatmap_data > vmax)] = "*"
    annot[nan_mask] = "?"

    # Create heatmap
    heatmap_obj = sns.heatmap(
        clamped_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt="s",
        annot_kws={"fontsize": 6, "fontweight": "bold"},
        cbar_kws={"label": "FxC50"},
        linewidths=0.5,
        linecolor='grey',
        xticklabels=group_col_labels,
        yticklabels=False,
        ax=ax
    )

    # Add grey background for NaN cells
    for i in range(group_heatmap_data.shape[0]):
        for j in range(group_heatmap_data.shape[1]):
            if nan_mask[i, j]:
                rect = plt.Rectangle((j, i), 1, 1,
                                   facecolor=nan_color,
                                   edgecolor='white',
                                   linewidth=0.5,
                                   zorder=0)
                ax.add_patch(rect)

    # Add y-axis labels manually
    for idx, label in enumerate(row_labels):
        ax.text(
            -0.6,
            idx + 0.5,
            label,
            ha='right',
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    # Formatting
    plt.xticks(rotation=45, ha='right')
    plt.title(f"FxC50 values - {group_display_name}", pad=20)
    plt.tight_layout()

    # Save
    plt.savefig(f"manuscript-figures/2/figure_2a_{group_key}.svg",
                format="svg",
                transparent=True)
    plt.show()