import polars as pl
import dash_bio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

marmoset_data = pl.read_csv("data/marm_data_wide_clustered.csv")

clustergram_features = ["TP2_MeanSUV",
                        "TP2_MeanHU",
                        "TP2_HardVol",
                        "TP2_SoftVol",
                        "TP2_StandDevHU"]

# --- Scaling and Centering Data ---
scale_expressions = []
for col_name in clustergram_features:
    mean_val = marmoset_data[col_name].mean()
    std_dev_val = marmoset_data[col_name].std()
    if std_dev_val is not None and std_dev_val > 0: 
        scale_expressions.append(
            ((pl.col(col_name) - mean_val) / std_dev_val).alias(col_name)
        )
    else:
        scale_expressions.append((pl.lit(0.0)).alias(col_name))

scaled_marmoset_data = marmoset_data.select([
    pl.col("cluster"),  
    *scale_expressions
])

medians_by_cluster = scaled_marmoset_data.group_by("cluster").agg(
    [pl.median(col).alias(col) for col in clustergram_features]
).sort("cluster")

medians_pd = (
    medians_by_cluster.to_pandas()
    .set_index('cluster')
    .transpose()
    .reset_index()
)
median_cluster_data_pd = medians_pd.rename(columns={'index': 'feature'})
median_cluster_data = pl.from_pandas(median_cluster_data_pd)


new_column_names = {
    str(c): str(c) for c in median_cluster_data.columns if c != 'feature'
}
median_cluster_data = median_cluster_data.rename(mapping=new_column_names)

# print("Median Scaled Cluster Data:")
# print(median_cluster_data)


clusters = [str(i) for i in range(0, 16)]  # assumes clusters 1-16 exist

clustergram_input_data = median_cluster_data.select(
    [pl.col(str(c)) for c in range(16)]  
).to_numpy()

clustergram = dash_bio.Clustergram(
     data=clustergram_input_data, 
     column_labels=clusters, 
     row_labels=clustergram_features,  
     width=1500,
     link_method="average",
     row_dist="cosine",
     col_dist="cosine",
     color_map="ylorrd",
     plot_bg_color="white",
     paper_bg_color="white",
)

clustergram.show()
clustergram.write_image("marmoset_LCD_clustergram.png")

# the clustergram indicates that there are two main clusters
# at the highest level of the tree.
# these two groupings functionally split the into two groups:
# severe, and less-severe
# according to the clustergram, the clusters that are classed as
# severe: [2,15,11,12,4,14,0,5]
# less severe: [1,13,10,6,9,7,3,8]
# we can assign these clusters to the actual dataframe accordingly
# and save that file accordingly for future analysis

# Add classification based on clustergram analysis
COOL_CLUSTERS = [1, 13, 10, 6, 9, 7, 3, 8]

# Add classification column
classification = (
    pl.when(pl.col('cluster').cast(pl.Int64).is_in(COOL_CLUSTERS))
    .then(pl.lit('cool'))
    .otherwise(pl.lit('hot'))
)
marmoset_data = marmoset_data.with_columns(classification.alias('classif'))

# Save the updated dataframe
marmoset_data.write_csv("data/marm_data_wide_clustered_classif.csv")


def plot_umap_by_classification(df: pl.DataFrame):
    """Plot UMAP visualization colored by cool/hot classification."""
    plt.figure(figsize=(10, 8))
    
    # Create color array based on classification
    colors = np.array(['lightblue' if c == 'cool' else 'red' 
                      for c in df['classif'].to_numpy()])
    
    # Create scatter plot
    plt.scatter(
        df['UMAP1'].to_numpy(),
        df['UMAP2'].to_numpy(),
        c=colors,
        s=100
    )

    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor='lightblue',
            label='Cool',
            markersize=10
        ),
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor='red',
            label='Hot',
            markersize=10
        )
    ]

    plt.legend(
        handles=legend_elements,
        title="Classification",
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )

    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP of Lesions by Classification')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("figures/marm_lesion_UMAP_classif.png", dpi=300, bbox_inches='tight')
    plt.show()


plot_umap_by_classification(marmoset_data)