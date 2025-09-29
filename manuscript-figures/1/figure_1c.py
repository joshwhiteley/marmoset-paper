import polars as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage

marmoset_data = pl.read_csv("data/marm_data_wide_clustered.csv")

clustergram_features = ["TP2_MeanSUV",
                        "TP2_MeanHU",
                        "TP2_HardVol",
                        "TP2_SoftVol",
                        "TP2_StandDevHU"
                        ]

# scale and center data
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

# move into a df
scaled_marmoset_data = marmoset_data.select(
    [
        pl.col("cluster"),
        *scale_expressions
    ]
)

# cluster medians for clustergram
medians_by_cluster = scaled_marmoset_data.group_by("cluster").agg(
    [pl.median(col).alias(col) for col in clustergram_features]
).sort("cluster")

# convert to pandas for plotting
medians_pd = (
    medians_by_cluster.to_pandas()
    .set_index('cluster')
    .transpose()
    .reset_index()
)

median_cluster_data_pd = medians_pd.rename(columns={'index': 'feature'})

median_cluster_data = pl.from_pandas(median_cluster_data_pd)

new_columns_names = {
    str(c): str(c) for c in median_cluster_data.columns if c != 'feature'
}
# clean up naming for supp output
median_cluster_data = median_cluster_data.rename(mapping=new_columns_names)

# save to supp/...
median_cluster_data.write_csv("supp/figure_1b_median_cluster_data.csv")

clusters = [str(i) for i in range(0, 16)] # assumes 1-16

# format for plotting
clustergram_input_data = median_cluster_data.select(
    [pl.col(str(c)) for c in range(16)]
).to_numpy()

# linkage matrices
row_linkage = linkage(clustergram_input_data,
                      method='average',
                      metric='cosine'
                      )

col_linkage = linkage(clustergram_input_data.transpose(),
                      method='average',
                      metric='cosine'
                      )

# clustergram
g = sns.clustermap(
    data=clustergram_input_data,
    row_linkage=row_linkage,
    col_linkage=col_linkage,
    cmap='YlOrRd',
    figsize=(12,8),
    col_cluster=True,
    row_cluster=False,
    yticklabels=clustergram_features
)

g.ax_heatmap.set_title('marmoset lcd cluster dendrogram', y=1.25)
g.ax_heatmap.set_ylabel('feature')
g.ax_heatmap.set_xlabel('cluster')
g.ax_heatmap.tick_params(axis='y', rotation=0)

g.savefig("manuscript-figures/1/figure_1c.svg", format='svg')
