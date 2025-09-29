from re import L
import polars as pl
import numpy as np
import scanpy.tools as tl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from anndata import AnnData
from scanpy.preprocessing import neighbors

from helpers.constants import PALETTE_16




CLUSTERING_FEATURES = [
    "TP2_MeanHU",
    "TP2_StandDevHU",
    "TP2_SoftVol",
    "TP2_HardVol",
    "TP2_MeanSUV"
]

def load_and_preprocess_data(data_path: str,
                             cluster_features: list[str]
                             ) -> tuple[pl.DataFrame, np.ndarray]:
    """ load and scale """
    data = pl.read_csv(data_path, infer_schema_length=10000)
    
    # extract features and scale
    X = data[cluster_features]
    transformer = StandardScaler().fit(X)
    X_scaled = transformer.transform(X)
    
    return data, X_scaled


def perform_clustering(X_scaled: np.ndarray) -> AnnData:
    # create AnnData object
    adata = AnnData(X_scaled)
    adata.obs_names = [f"lesion_{i}" for i in range(1, X_scaled.shape[0] + 1)]

    # dimensionality reduction
    tl.pca(adata)
    
    # clustering
    neighbors(adata, metric='cosine', n_pcs=len(CLUSTERING_FEATURES))
    tl.umap(adata)
    tl.leiden(adata, resolution=1)
    
    return adata


def create_merged_df(data: pl.DataFrame,
                     adata: AnnData) -> pl.DataFrame:
    """ create merged dataframe with original data, UMAP coordinates, and clusters """
    # extract umap coords + clusters
    umap_df = pl.DataFrame(adata.obsm['X_umap'])
    umap_df.columns = ['UMAP1', 'UMAP2']
    
    clusters_df = pl.DataFrame(adata.obs[['leiden']])
    clusters_df.columns = ['cluster']
    
    # merge all data
    merged_df = pl.concat([
        data, umap_df, clusters_df],
        how='horizontal'
    )
    
    return merged_df

def plot_umap_by_cluster(merged_df: pl.DataFrame):
    """ plot UMAP visualization colored by cluster """
    plt.figure(figsize=(10, 8))
    
    # numeric cluster labels
    cluster_values = merged_df['cluster'].cast(pl.Int64).to_numpy()
    
    # sort and remove duplicates
    unique_clusters = sorted(set(cluster_values))
    
    # build a mapping from cluster -> color tuple
    color_map = {
        cluster: PALETTE_16[i % len(PALETTE_16)]
        for i, cluster in enumerate(unique_clusters)
    }
    
    # map point's cluster to color
    point_colors = [color_map[c] for c in cluster_values]
    
    # scatter w/ explicit colors
    plt.scatter(
        merged_df['UMAP1'].to_numpy(),
        merged_df['UMAP2'].to_numpy(),
        c=point_colors,
        s=100,
        edgecolor='k',
        linewidth=0.2
    )
    
    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=color_map[cluster],
            label=str(cluster),
            markersize=10,
            markeredgecolor='k',
            markeredgewidth=0.5
        )
        for cluster in unique_clusters
    ]
    
    plt.legend(
        handles=legend_elements,
        title="cluster",
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        ncol=2
    )
    
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title("UMAP of lesions by cluster")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    
    plt.savefig("manuscript-figures/1/figure_1a.svg", format='svg')
    plt.show()

if __name__ == "__main__":
    data_path = "data/marm_data_wide.csv"
    
    data, X_scaled = load_and_preprocess_data(data_path, CLUSTERING_FEATURES)
    adata = perform_clustering(X_scaled)
    merged_df = create_merged_df(data, adata)
    plot_umap_by_cluster(merged_df)
