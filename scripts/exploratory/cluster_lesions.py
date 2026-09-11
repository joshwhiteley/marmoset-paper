import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from anndata import AnnData
from scanpy.preprocessing import neighbors
import scanpy.tools as tl
import seaborn as sns

from marmoset_paper.helpers.constants import PALETTE_16

"""
Script to perform clustering analysis on marmoset lesion data using UMAP and Leiden
clustering. Produces visualizations and classifies lesions as 'cool' or 'hot' based
on cluster membership.
"""


# Define clustering features and parameters
CLUSTERING_FEATURES = [
    "TP2_MeanHU",
    "TP2_StandDevHU",
    "TP2_SoftVol",
    "TP2_HardVol",
    "TP2_MeanSUV"
]


def load_and_preprocess_data(data: pl.DataFrame,
                             clustering_features: list[str]
                             ) -> tuple[pl.DataFrame, np.ndarray]:
    """Load and preprocess the marmoset data."""
    # Load the data
    marm_data = pl.read_csv(data, infer_schema_length=10000)

    # Extract features and scale them
    X = marm_data[clustering_features]
    transformer = StandardScaler().fit(X)
    scaled_X = transformer.transform(X)

    return marm_data, scaled_X


def perform_clustering(scaled_X: np.ndarray) -> AnnData:
    """Perform dimensionality reduction and clustering on the scaled data."""
    # Create AnnData object and perform clustering
    adata = AnnData(scaled_X)
    adata.obs_names = [f'lesion_{i}' for i in range(1, scaled_X.shape[0] + 1)]

    # Perform dimensionality reduction and clustering
    tl.pca(adata)
    neighbors(adata, metric='cosine', n_pcs=len(CLUSTERING_FEATURES))
    tl.umap(adata)
    tl.leiden(adata, resolution=1)

    return adata


def create_merged_dataframe(marm_data: pl.DataFrame,
                            adata: AnnData) -> pl.DataFrame:
    """Create a merged dataframe with original data, UMAP coordinates, and clusters."""
    # Extract UMAP coordinates and clusters
    umap_df = pl.DataFrame(adata.obsm['X_umap'])
    umap_df.columns = ['UMAP1', 'UMAP2']
    clusters_df = pl.DataFrame(adata.obs[['leiden']])
    clusters_df.columns = ['cluster']

    # Merge all data
    merged_df = pl.concat([marm_data, umap_df, clusters_df], how='horizontal')

    return merged_df

def plot_umap_by_cluster(merged_df: pl.DataFrame):
    """Plot UMAP visualization colored by cluster."""
    plt.figure(figsize=(10, 8))
    
    # get numeric cluster labels
    cluster_values = merged_df['cluster'].cast(pl.Int64).to_numpy()
    
    # sort & dedupe
    unique_clusters = sorted(set(cluster_values))
    
    # build a mapping from cluster → color tuple
    color_map = {
        cluster: PALETTE_16[i % len(PALETTE_16)]
        for i, cluster in enumerate(unique_clusters)
    }
    
    # map each point’s cluster to its color
    point_colors = [color_map[c] for c in cluster_values]
    
    # scatter with explicit colors
    plt.scatter(
        merged_df['UMAP1'].to_numpy(),
        merged_df['UMAP2'].to_numpy(),
        c=point_colors,
        s=100,
        edgecolor='k',      # optional: give points a thin black border
        linewidth=0.2
    )
    
    # legend handles using the same mapping
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
        title="Cluster",
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        ncol=2
    )
    
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP of Lesions by Cluster')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("figures/marm_lesion_UMAP_cluster.png", dpi=300, bbox_inches='tight')
    plt.savefig("figures/marm_lesion_UMAP_cluster.svg", format='svg')
    plt.savefig("figures/marm_lesion_UMAP_cluster.eps", format='eps')
    plt.show()

def create_feature_heatmap(data_path: str):
    """Create heatmap and swarm plots of features by cluster.
    
    Args:
        data_path: Path to the clustered data file
    """
    # Read the data
    df = pl.read_csv(data_path)
    
    # Define features of interest
    features = [
        "TP2_MeanHU",
        "TP2_StandDevHU",
        "TP2_SoftVol",
        "TP2_HardVol",
        "TP2_MeanSUV"
    ]
    
    # Calculate DeltaTp2_6 features
    delta_features = []
    for feat in features:
        base_feat = feat.split('_', 1)[1]  # Get feature name without TP2_
        tp6_feat = f"TP6_{base_feat}"
        delta_feat = f"DeltaTp2_6_{base_feat}"
        df = df.with_columns(
            (pl.col(tp6_feat) - pl.col(feat)).alias(delta_feat)
        )
        delta_features.append(delta_feat)
    
    # Create swarm plots
    plt.figure(figsize=(15, 10))
    for i, feat in enumerate(delta_features, 1):
        plt.subplot(3, 2, i)
        sns.swarmplot(x=df['cluster'], y=df[feat], size=3)
        plt.axhline(y=df[feat].median(), color='r', linestyle='--')
        plt.title(feat.replace('DeltaTp2_6_', 'change in '))
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(
        "figures/feature_swarm_plots.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Create heatmap
    # Calculate median values for each cluster
    cluster_medians = []
    for cluster in sorted(df['cluster'].unique()):
        cluster_data = df.filter(pl.col('cluster') == cluster)
        medians = [cluster_data[feat].median() for feat in delta_features]
        cluster_medians.append(medians)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        np.array(cluster_medians).T,
        cmap='RdBu_r',
        center=0,
        xticklabels=sorted(df['cluster'].unique()),
        yticklabels=[f.replace('DeltaTp2_6_', 'change in ')
                    for f in delta_features]
    )
    plt.title('Median Feature Changes by Cluster')
    plt.tight_layout()
    plt.savefig(
        "figures/feature_heatmap.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Save the data with delta features
    df.write_csv("data/marm_data_wide_clustered_delta.csv")


def main():
    """Main function to run the analysis."""
    # Load and preprocess data
    marm_data, scaled_X = load_and_preprocess_data(data="data/marm_data_wide.csv",
                                                   clustering_features=CLUSTERING_FEATURES)

    # Perform clustering
    adata = perform_clustering(scaled_X)

    # Create merged dataframe with classifications
    merged_df = create_merged_dataframe(marm_data, adata)

    # Generate visualizations
    plot_umap_by_cluster(merged_df)

    # Save results
    #merged_df.write_csv("data/marm_data_wide_clustered.csv")

    # Create feature heatmap
    create_feature_heatmap("data/marm_data_wide_clustered.csv")


if __name__ == "__main__":
    main()
