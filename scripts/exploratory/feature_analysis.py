import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


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
    
    # Create figures directory if it doesn't exist
    Path("figures").mkdir(exist_ok=True)
    
    # Scale the delta features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[delta_features])
    
    # Create scaled features with new names
    scaled_features = [f"{feat}_scaled" for feat in delta_features]
    scaled_df = pl.DataFrame(scaled_data, schema=scaled_features)
    
    # Combine original and scaled data
    df = pl.concat([df, scaled_df], how='horizontal')
    
    # Create swarm plots with scaled data
    plt.figure(figsize=(15, 10))
    for i, (feat, scaled_feat) in enumerate(zip(delta_features, scaled_features), 1):
        plt.subplot(3, 2, i)
        # Use stripplot instead of swarmplot to avoid warnings
        sns.stripplot(x=df['cluster'], y=df[scaled_feat], size=3, alpha=0.5)
        plt.axhline(y=df[scaled_feat].median(), color='r', linestyle='--')
        plt.title(feat.replace('DeltaTp2_6_', 'change in '))
        plt.xticks(rotation=45)
        plt.ylabel('Scaled Value')
    
    plt.tight_layout()
    plt.savefig(
        "figures/feature_swarm_plots_scaled.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Create heatmap with scaled data
    # Calculate median values for each cluster
    cluster_medians = []
    for cluster in sorted(df['cluster'].unique()):
        cluster_mask = df['cluster'] == cluster
        cluster_data = df.filter(cluster_mask)
        medians = [cluster_data[scaled_feat].median() 
                  for scaled_feat in scaled_features]
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
    plt.title('Median Scaled Feature Changes by Cluster')
    plt.tight_layout()
    plt.savefig(
        "figures/feature_heatmap_scaled.png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    
    # Save the data with delta features and scaled values
    df.write_csv("data/marm_data_wide_clustered_delta_scaled.csv")


def main():
    """Main function to run the feature analysis."""
    create_feature_heatmap("data/marm_data_wide_clustered.csv")


if __name__ == "__main__":
    main()