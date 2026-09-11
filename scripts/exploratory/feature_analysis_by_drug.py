import polars as pl
import pandas as pd # Pandas is needed for easier Seaborn integration and median calculation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# Clustermap uses hierarchical clustering from scipy
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist

def analyze_features_by_drug(data_path: str, output_dir: str = "figures/compound_analysis"):
    """Analyzes scaled delta features grouped by Compound.

    Generates swarm plots, ranked median bar plots, a heatmap of medians,
    and a clustermap (heatmap with hierarchical clustering) for each
    scaled delta feature to show trends across different compounds.

    Args:
        data_path: Path to the data file with delta and scaled features.
        output_dir: Directory to save the generated plots.
    """
    # Read the data
    try:
        df = pl.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    except Exception as e:
        print(f"Error reading data file: {e}")
        return

    # Ensure the 'Compound' column exists
    if 'Compound' not in df.columns:
        print("Error: 'Compound' column not found in the DataFrame.")
        compound_col = next((col for col in df.columns if col.lower() == 'compound'), None)
        if compound_col:
            print(f"Found potential column: '{compound_col}'. Renaming to 'Compound'.")
            df = df.rename({compound_col: 'Compound'})
        else:
            print("Please ensure a column named 'Compound' exists in your data.")
            return

    # Identify scaled delta feature columns
    scaled_delta_features = [
        col for col in df.columns
        if col.endswith('_scaled') and 'DeltaTp2_6' in col
    ]

    if not scaled_delta_features:
        print("Error: No scaled delta feature columns (e.g., 'DeltaTp2_6_..._scaled') found.")
        print("Available columns:", df.columns)
        return

    # Create the output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Convert Polars DataFrame to Pandas for easier plotting and aggregation
    df_pd = df.to_pandas()

    # Clean feature names for plot labels
    clean_feature_labels = {
        feat: feat.replace('_scaled', '').replace('DeltaTp2_6_', 'Change in ')
        for feat in scaled_delta_features
    }

    # --- Calculate Median Values per Compound ---
    print("Calculating median values per compound...")
    try:
        # Group by Compound and calculate the median for each scaled feature
        median_data = df_pd.groupby('Compound')[scaled_delta_features].median()
        # Rename columns for better readability in plots
        median_data_renamed = median_data.rename(columns=clean_feature_labels)
        print("Median calculation complete.")
    except Exception as e:
        print(f"Error calculating medians: {e}")
        return

    # --- Generate Plots for Each Scaled Delta Feature ---
    for feature in scaled_delta_features:
        print(f"Processing feature: {feature}...")

        # Clean feature name for titles/filenames
        clean_feature_name = feature.replace('_scaled', '').replace('DeltaTp2_6_', 'Change_')
        plot_title_feature_name = clean_feature_labels[feature] # Use pre-cleaned name

        # 1. Swarm Plot (Combined with Box Plot for clarity)
        plt.figure(figsize=(12, 8))
        sorted_compounds = sorted(df_pd['Compound'].unique())
        sns.boxplot(
            x='Compound', y=feature, data=df_pd,
            order=sorted_compounds,
            palette='viridis', showfliers=False
        )
        sns.stripplot(
            x='Compound', y=feature, data=df_pd,
            order=sorted_compounds,
            size=4, alpha=0.6, color='black', jitter=True
        )
        plt.title(f'Distribution of {plot_title_feature_name} by Compound')
        plt.ylabel('Scaled Value')
        plt.xlabel('Compound')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        swarm_plot_path = Path(output_dir) / f"{clean_feature_name}_swarm_by_compound.png"
        plt.savefig(swarm_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved swarm plot: {swarm_plot_path}")

        # 2. Rank-Ordered Bar Plot of Medians
        # Use the pre-calculated median_data for efficiency
        median_feature_data = median_data[[feature]].reset_index() # Select current feature
        median_feature_data_sorted = median_feature_data.sort_values(by=feature, ascending=False)

        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(
            x='Compound', y=feature, data=median_feature_data_sorted,
            palette='viridis', order=median_feature_data_sorted['Compound']
        )
        plt.title(f'Median {plot_title_feature_name} by Compound (Rank Ordered)')
        plt.ylabel('Median Scaled Value')
        plt.xlabel('Compound')
        plt.xticks(rotation=45, ha='right')

        for container in barplot.containers:
            barplot.bar_label(container, fmt='%.2f', fontsize=8, padding=3)

        plt.tight_layout()
        bar_plot_path = Path(output_dir) / f"{clean_feature_name}_median_rank_by_compound.png"
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved median bar plot: {bar_plot_path}")

    # --- Generate Aggregate Plots (Heatmap & Clustermap) ---

    if not median_data_renamed.empty:
        print("\nGenerating aggregate plots (Heatmap and Clustermap)...")

        # 3. Heatmap of Median Values
        plt.figure(figsize=(10, max(6, len(median_data_renamed.index) * 0.5))) # Adjust height based on number of compounds
        sns.heatmap(
            median_data_renamed,
            cmap='RdBu_r', # Red-Blue diverging colormap, good for +/- changes centered at 0
            center=0,      # Center the colormap at 0
            annot=True,    # Show median values on the heatmap
            fmt=".2f",     # Format annotations to 2 decimal places
            linewidths=.5, # Add lines between cells
            cbar_kws={'label': 'Median Scaled Change'} # Label for the color bar
        )
        plt.title('Median Scaled Feature Changes by Compound')
        plt.xlabel('Feature Change')
        plt.ylabel('Compound')
        plt.xticks(rotation=45, ha='right') # Rotate feature labels if needed
        plt.yticks(rotation=0) # Keep compound labels horizontal
        plt.tight_layout()
        heatmap_path = Path(output_dir) / "compound_median_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved heatmap: {heatmap_path}")

        # 4. Clustermap (Heatmap with Hierarchical Clustering)
        # Seaborn's clustermap automatically performs hierarchical clustering
        # Default uses 'average' linkage and 'euclidean' distance
        try:
            # Adjust figsize dynamically
            cluster_fig_height = max(8, len(median_data_renamed.index) * 0.6)
            cluster_fig_width = max(10, len(median_data_renamed.columns) * 1.2)

            clustergrid = sns.clustermap(
                median_data_renamed,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt=".2f",
                linewidths=.5,
                figsize=(cluster_fig_width, cluster_fig_height),
                cbar_kws={'label': 'Median Scaled Change'}
                # Can customize linkage method and metric, e.g.:
                # method='ward', metric='euclidean'
            )
            # Rotate labels for better readability if they overlap
            plt.setp(clustergrid.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
            plt.setp(clustergrid.ax_heatmap.get_yticklabels(), rotation=0)
            clustergrid.fig.suptitle('Hierarchical Clustering of Compounds by Median Feature Changes', y=1.02) # Add title above clustermap
            clustermap_path = Path(output_dir) / "compound_median_clustermap.png"
            # Use bbox_inches='tight' for clustermap as well
            plt.savefig(clustermap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved clustermap: {clustermap_path}")
        except Exception as e:
             print(f"  Error generating clustermap: {e}. Skipping clustermap.")
             # This might happen if there's only one compound or one feature after filtering.

    else:
        print("\nSkipping aggregate plots (Heatmap, Clustermap) because no median data was generated.")


    print(f"\nAnalysis complete. Plots saved in '{output_dir}'.")


def main():
    """Main function to run the feature analysis by drug."""
    input_data_path = "data/marm_data_wide_clustered_delta_scaled.csv"
    analyze_features_by_drug(input_data_path)


if __name__ == "__main__":
    main()
