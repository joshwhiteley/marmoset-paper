import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from helpers.constants import LESS_SEVERE_COLOR_TUPLE, SEVERE_COLOR_TUPLE


def plot_umap_by_classification(df: pl.DataFrame):
    """
    Plot UMAP visualization colored by cool/hot classification.
    """
    plt.figure(figsize=(10, 8))
    
    # color array based on classification
    colors = np.array([
        LESS_SEVERE_COLOR_TUPLE if c == 'cool'
        else SEVERE_COLOR_TUPLE for c in df['classif'].to_numpy()
    ])
    
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
            markerfacecolor=LESS_SEVERE_COLOR_TUPLE,
            label='cool',
            markersize=10
        ),
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=SEVERE_COLOR_TUPLE,
            label='hot',
            markersize=10
        )
    ]
    
    plt.legend(
        handles=legend_elements,
        title='classification',
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP of Lesions by Classification')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig("manuscript-figures/1/figure_1b.svg", format='svg')
    plt.show()


if __name__ == "__main__":
    marmoset_data = pl.read_csv("data/marm_data_wide_clustered_classif.csv", infer_schema_length=10000)
    plot_umap_by_classification(marmoset_data)