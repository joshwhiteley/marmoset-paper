import polars as pl
import numpy as np
from scipy.stats import spearmanr, pearsonr
from itertools import combinations
from typing import List, Dict, Tuple

# Load in vitro data
diamond_data = pl.read_csv("data/in_vitro_modeling.csv")

# Define feature groups for correlation analysis
# Easily expandable - just add more groups or modify existing ones
CORRELATION_GROUPS = {
    "5N_comparison": ["FxC50_5Nequip", "FxC50_5NcasPK", "FxC50_5NcellPK"],
    "7H_comparison": ["FxC50_7Hequip", "FxC50_7HcasPK", "FxC50_7HcellPK"],
    "7N_comparison": ["FxC50_7Nequip", "FxC50_7NcasPK", "FxC50_7NcellPK"],
    "butyrate_constant": ["FxC50_butyrate_Constant", "FxC50_butyrateCasPK_Constant", "FxC50_butyrateCellPK_Constant"],
    "butyrate_terminal": ["FxC50_butyrate_Terminal", "FxC50_butyrateCasPK_Terminal", "FxC50_butyrateCellPK_Terminal"],
    "cholesterol_constant": ["FxC50_cholesterol_Constant", "FxC50_cholesterolCasPK_Constant", "FxC50_cholesterolCellPK_Constant"],
    "cholesterol_terminal": ["FxC50_cholesterol_Terminal", "FxC50_cholesterolCasPK_Terminal", "FxC50_cholesterolCellPK_Terminal"],
    "dormancy_constant": ["FxC50_dormancy_Constant", "FxC50_dormancyCasPK_Constant", "FxC50_dormancyCellPK_Constant"],
    "dormancy_terminal": ["FxC50_dormancy_Terminal", "FxC50_dormancyCasPK_Terminal", "FxC50_dormancyCellPK_Terminal"],
}


def calculate_correlations(
    data: pl.DataFrame,
    features: List[str],
    method: str = "spearman"
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Calculate pairwise correlations between features.

    Parameters:
    -----------
    data : pl.DataFrame
        Input dataframe
    features : List[str]
        List of feature column names to correlate
    method : str
        Correlation method: 'spearman' or 'pearson'

    Returns:
    --------
    Dict mapping (feature1, feature2) -> {'rho': value, 'p_value': value, 'n': count}
    """
    results = {}

    for feat1, feat2 in combinations(features, 2):
        # Extract the two columns and drop nulls
        subset = data.select([feat1, feat2]).drop_nulls()

        if len(subset) < 3:  # Need at least 3 points for correlation
            results[(feat1, feat2)] = {
                'rho': np.nan,
                'p_value': np.nan,
                'n': len(subset)
            }
            continue

        # Convert to numpy for scipy
        arr1 = subset[feat1].to_numpy()
        arr2 = subset[feat2].to_numpy()

        # Calculate correlation
        if method == "spearman":
            rho, p_value = spearmanr(arr1, arr2)
        elif method == "pearson":
            rho, p_value = pearsonr(arr1, arr2)
        else:
            raise ValueError(f"Unknown method: {method}")

        results[(feat1, feat2)] = {
            'rho': rho,
            'p_value': p_value,
            'n': len(subset)
        }

    return results


def print_correlation_results(
    results: Dict[Tuple[str, str], Dict[str, float]],
    group_name: str
):
    """Print correlation results in a readable format."""
    print(f"\n{'='*80}")
    print(f"Correlation Group: {group_name}")
    print(f"{'='*80}")

    for (feat1, feat2), stats in results.items():
        # Simplify feature names for display
        f1_short = feat1.replace("FxC50_", "")
        f2_short = feat2.replace("FxC50_", "")

        print(f"\n{f1_short} vs {f2_short}:")
        print(f"  ρ = {stats['rho']:.3f}")
        print(f"  p-value = {stats['p_value']:.4f}")
        print(f"  n = {stats['n']}")

        # Significance indicator
        if stats['p_value'] < 0.001:
            sig = "***"
        elif stats['p_value'] < 0.01:
            sig = "**"
        elif stats['p_value'] < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(f"  Significance: {sig}")


def create_correlation_summary_table(
    all_results: Dict[str, Dict[Tuple[str, str], Dict[str, float]]]
) -> pl.DataFrame:
    """Create a summary table of all correlations."""
    rows = []

    for group_name, results in all_results.items():
        for (feat1, feat2), stats in results.items():
            rows.append({
                "group": group_name,
                "feature_1": feat1.replace("FxC50_", ""),
                "feature_2": feat2.replace("FxC50_", ""),
                "rho": stats['rho'],
                "p_value": stats['p_value'],
                "n": stats['n']
            })

    return pl.DataFrame(rows)


if __name__ == "__main__":
    # Calculate correlations for all groups
    all_results = {}

    for group_name, features in CORRELATION_GROUPS.items():
        print(f"\nProcessing group: {group_name}")
        results = calculate_correlations(
            diamond_data,
            features,
            method="spearman"
        )
        all_results[group_name] = results
        print_correlation_results(results, group_name)

    # Create summary table
    summary_df = create_correlation_summary_table(all_results)

    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(summary_df)

    # Save to CSV
    summary_df.write_csv("data/in_vitro_correlations_summary.csv")
    print("\n✓ Summary table saved to: data/in_vitro_correlations_summary.csv")
