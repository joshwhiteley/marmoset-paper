#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid blocking
import matplotlib.pyplot as plt
from radiodensity_corr_plotting import load_correlation_plotter

# Configuration
LESS_SEVERE_RHO_VALUES_CSV = "data/correlations/less_severe_rho_values_mean.csv"
LESS_SEVERE_P_VALUES_CSV = "data/correlations/less_severe_p_values_mean.csv"
SEVERE_RHO_VALUES_CSV = "data/correlations/severe_rho_values_mean.csv"
SEVERE_P_VALUES_CSV = "data/correlations/severe_p_values_mean.csv"
ALPHA = 0.05

# Define the pathology measures and custom order
pathology_rows = ["TP6_MeanHU", 
                  "TP6_TotalVol",
                  "delta_MeanHU", 
                  "delta_TotalVol"]

# Custom pathology order as requested: 
# | mean radiodensity | delta radiodensity | mean PET activity | delta mean PET activity | ...
custom_pathology_order = [
    "TP6_MeanHU",      # mean radiodensity
    "delta_MeanHU",    # delta radiodensity
    "TP6_TotalVol",    # total volume
    "delta_TotalVol"   # delta total volume
]

def main():
    # Load the correlation plotter
    print("Loading correlation plotter...")
    plotter = load_correlation_plotter(
        less_severe_rho_file=LESS_SEVERE_RHO_VALUES_CSV,
        less_severe_p_file=LESS_SEVERE_P_VALUES_CSV,
        severe_rho_file=SEVERE_RHO_VALUES_CSV,
        severe_p_file=SEVERE_P_VALUES_CSV,
        alpha=ALPHA
    )
    
    # Print data summary for debugging
    print("\n=== DEBUGGING INFORMATION ===")
    plotter.print_data_summary(pathology_rows)
    print("\n" + "="*50 + "\n")
    
    # Create enhanced legend that includes information about grey squares
    print("Creating legend...")
    legend_fig = plotter.create_legend(
        figsize=(16, 3), 
        orientation='horizontal', 
        include_metric_legend=True,
        improved_colormap=True, 
        show_grey_square=True, 
        save_to="figures_v2/3/correlations_legend_v2.svg"
    )
    print("Legend saved successfully!")
    plt.close(legend_fig)  # Close figure to free memory
    
    # Create new transposed heatmaps with timepoint collapse
    print("Creating NEW TRANSPOSED heatmaps with timepoint selection...")
    
    # Less severe lesions - NEW LAYOUT (transposed, collapsed timepoints)
    fig_less_severe, ax_less_severe = plotter.create_individual_heatmap(
        severity='less_severe',
        plot_type='rho',
        title=f"Less Severe Lesions - Transposed Layout (p < {ALPHA})",
        figsize=(8, 20),  # Taller for more features as rows
        add_grid=True,
        color_ylabels=True,  # Color Y-axis labels (features)
        improved_colormap=True,
        debug=True,
        pathology_rows=pathology_rows,
        save_to="figures_v2/3/transposed_less_severe_correlations.svg",
        collapse_timepoints=True,  # NEW: Collapse timepoints
        transpose=True,  # NEW: Transpose layout
        custom_pathology_order=custom_pathology_order  # NEW: Custom column order
    )
    print("Less severe heatmap saved successfully!")
    plt.close(fig_less_severe)  # Close figure to free memory
    
    # Severe lesions - NEW LAYOUT (transposed, collapsed timepoints)
    fig_severe, ax_severe = plotter.create_individual_heatmap(
        severity='severe',
        plot_type='rho',
        title=f"Severe Lesions - Transposed Layout (p < {ALPHA})",
        figsize=(8, 20),  # Taller for more features as rows
        add_grid=True,
        color_ylabels=True,  # Color Y-axis labels (features)
        improved_colormap=True,
        debug=True,
        pathology_rows=pathology_rows,
        save_to="figures_v2/3/transposed_severe_correlations.svg",
        collapse_timepoints=True,  # NEW: Collapse timepoints
        transpose=True,  # NEW: Transpose layout
        custom_pathology_order=custom_pathology_order  # NEW: Custom column order
    )
    print("Severe heatmap saved successfully!")
    plt.close(fig_severe)  # Close figure to free memory
    
    # Create feature type colorbar
    print("Creating feature type colorbar...")
    # Get the collapsed/ordered features from the plotter
    all_features = plotter.less_severe_rho['feature'].tolist()
    if True:  # collapse_timepoints was True in the heatmaps
        ordered_features = plotter._get_collapsed_features(
            plotter.less_severe_rho, 
            plotter.less_severe_p, 
            pathology_rows
        )
    else:
        ordered_features = all_features
    
    # Categorize and order features the same way as in the heatmaps
    feature_categories = plotter._categorize_feature(ordered_features)
    final_ordered_features = []
    for category_key, cat_features in feature_categories.items():
        final_ordered_features.extend(cat_features)
    
    # Create the feature type colorbar
    colorbar_fig = plotter.create_feature_type_colorbar(
        ordered_features=final_ordered_features,
        figsize=(1, 20),  # Match the height of your heatmaps
        orientation='vertical',
        save_to="figures_v2/3/feature_type_colorbar.svg"
    )
    print("Feature type colorbar saved successfully!")
    plt.close(colorbar_fig)
    
    # Create standalone feature type legend
    print("Creating standalone feature type legend...")
    legend_fig, ax = plt.subplots(figsize=(8, 2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create metric legend elements
    legend_elements = []
    for metric, color in plotter.metric_colors.items():
        if metric != 'default':  # Skip default color
            legend_elements.append(
                plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                         markersize=15, label=metric, markeredgecolor='black')
            )
    
    # Add the legend
    ax.legend(handles=legend_elements, loc='center', ncol=len(legend_elements), 
              title='Feature Types by Metric', title_fontsize=14, fontsize=12,
              frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig("figures_v2/3/feature_type_legend.svg", dpi=300, bbox_inches='tight')
    print("Feature type legend saved successfully!")
    plt.close(legend_fig)
    
    print("All plots completed successfully!")

if __name__ == "__main__":
    main()