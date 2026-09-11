
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Union, List, Optional
import shap
import warnings

# Suppress FutureWarnings from SHAP
warnings.filterwarnings('ignore', category=FutureWarning)

from marmoset_paper.helpers.ModelingFunctions import get_compound_abbreviation, export_shap_feature_data
from marmoset_paper.helpers.FeatureAnalysisFunctions import categorize_features
from marmoset_paper.helpers.constants import FEATURE_CATEGORY_COLORS

def plot_lesion_progressions(
    compound: str,
    num_lesions: int,
    test_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    debug: bool = False,
    save_dir: str = "figures/predictions",
    save_bool: bool = True,
    random_state: int = 42,
    model_prefix: str = "none",
    output_format: str = "none",
) -> None:
    """Plot lesion progressions for a given compound.
    
    args:
    ------
    - compound: Name of the compound to plot
    - num_lesions: Number of random lesions to plot
    - test_df: DataFrame containing true values
    - pred_df: DataFrame containing predicted values
    - debug: Whether to print debug information
    - save_dir: Directory to save figures
    - save_bool: Whether to save the figure
    - random_state: Random seed for reproducibility
    - model_prefix: b, d, or m, prefix for saving model names
    - output_format: how to save the file
        
    returns:
    ----- 
    -> None
    -> saves plots of lesion progression if save_bool is true.
    
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    abbrev = get_compound_abbreviation(compound)
    
    # filter data by compound
    compound_test = test_df.filter(pl.col("Compound") == compound)
    compound_pred = pred_df.filter(pl.col("Compound") == compound)
    
    if debug:
        lesion_counts = compound_test.group_by("Lesion").agg(
            pl.len().alias("count")
        )
        print(f"\nNumber of data points per lesion for {compound}:")
        print(lesion_counts)
    
    # random lesion selection for plotting
    available_lesions = compound_test["Lesion"].unique().to_list()
    np.random.seed(random_state)
    selected_lesions = np.random.choice(
        available_lesions,
        size=min(num_lesions, len(available_lesions)),
        replace=False
    )
    
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f"MeanHU Lesion Progressions for {compound}", fontsize=16)
    
    # timepoints could be shifted to weeks
    timepoints = ['TP2', 'TP3', 'TP4', 'TP5', 'TP6']
    x = np.arange(len(timepoints))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_lesions)))
    
    # plot each lesion
    for idx, lesion in enumerate(selected_lesions):
        lesion_test = compound_test.filter(pl.col("Lesion") == lesion)
        lesion_pred = compound_pred.filter(pl.col("Lesion") == lesion)
        
        hu_true = [lesion_test[f"TP{i}_MeanHU"].to_list()[0] 
                for i in range(2, 7)]
        hu_pred = [lesion_pred[f"TP{i}_MeanHU"].to_list()[0] 
                for i in range(3, 7)]
        hu_pred = [hu_true[0]] + hu_pred 
        
        # MeanHU plotting
        ax1.plot(x, hu_true, 'o-', color=colors[idx],
                label=f'Lesion {lesion} (True)', alpha=0.7)
        ax1.plot(x, hu_pred, 'o--', color=colors[idx],
                label=f'Lesion {lesion} (Pred)', alpha=0.7)
    
    ax1.set_title('MeanHU')
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('MeanHU')
    ax1.set_ylim(-375, -25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(timepoints)
    ax1.grid(True, alpha=0.3)

    # custom legend
    line_style_legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label='True'),
        Line2D([0], [0], color='black', linestyle='--', label='Predicted')
    ]

    legend1 = ax1.legend(bbox_to_anchor=(1.05, 1),
                        loc='upper left', title="Lesions")
    legend2 = ax1.legend(
        handles=line_style_legend_elements,
        bbox_to_anchor=(1.05, 0.5),
        loc='center left',
        title="Line Style"
    )
    ax1.add_artist(legend1)
    
    plt.tight_layout()
    if save_bool:
        now = datetime.datetime.now().strftime("%Y%m%d")
        save_file = save_path / f"{model_prefix}_{abbrev}_lesion_progressions_{now}.{output_format}"
        plt.savefig(save_file, bbox_inches='tight', dpi=300, format=output_format)
    plt.close()

def plot_ground_truth_lesion_trajectory(
    data_df: pl.DataFrame,
    compound: str,
    num_lesions_to_plot: int = 5,
    specific_lesions: Optional[List[int]] = None,
    timepoint_style: str = "week",
    save_dir: str = "figures/ground_truth",
    save_bool: bool = True,
    random_state: int = 42,
    output_format: str = "png"
) -> None:
    """
    Plots the ground truth lesion trajectories for a given compound.

    args:
    -----
    - data_df: pl.DataFrame containing the ground truth lesion data. 
    -- expected columns: 'Compound', 'Lesion', 'TP{i}_MeanHU', 'TP{i}_MeanSUV' (for i=2..6).
    - compound: name of the compound to plot.
    - num_lesions_to_plot: number of random lesions to plot if specific_lesions is None.
    - specific_lesions: list of specific lesion IDs to plot. If provided, num_lesions_to_plot is ignored.
    - save_dir: directory to save the figure.
    - save_bool: whether to save the figure.
    - random_state: random seed for reproducibility when selecting random lesions.
    - output_format: type of file to save plots as
    
    returns:
    ------
    -> None
    -> saves plots of ground truth lesion trajectories if save_bool is true
    
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    try:
        abbrev = get_compound_abbreviation(compound) 
    except Exception as e:
        print(f"Warning: Could not get abbreviation for {compound}. Using full name. Error: {e}")
        abbrev = compound.replace("+", "_")

    compound_data = data_df.filter(pl.col("Compound") == compound)

    if compound_data.is_empty():
        print(f"No data found for compound: {compound}")
        return

    available_lesions = compound_data["Lesion"].unique().to_list()
    
    if not available_lesions:
        print(f"No lesions found for compound: {compound}")
        return

    if specific_lesions:
        selected_lesions = [l for l in specific_lesions if l in available_lesions]
        if not selected_lesions:
            print(f"None of the specified lesions {specific_lesions} found for compound {compound}.")
            return
        print(f"Plotting specified lesions for {compound}: {selected_lesions}")
    else:
        np.random.seed(random_state)
        selected_lesions = np.random.choice(
            available_lesions,
            size=min(num_lesions_to_plot, len(available_lesions)),
            replace=False,
        )
        print(f"Plotting {len(selected_lesions)} random lesions for {compound}: {selected_lesions.tolist()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) 
    fig.suptitle(f"Ground Truth Lesion Trajectories for {compound}", fontsize=16)

    # timepoints
    if timepoint_style == "week":
        timepoints = ['week 0', 'week 2', 'week 4', 'week 6', 'week 8']
    elif timepoint_style == 'tp': 
        timepoints = ['TP2', 'TP3', 'TP4', 'TP5', 'TP6']
    else:
        return KeyError(f"Invalid timepoint_style argument: {timepoint_style}. Choose from 'week' or 'tp'.")
    
    x = np.arange(len(timepoints))

    colors = plt.cm.tab10(np.linspace(0, 1, len(selected_lesions)))

    # plot each selected lesion
    for idx, lesion in enumerate(selected_lesions):
        lesion_data = compound_data.filter(pl.col("Lesion") == lesion)

        if lesion_data.is_empty():
            print(f"Warning: No data found for lesion {lesion} in compound {compound} after filtering.")
            continue
            
        # extract ground truth values - take the first row if multiple exist for a lesion
        try:
            hu_true = [lesion_data[f"TP{i}_MeanHU"].to_list()[0] for i in range(2, 7)]
            suv_true = [lesion_data[f"TP{i}_MeanSUV"].to_list()[0] for i in range(2, 7)]
        except IndexError:
            print(f"Warning: Could not extract all timepoints for lesion {lesion}. Skipping.")
            continue
        except pl.ColumnNotFoundError as e:
            print(f"Warning: Missing expected column for lesion {lesion}. Error: {e}. Skipping.")
            continue


        # HU
        ax1.plot(x, hu_true, 'o-', color=colors[idx], label=f'Lesion {lesion}', alpha=0.8)

        # SUV
        ax2.plot(x, suv_true, 'o-', color=colors[idx], label=f'Lesion {lesion}', alpha=0.8)

    # meanHU plot
    ax1.set_title('MeanHU')
    ax1.set_xlabel('Timepoint')
    ax1.set_ylabel('MeanHU')
    ax1.set_xticks(x)
    ax1.set_xticklabels(timepoints)
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Lesions")

    # meanSUV plot
    ax2.set_title('MeanSUV')
    ax2.set_xlabel('Timepoint')
    ax2.set_ylabel('MeanSUV')
    ax2.set_xticks(x)
    ax2.set_xticklabels(timepoints)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_bool:
        save_file = save_path / f"{abbrev}_ground_truth_lesions.{output_format}"
        try:
            plt.savefig(save_file, bbox_inches='tight', dpi=300, format=output_format)
            print(f"MSE plot saved to: {save_file}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.close(fig) 


def plot_performance_metrics(
    metrics_df: pl.DataFrame,
    save_path: Union[str, Path],
    outputs: list,
    metrics: list = ["mse", "mae", "r2"],
    save_prefix: Optional[str] = "",
    save_format: Optional[str] = "png",
    save_bool: bool = True,
) -> None:
    """
    Generates and saves a bar chart visualizing performance metrics results by timepoint.

    args:
    -----
    - metrics_df: pl.DataFrame with various error metric information
    - save_path: where to save the figures
    - outputs: list of different PET/CT outputs (TP{}_meanHU, etc)
    - metrics: list of error metrics to plot ["mse", "mae", "r2"]
    - save_prefix: model type [b, m, d]
    - save_format: filetype of figures saved
    - save_bool: save the figures or not
    
    returns:
    -----
    -> None
    -> saved plots of performance metrics in save_path if save_bool is true

    """
    
    now = datetime.datetime.now().strftime("%Y%m%d")
    timepoints = sorted(metrics_df['timepoint'].unique().to_list())
    
    output_groups = {
        "HU": [o for o in outputs if "HU" in o],
        "SUV": [o for o in outputs if "SUV" in o]
    }
    
    for metric in metrics:
        for group_name, group_outputs in output_groups.items():
            if not group_outputs:
                continue
            plt.figure(figsize=(10, 6))
            bar_width = 0.8 / len(group_outputs)
            indices = np.arange(len(timepoints))
            for i, output in enumerate(group_outputs):
                df_plot = metrics_df.filter(pl.col('output') == output).sort('timepoint')
                yvals = df_plot[metric].to_list()
                plt.bar(indices + i * bar_width, yvals, width=bar_width, label=output)
                for j, y in enumerate(yvals):
                    plt.text(indices[j] + i * bar_width, y, f"{y:.2f}", ha='center', va='bottom', fontsize=8)
            plt.xlabel('Timepoint')
            plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} by timepoint ({group_name}) -- {save_prefix}")
            plt.xticks(indices + bar_width * (len(group_outputs) - 1) / 2, [f"TP{int(tp)}" for tp in timepoints])
            plt.legend(title="Output")
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            if save_bool:
                Path(save_path).mkdir(parents=True, exist_ok=True)
                filename = f"{save_prefix}_{metric}_{group_name}_tp_{now}.{save_format}"
                plt.savefig(Path(save_path) / filename, bbox_inches="tight", dpi=300, format=save_format)
                print(f"Saved: {Path(save_path) / filename}")
            plt.close()

    
def plot_feature_importance_bars(
    importance_df: pd.DataFrame,
    tp_key: str,
    save_path: Union[str, Path],
    importance_col: str = 'importance',
    top_n: int = 20,
    title_suffix: str = 'mean decrease in impurity',
    save_bool: bool = True,
    save_format: str = 'png',
    model_prefix: str = "none"
) -> None:
    """
    creates and saves a bar chart for feature importance, colored by cat.
    
    args:
    ------
    - importance_df: pd.DataFrame with 'feature' and {importance_col} columns
    - tp_key: timepoint key (tp3, etc.)
    - save_path: directory to save plot if save_bool is true
    - importance_col: name of column with importance values
    - top_n: number of features to display
    - title_suffix: suffix for plot title
    - save_bool: true/false for saving plots generated
    - save_format: type of file to save
    - model_prefix: type of model
    
    returns:
    -----
    -> None
    -> saves plots of feature importance stuff 
    """
    
    if importance_df is None or importance_df.empty:
        print(f"importance df is empty for {tp_key}")
        return
    
    if importance_col not in importance_df.columns:
        print(f"error: importance column {importance_col} not found in df")
        return
    
    importance_df['category'] = importance_df['feature'].apply(categorize_features)
    plot_df = importance_df[importance_df['category'] != 'metadata'].copy()
    plot_df = plot_df.nlargest(top_n, importance_col).sort_values(by=importance_col, ascending=True)
    
    if plot_df.empty:
        print(f"no non-metadata features with >0 importance found for {tp_key}")
        return
    
    plt.figure(figsize=(10, max(6, top_n*0.3)))
    
    colors = [FEATURE_CATEGORY_COLORS.get(cat, FEATURE_CATEGORY_COLORS["unknown"]) for cat in plot_df['category']]
    bars = plt.barh(plot_df['feature'], plot_df[importance_col], color=colors)
    
    plt.xlabel(f"importance ({title_suffix})")
    plt.ylabel("feature")
    plt.title(f"top {len(plot_df)} feature importances for {tp_key.upper()}")
    plt.gca().margins(y=0.01)
    
    legend_handles = [plt.Rectangle((0,0),1,1, color=FEATURE_CATEGORY_COLORS[cat]) for cat in sorted(plot_df['category'].unique())]
    legend_labels = sorted(plot_df['category'].unique())
    plt.legend(legend_handles, legend_labels, title='feature category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    if save_bool:
        now = datetime.datetime.now().strftime("%Y%m%d")
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(parents=True, exist_ok=True)
        filename = f"{model_prefix}_{tp_key.upper()}_feature_importance_{title_suffix.replace(' ', '_')}_{now}.{save_format}"
        full_save_path = save_path_obj / filename
        try:
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight', format=save_format)
            print(f"{tp_key.upper()} importance plot saved to: {full_save_path}")
        except Exception as e:
            print(f"error saving importance plot: \n {e}")
            
def plot_shap_summary_dot(
    shap_values: np.ndarray,
    X_data_df: pd.DataFrame,
    tp_key: str,
    output_name: str,
    save_path: Union[str, Path],
    save_bool: bool = True,
    save_format: str = 'png',
    save_shap_df: bool = True,
    model_prefix: str = "none",
    debug: bool = False
) -> None:
    """
    generates and saves a SHAP summary dot plot for specific model output,
    plus a zoomed-in version that drops the top-N features depending on tp_key.
    """
    try:
        np.random.seed(42)

        # 1) Main SHAP summary
        plt.figure()
        shap.summary_plot(
            shap_values,
            X_data_df,
            plot_type="dot",
            show=False
        )
        plt.title(f"SHAP summary for {tp_key.upper()} - prediction: {output_name}")
        plt.ylabel("feature name")

        if save_bool:
            now = datetime.datetime.now().strftime("%Y%m%d")
            save_path_obj = Path(save_path)
            save_path_obj.mkdir(parents=True, exist_ok=True)

            # save main plot
            main_filename = f"{model_prefix}_{tp_key}_shap_summary_{output_name}_{now}.{save_format}"
            plt.tight_layout()
            plt.savefig(save_path_obj / main_filename, dpi=300, bbox_inches='tight', format=save_format)
            print(f"{tp_key.upper()} SHAP plot saved to: {save_path_obj / main_filename}")

            # 2) Zoomed-in SHAP summary: drop the top-N features
            cutoff_map = {
                'tp3': 2,
                'tp4': 4,
                'tp5': 6,
                'tp6': 8,
            }
            cutoff = cutoff_map.get(tp_key.lower())
            if cutoff:
                # rank features by mean absolute SHAP
                mean_abs = np.mean(np.abs(shap_values), axis=0)
                order = np.argsort(mean_abs)[::-1]
                zoom_indices = order[cutoff:]

                # slice out just the lower-ranked features
                zoom_shap = shap_values[:, zoom_indices]
                zoom_X    = X_data_df.iloc[:, zoom_indices]

                plt.figure()
                shap.summary_plot(
                    zoom_shap,
                    zoom_X,
                    plot_type="dot",
                    show=False
                )
                plt.title(f"SHAP summary for {tp_key.upper()} - prediction: {output_name} (zoomed)")

                zoom_filename = f"{model_prefix}_{tp_key}_shap_summary_{output_name}_{now}_zoomed.{save_format}"
                plt.tight_layout()
                plt.savefig(save_path_obj / zoom_filename, dpi=300, bbox_inches='tight', format=save_format)
                print(f"{tp_key.upper()} zoomed SHAP plot saved to: {save_path_obj / zoom_filename}")
                plt.close()

        plt.close()

    except Exception as e:
        print(f"error generating SHAP plot for {tp_key}, output {output_name}: {e}")
        plt.close()

    # optionally write out the raw SHAP values to CSV
    if save_shap_df:
        shap_df = pd.DataFrame(shap_values, columns=X_data_df.columns)
        shap_df_path = Path(save_path) / f"{model_prefix}_{tp_key}_shap_values_{output_name}.csv"
        shap_df.to_csv(shap_df_path, index=False)
        print(f"- {tp_key.upper()} SHAP values DataFrame saved to: {shap_df_path}")

    # Export SHAP data for available features only
    try:
        # Only export if the feature exists in the current dataset
        target_features = ['LoeweFIC90_cholesterol_Constant']
        available_features = [feat for feat in target_features if feat in X_data_df.columns]

        for feat in available_features:
            export_shap_feature_data(
                shap_values=shap_values,
                X_df=X_data_df,
                feature_name=feat,
                tp_key=tp_key,
                output_name=output_name,
                model_prefix=model_prefix,
                save_path=save_path
            )

    except Exception as e:
        print(f"error exporting SHAP data for {tp_key}, output {output_name}: {e}")