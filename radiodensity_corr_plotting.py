import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import OrderedDict
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import List, Optional, Union, Tuple, Dict


class CorrelationHeatmapPlotter:
    """
    Enhanced correlation plotting class that creates heatmap-style plots 
    for rho and p values with configurable rows and grouped X values.
    Shows non-significant correlations as light grey squares.
    Updated to select best timepoint per feature and transpose layout.
    """
    
    def __init__(self, 
                 less_severe_rho_file: str,
                 less_severe_p_file: str,
                 severe_rho_file: str,
                 severe_p_file: str,
                 alpha: float = 0.05):
        """
        Initialize with separate files for rho and p values for both severity levels.
        """
        self.less_severe_rho = self._load_and_align(less_severe_rho_file)
        self.less_severe_p = self._load_and_align(less_severe_p_file)
        self.severe_rho = self._load_and_align(severe_rho_file)
        self.severe_p = self._load_and_align(severe_p_file)
        self.alpha = alpha
        
        # Special value to mark non-significant correlations
        self.NONSIG_VALUE = 999.0
        
        # Get available pathology columns (excluding feature columns)
        self.pathology_columns = [col for col in self.less_severe_rho.columns 
                                 if col not in ['feature', 'feature_key']]
        
        # Define color scheme for different metrics
        self.metric_colors = {
            'AUC': '#FF8C00',      # DarkOrange
            'Einf': '#2E8B57',     # SeaGreen  
            'FxC': '#FF1493',      # DeepPink (for FIC/FBC)
            'GRmax': '#8A2BE2',    # BlueViolet
            'default': '#000000'   # Black for unrecognized metrics
        }
    
    def _save_figure(self, fig: plt.Figure, save_path: str, dpi: int = 300, 
                    bbox_inches: str = 'tight', debug: bool = False) -> None:
        """
        Save figure to specified path with proper directory creation.
        
        Parameters:
        -----------
        fig : plt.Figure
            The figure to save
        save_path : str
            Full path where to save the figure (including filename)
        dpi : int
            DPI for saving (default 300)
        bbox_inches : str
            Bbox setting for saving (default 'tight')
        debug : bool
            Whether to print debug information
        """
        if save_path is None:
            return
            
        # Extract directory from full path
        save_dir = os.path.dirname(save_path)
        
        # Create directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if debug:
                print(f"Created/verified directory: {save_dir}")
        
        # Save the figure
        fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        if debug:
            print(f"Saved figure to: {save_path}")
    
    def _load_and_align(self, filepath: str) -> pd.DataFrame:
        """Load CSV and create aligned feature keys."""
        df = pd.read_csv(filepath)
        df = self._build_alignment(df)
        return df
    
    def _build_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build feature alignment similar to correlations_plotting.ipynb."""
        df = df.copy()
        feat_col = next((c for c in df.columns if str(c).strip().lower() == "in_vitro_feature"), df.columns[0])
        df.rename(columns={feat_col: "feature"}, inplace=True)
        df["feature_key"] = df["feature"].map(self._canonical_feature)
        return df
    
    def _canonical_feature(self, s: str) -> Optional[str]:
        """Return canonical key for EQ/PK/LID features."""
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return None
        s0 = str(s).strip()

        # LIDs (p-style; allow trailing spaces)
        m = re.match(r"^(?P<prefix>[57][NH])\s+(?P<pk>(?:casPK|cellPK|equip))\s+(?P<metric>AUC\d+|FBC\d+)\s*$",
                        s0, re.IGNORECASE)
        if m:
            return f"LID|{m.group('prefix').upper()}|{m.group('pk').lower()}|{m.group('metric').upper()}"

        # LIDs (rho-style variants)
        m = re.match(r"^(?P<prefix>[57][NH])\s*PK(?P<pk>cas|cell|equip)(?P<metric>AUC\d+|FBC\d+)\s*$",
                        s0, re.IGNORECASE)
        if m:
            return f"LID|{m.group('prefix').upper()}|{m.group('pk').lower()}|{m.group('metric').upper()}"

        # PK (rho-style) with OPTIONAL phase
        m = re.match(r"^PK(?P<which>cas|cell)(?P<subs>[A-Za-z]+)(?P<metric>AUC\d+|GRinf)(?P<phase>CT|C|T)?$",
                        s0, re.IGNORECASE)
        if m:
            which = m.group("which").lower()
            subs = m.group("subs").lower()
            metric = m.group("metric").upper()
            phase = (m.group("phase") or "").upper()
            return f"PK|{which}|{subs}|{metric}|{phase}"

        # PK (p-style)
        m = re.match(r"^PK_(?P<which>cas|cell)\s+(?P<metric>AUC\d+|GRinf)\s+(?P<subs>[A-Za-z]+)\s*\((?P<phase>[^)]+)\)\s*$",
                        s0, re.IGNORECASE)
        if m:
            return f"PK|{m.group('which').lower()}|{m.group('subs').lower()}|{m.group('metric').upper()}|{self._abbr_phase(m.group('phase'))}"

        # EQ (rho-style) with OPTIONAL phase
        m = re.match(r"^(?P<subs>[A-Za-z]+)(?P<metric>AUC\d+|FIC\d+|FBC\d+|GRinf)(?P<phase>CT|C|T)?$",
                        s0, re.IGNORECASE)
        if m:
            subs = m.group("subs").lower()
            metric = m.group("metric").upper()
            phase = (m.group("phase") or "").upper()
            return f"EQ|{metric}|{subs}|{phase}"

        # EQ (p-style)
        m = re.match(r"^(?P<metric>[A-Z]+[0-9]*|GRinf)\s+(?P<subs>[A-Za-z]+)\s*\((?P<phase>[^)]+)\)",
                        s0, re.IGNORECASE)
        if m:
            return f"EQ|{m.group('metric').upper()}|{m.group('subs').lower()}|{self._abbr_phase(m.group('phase'))}"

        return None
    
    def _abbr_phase(self, text: str) -> str:
        """Abbreviate phase descriptions."""
        t = str(text).strip().lower()
        if "constant" in t and "terminal" in t:
            return "CT"
        if "constant" in t:
            return "C"
        if "terminal" in t:
            return "T"
        return t[:1].upper() if t else ""
    
    def _extract_metric_type(self, feature: str) -> str:
        """Extract metric type from feature name for color coding."""
        feature_upper = feature.upper()
        
        # Check for specific patterns - more comprehensive matching
        if 'AUC' in feature_upper:
            return 'AUC'
        elif 'EINF' in feature_upper or 'E_INF' in feature_upper:
            return 'Einf'
        elif any(pattern in feature_upper for pattern in ['FIC', 'FBC', 'F50', 'F90', 'FXC50', 'FXC90']):
            return 'FxC'
        elif 'GRMAX' in feature_upper or 'GR_MAX' in feature_upper or 'GR MAX' in feature_upper:
            return 'GRmax'
        else:
            return 'default'
    
    def _get_feature_colors(self, features: List[str]) -> List[str]:
        """Get color list for features based on their metric type."""
        colors = []
        for feature in features:
            metric_type = self._extract_metric_type(feature)
            colors.append(self.metric_colors[metric_type])
        return colors
    
    def _select_best_timepoint_feature(self, features_with_timepoints: List[str], 
                                        rho_df: pd.DataFrame, p_df: pd.DataFrame, 
                                        pathology_columns: List[str]) -> str:
        """Select the feature with the highest absolute correlation across all pathology measures."""
        best_feature = None
        max_abs_corr = -1
        
        for feature in features_with_timepoints:
            total_abs_corr = 0
            valid_correlations = 0
            
            for path_col in pathology_columns:
                rho_val = self._get_value(rho_df, feature, path_col)
                p_val = self._get_value(p_df, feature, path_col)
                
                if rho_val is not None and p_val is not None and p_val < self.alpha:
                    # Apply Einf flipping for comparison
                    if 'einf' in feature.lower() or 'e_inf' in feature.lower():
                        rho_val = -rho_val
                    total_abs_corr += abs(rho_val)
                    valid_correlations += 1
            
            # Average absolute correlation for this feature
            if valid_correlations > 0:
                avg_abs_corr = total_abs_corr / valid_correlations
                if avg_abs_corr > max_abs_corr:
                    max_abs_corr = avg_abs_corr
                    best_feature = feature
        
        return best_feature if best_feature else features_with_timepoints[0]
    
    def _group_features_by_base_name(self, features: List[str]) -> Dict[str, List[str]]:
        """Group features by base name (without timepoint suffixes)."""
        feature_groups = {}
        
        for feature in features:
            # Remove common timepoint suffixes to get base name
            base_name = feature
            for suffix in ['_Constant', '_Terminal', '_C', '_T', '(constant)', '(terminal)', 
                          'C', 'T', 'CT']:  # Added more patterns
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)].strip()
                    break
            
            if base_name not in feature_groups:
                feature_groups[base_name] = []
            feature_groups[base_name].append(feature)
        
        return feature_groups
    
    def _get_collapsed_features(self, rho_df: pd.DataFrame, p_df: pd.DataFrame, 
                               pathology_columns: List[str]) -> List[str]:
        """Get list of features with best timepoint selected per feature group."""
        all_features = rho_df['feature'].tolist()
        feature_groups = self._group_features_by_base_name(all_features)
        
        collapsed_features = []
        for base_name, grouped_features in feature_groups.items():
            if len(grouped_features) == 1:
                # Only one variant, keep it
                collapsed_features.append(grouped_features[0])
            else:
                # Multiple variants, select best one
                best_feature = self._select_best_timepoint_feature(
                    grouped_features, rho_df, p_df, pathology_columns
                )
                collapsed_features.append(best_feature)
        
        return collapsed_features
    
    def _classify_feature_condition(self, feature: str) -> str:
        """Classify feature into simple equipotent, simple PK, or LIDS based on condition."""
        feature_lower = feature.lower()
        
        # LIDS: features containing "5n", "7n", or "7h"
        if any(lid_marker in feature_lower for lid_marker in ["5n", "7n", "7h"]):
            return "LIDS"
        
        # Simple PK: features containing "cellpk" or "caspk" (but not LIDS markers)
        elif any(pk_marker in feature_lower for pk_marker in ["cellpk", "caspk"]):
            return "simple PK"
        
        # Simple equipotent: everything else
        else:
            return "simple equipotent"
    
    def _extract_condition_timepoint(self, feature: str) -> tuple:
        """Extract condition and timepoint from feature name for sorting."""
        feature_lower = feature.lower()
        
        # Try to extract timepoint (like C, T, CT at the end)
        timepoint = ""
        if feature_lower.endswith("ct"):
            timepoint = "CT"
        elif feature_lower.endswith("c"):
            timepoint = "C"
        elif feature_lower.endswith("t"):
            timepoint = "T"
        
        # Remove timepoint from condition for cleaner grouping
        condition = feature
        if timepoint:
            condition = feature[:-len(timepoint)].strip()
        
        return condition, timepoint
    
    def _categorize_feature(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize and sort features by condition-timepoint groupings."""
        
        # First, classify all features
        classified = {}
        for feature in features:
            classification = self._classify_feature_condition(feature)
            condition, timepoint = self._extract_condition_timepoint(feature)
            
            if classification not in classified:
                classified[classification] = {}
            if condition not in classified[classification]:
                classified[classification][condition] = []
            
            classified[classification][condition].append(feature)
        
        # Now organize in the requested order
        ordered_categories = OrderedDict()
        
        # Process in order: simple equipotent, simple PK, LIDS
        for classification in ["simple equipotent", "simple PK", "LIDS"]:
            if classification in classified:
                # Sort conditions alphabetically within each classification
                sorted_conditions = sorted(classified[classification].keys())
                
                for condition in sorted_conditions:
                    # Sort features within each condition (to handle multiple timepoints consistently)
                    sorted_features = sorted(classified[classification][condition])
                    
                    # Create a unique key for this condition group
                    group_key = f"{classification} - {condition}"
                    ordered_categories[group_key] = sorted_features
        
        return ordered_categories
    
    def _pretty_metric(self, metric: str) -> str:
        """Format metric names for display."""
        m = metric.upper()
        if m.startswith("FIC") or m.startswith("FBC"):
            return "FxC" + re.sub(r"\D", "", m)
        if m.startswith("AUC"):
            return "AUC" + re.sub(r"\D", "", m)
        if m.startswith("GRINF"):
            return "GRinf"
        return m
    
    def _clean_feature_names(self, features: List[str]) -> List[str]:
        """Clean feature names for display."""
        cleaned = []
        for feature in features:
            clean_name = feature.replace("(constant)", "C").replace("(terminal)", "T")
            clean_name = re.sub(r'\s+', ' ', clean_name).strip()
            cleaned.append(clean_name)
        return cleaned
    
    def _apply_colored_xlabels(self, ax: plt.Axes, labels: List[str], colors: List[str]):
        """Apply colored labels to x-axis."""
        # Set the labels first
        ax.set_xticklabels(labels, rotation=90)
        
        # Get the tick labels and apply colors
        for tick_label, color in zip(ax.get_xticklabels(), colors):
            tick_label.set_color(color)
            tick_label.set_weight('bold')  # Make them bold for better visibility
    
    def _apply_colored_ylabels(self, ax: plt.Axes, labels: List[str], colors: List[str]):
        """Apply colored labels to y-axis."""
        # Set the labels first
        ax.set_yticklabels(labels)
        
        # Get the tick labels and apply colors
        for tick_label, color in zip(ax.get_yticklabels(), colors):
            tick_label.set_color(color)
            tick_label.set_weight('bold')  # Make them bold for better visibility
    
    def _create_correlation_colormap_with_grey(self, cmap_colors=("#d73027", "#4575b4"), grey_color='#D3D3D3'):
        """
        Create a custom colormap that shows correlations in color and non-significant values as grey squares.
        
        Parameters:
        -----------
        cmap_colors : tuple
            Colors for negative and positive correlations
        grey_color : str  
            Color for non-significant correlations
        
        Returns:
        --------
        tuple : (colormap, normalizer)
            Custom colormap and normalizer for use with imshow/heatmap
        """
        
        # Create base correlation colormap (red to blue through white)
        colors = [
            cmap_colors[0],    # Strong red for negative correlations
            "#fee0d2",         # Light red 
            "#ffffff",         # White for zero
            "#deebf7",         # Light blue
            cmap_colors[1]     # Strong blue for positive correlations
        ]
        
        # Create discrete boundaries for better control
        n_corr_colors = 100
        base_cmap = LinearSegmentedColormap.from_list('correlation_base', colors, N=n_corr_colors)
        
        # Create boundaries: correlation values (-1 to 1) + special value for non-significant
        corr_bounds = np.linspace(-1, 1, n_corr_colors + 1)
        all_bounds = np.concatenate([corr_bounds, [self.NONSIG_VALUE - 0.1, self.NONSIG_VALUE + 0.1]])
        
        # Get colors: correlation colors + grey for non-significant
        corr_colors = [base_cmap(i / (n_corr_colors - 1)) for i in range(n_corr_colors)]
        all_colors = corr_colors + [mcolors.to_rgba(grey_color), mcolors.to_rgba(grey_color)]
        
        # Create the custom colormap and normalizer
        custom_cmap = ListedColormap(all_colors)
        custom_norm = BoundaryNorm(all_bounds, custom_cmap.N)
        
        return custom_cmap, custom_norm
    
    def _create_improved_colormap(self, cmap_colors: Tuple[str, str] = ("#d73027", "#4575b4")) -> LinearSegmentedColormap:
        """Create an improved colormap with better color differentiation (legacy method)."""
        # Use more distinct intermediate colors
        colors = [
            cmap_colors[0],    # Strong red for negative correlations
            "#fee0d2",         # Light red 
            "#ffffff",         # White for zero
            "#deebf7",         # Light blue
            cmap_colors[1]     # Strong blue for positive correlations
        ]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('improved_correlation', colors, N=n_bins)
        return cmap
    
    def _get_data_statistics(self, data_matrix: np.ndarray, mask_matrix: np.ndarray) -> Dict[str, float]:
        """Get statistics about the correlation data for debugging."""
        valid_data = data_matrix[~mask_matrix]
        # Exclude non-significant marker values from statistics
        valid_data = valid_data[valid_data != self.NONSIG_VALUE]
        
        if len(valid_data) == 0:
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "n_values": 0}
        
        return {
            "min": np.min(valid_data),
            "max": np.max(valid_data),
            "mean": np.mean(valid_data),
            "std": np.std(valid_data),
            "n_values": len(valid_data),
            "n_positive": np.sum(valid_data > 0),
            "n_negative": np.sum(valid_data < 0),
            "n_zero": np.sum(valid_data == 0)
        }
    
    def print_data_summary(self, pathology_rows: Optional[List[str]] = None) -> None:
        """Print summary statistics about correlation data for debugging."""
        if pathology_rows is None:
            pathology_rows = self.pathology_columns
        
        all_features = self.less_severe_rho['feature'].tolist()
        
        print("=== DATA SUMMARY ===")
        print(f"Number of features: {len(all_features)}")
        print(f"Number of pathology measures: {len(pathology_rows)}")
        print(f"Alpha threshold: {self.alpha}")
        print(f"Non-significant marker value: {self.NONSIG_VALUE}")
        
        # Sample some features to check Einf detection
        einf_features = [f for f in all_features if 'einf' in f.lower() or 'e_inf' in f.lower()]
        print(f"\nDetected Einf features ({len(einf_features)}):")
        for feature in einf_features[:5]:  # Show first 5
            print(f"  - {feature}")
        if len(einf_features) > 5:
            print(f"  ... and {len(einf_features) - 5} more")
        
        # Check correlation value ranges
        for severity, rho_df, p_df in [
            ("Less Severe", self.less_severe_rho, self.less_severe_p),
            ("Severe", self.severe_rho, self.severe_p)
        ]:
            print(f"\n{severity} Lesions:")
            all_rho_vals = []
            all_p_vals = []
            significant_rho_vals = []
            
            for path_col in pathology_rows:
                for feature in all_features:
                    rho_val = self._get_value(rho_df, feature, path_col)
                    p_val = self._get_value(p_df, feature, path_col)
                    
                    if rho_val is not None:
                        all_rho_vals.append(rho_val)
                    if p_val is not None:
                        all_p_vals.append(p_val)
                    
                    if rho_val is not None and p_val is not None and p_val < self.alpha:
                        # Apply Einf flipping for summary
                        if 'einf' in feature.lower() or 'e_inf' in feature.lower():
                            rho_val = -rho_val
                        significant_rho_vals.append(rho_val)
            
            print(f"  All rho values: min={np.min(all_rho_vals):.3f}, max={np.max(all_rho_vals):.3f}, n={len(all_rho_vals)}")
            print(f"  Significant rho values (p<{self.alpha}): min={np.min(significant_rho_vals):.3f}, max={np.max(significant_rho_vals):.3f}, n={len(significant_rho_vals)}")
            print(f"  P-values: min={np.min(all_p_vals):.3f}, max={np.max(all_p_vals):.3f}")
    
    def _prepare_data_matrix(self, rho_df: pd.DataFrame, p_df: pd.DataFrame, 
                            pathology_rows: List[str], ordered_features: List[str], 
                            plot_type: str = 'rho', debug: bool = False, transpose: bool = True):
        """Prepare data matrix for a single severity level with grey squares for non-significant values.
        
        Parameters:
        -----------
        transpose : bool
            If True, features are rows and pathology measures are columns
            If False, pathology measures are rows and features are columns (original layout)
        """
        if transpose:
            # Transposed: features as rows, pathology as columns
            n_rows = len(ordered_features)
            n_cols = len(pathology_rows)
        else:
            # Original: pathology as rows, features as columns
            n_rows = len(pathology_rows)
            n_cols = len(ordered_features)
        
        if plot_type == 'rho':
            data_matrix = np.full((n_rows, n_cols), np.nan)
            mask_matrix = np.full((n_rows, n_cols), True, dtype=bool)
            
            einf_flipped_count = 0
            total_significant = 0
            total_nonsignificant = 0
            
            for i_feat, feature in enumerate(ordered_features):
                for j_path, path_col in enumerate(pathology_rows):
                    # Get rho and p values
                    rho_val = self._get_value(rho_df, feature, path_col)
                    p_val = self._get_value(p_df, feature, path_col)
                    
                    if rho_val is not None and p_val is not None:
                        # Show rho if p < alpha (significant)
                        if p_val < self.alpha:
                            original_rho = rho_val
                            # Flip Einf correlations (multiply by -1)
                            if 'einf' in feature.lower() or 'e_inf' in feature.lower():
                                rho_val = -rho_val
                                einf_flipped_count += 1
                                if debug:
                                    print(f"Flipped Einf: {feature} from {original_rho:.3f} to {rho_val:.3f}")
                            
                            # Set matrix positions based on transpose
                            if transpose:
                                data_matrix[i_feat, j_path] = rho_val
                                mask_matrix[i_feat, j_path] = False
                            else:
                                data_matrix[j_path, i_feat] = rho_val
                                mask_matrix[j_path, i_feat] = False
                            total_significant += 1
                        else:
                            # Mark non-significant correlations with special value for grey squares
                            if transpose:
                                data_matrix[i_feat, j_path] = self.NONSIG_VALUE
                                mask_matrix[i_feat, j_path] = False
                            else:
                                data_matrix[j_path, i_feat] = self.NONSIG_VALUE
                                mask_matrix[j_path, i_feat] = False
                            total_nonsignificant += 1
            
            if debug:
                print(f"Data preparation complete: {total_significant} significant correlations, {total_nonsignificant} non-significant, {einf_flipped_count} Einf values flipped")
                stats = self._get_data_statistics(data_matrix, mask_matrix)
                print(f"Final data range: {stats['min']:.3f} to {stats['max']:.3f}, mean: {stats['mean']:.3f}")
            
            return data_matrix, mask_matrix
            
        elif plot_type == 'p':
            data_matrix = np.full((n_rows, n_cols), np.nan)
            mask_matrix = np.full((n_rows, n_cols), True, dtype=bool)
            
            for i_feat, feature in enumerate(ordered_features):
                for j_path, path_col in enumerate(pathology_rows):
                    p_val = self._get_value(p_df, feature, path_col)
                    
                    if p_val is not None:
                        if transpose:
                            data_matrix[i_feat, j_path] = p_val
                            mask_matrix[i_feat, j_path] = False
                        else:
                            data_matrix[j_path, i_feat] = p_val
                            mask_matrix[j_path, i_feat] = False
            
            return data_matrix, mask_matrix
    
    def create_feature_type_colorbar(self, ordered_features: List[str], 
                                    figsize: Tuple[float, float] = (1, 8),
                                    orientation: str = 'vertical',
                                    save_to: Optional[str] = None) -> plt.Figure:
        """Create a colorbar showing feature types alongside the main plot."""
        
        # Get colors for each feature
        feature_colors = self._get_feature_colors(ordered_features)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        if orientation == 'vertical':
            # Create vertical colorbar
            color_array = np.array(feature_colors).reshape(-1, 1)
            # Convert color names to RGB for display
            color_rgb = [mcolors.to_rgb(color) for color in feature_colors]
            color_array_rgb = np.array(color_rgb).reshape(-1, 1, 3)
            
            ax.imshow(color_array_rgb, aspect='auto', extent=[0, 1, 0, len(ordered_features)])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, len(ordered_features))
            ax.set_xticks([])
            ax.set_yticks(np.arange(len(ordered_features)) + 0.5)
            ax.set_yticklabels([])  # We'll add labels separately if needed
            ax.invert_yaxis()  # Match the main plot orientation
            
        ax.set_title('Feature\nTypes', fontsize=12, weight='bold')
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.tight_layout()
        
        # Save if requested
        if save_to is not None:
            self._save_figure(fig, save_to)
        
        return fig
    
    def create_individual_heatmap(self,
                                 severity: str = 'less_severe',  # 'less_severe' or 'severe'
                                 pathology_rows: Optional[List[str]] = None,
                                 include_features: Optional[List[str]] = None,
                                 exclude_features: Optional[List[str]] = None,
                                 plot_type: str = 'rho',  # 'rho' or 'p'
                                 figsize: Tuple[int, int] = (10, 16),  # Adjusted for transpose
                                 cmap_colors: Tuple[str, str] = ("#d73027", "#4575b4"),
                                 title: Optional[str] = None,
                                 add_grid: bool = True,
                                 color_ylabels: bool = True,  # Changed from color_xlabels
                                 improved_colormap: bool = True,
                                 save_to: Optional[str] = None,
                                 dpi: int = 300,
                                 debug: bool = False,
                                 collapse_timepoints: bool = True,  # New parameter
                                 transpose: bool = True,  # New parameter
                                 custom_pathology_order: Optional[List[str]] = None) -> Tuple[plt.Figure, plt.Axes]:  # New parameter
        """
        Create a single heatmap for either less severe or severe lesions with grey squares for non-significant correlations.
        
        Parameters:
        -----------
        severity : str
            'less_severe' or 'severe'
        pathology_rows : List[str], optional
            Which pathology columns to include
        include_features : List[str], optional
            Which features to include
        exclude_features : List[str], optional
            Which features to exclude
        plot_type : str
            'rho' for correlation values, 'p' for p-values
        figsize : Tuple[int, int]
            Figure size
        cmap_colors : Tuple[str, str]
            Colors for the colormap (negative, positive)
        title : str, optional
            Plot title (auto-generated if None)
        add_grid : bool
            Whether to add grid lines to the plot
        color_ylabels : bool
            Whether to color-code Y-axis labels by metric type (when transposed)
        improved_colormap : bool
            Whether to use improved colormap with better differentiation
        save_to : str, optional
            Path to save the figure (including filename and extension)
        dpi : int
            DPI for saving (default 300)
        debug : bool
            Whether to print debug information
        collapse_timepoints : bool
            Whether to collapse timepoints and select best per feature
        transpose : bool
            Whether to transpose the plot (features as rows, pathology as columns)
        custom_pathology_order : List[str], optional
            Custom order for pathology measures
        """
        
        # Select data based on severity
        if severity == 'less_severe':
            rho_df = self.less_severe_rho
            p_df = self.less_severe_p
            severity_label = "Less Severe Lesions"
        else:
            rho_df = self.severe_rho
            p_df = self.severe_p
            severity_label = "Severe Lesions"
        
        # Filter pathology columns and apply custom order if provided
        if pathology_rows is None:
            pathology_rows = self.pathology_columns
        else:
            pathology_rows = [col for col in pathology_rows if col in self.pathology_columns]
        
        # Apply custom pathology order if provided
        if custom_pathology_order is not None:
            # Reorder pathology_rows according to custom_pathology_order
            ordered_pathology = []
            for col in custom_pathology_order:
                if col in pathology_rows:
                    ordered_pathology.append(col)
            # Add any remaining columns not in custom order
            for col in pathology_rows:
                if col not in ordered_pathology:
                    ordered_pathology.append(col)
            pathology_rows = ordered_pathology
        
        # Get features
        all_features = rho_df['feature'].tolist()
        
        # Collapse timepoints if requested
        if collapse_timepoints:
            all_features = self._get_collapsed_features(rho_df, p_df, pathology_rows)
        
        # Filter features
        if include_features is not None:
            features = [f for f in include_features if f in all_features]
        else:
            features = all_features.copy()
        
        if exclude_features is not None:
            features = [f for f in features if f not in exclude_features]
        
        # Categorize and order features by condition-timepoint
        feature_categories = self._categorize_feature(features)
        ordered_features, category_boundaries, pos = [], [], 0
        for category_key, cat_features in feature_categories.items():
            ordered_features.extend(cat_features)
            # Extract the main classification for labeling
            main_class = category_key.split(' - ')[0]
            category_boundaries.append((pos, pos + len(cat_features), main_class))
            pos += len(cat_features)
        
        # Merge adjacent categories with same classification for cleaner boundaries
        merged_boundaries = []
        current_start, current_end, current_class = None, None, None
        
        for start, end, class_name in category_boundaries:
            if current_class == class_name:
                # Extend current boundary
                current_end = end
            else:
                # Save previous boundary if exists
                if current_class is not None:
                    merged_boundaries.append((current_start, current_end, current_class))
                # Start new boundary
                current_start, current_end, current_class = start, end, class_name
        
        # Don't forget the last boundary
        if current_class is not None:
            merged_boundaries.append((current_start, current_end, current_class))
        
        # Prepare data matrix
        data_matrix, mask_matrix = self._prepare_data_matrix(
            rho_df, p_df, pathology_rows, ordered_features, plot_type, debug=debug, transpose=transpose
        )
        
        # Set up figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Clean feature names for display
        display_features = self._clean_feature_names(ordered_features)
        
        # Clean pathology names for display
        display_pathology = [col.replace('TP6_', '').replace('delta_', 'Δ ') for col in pathology_rows]
        
        # Set up colormap and create heatmap
        if plot_type == 'p':
            # Standard p-value plot
            if transpose:
                sns.heatmap(data_matrix, mask=mask_matrix,
                           xticklabels=display_pathology, yticklabels=display_features,
                           cmap='viridis_r', vmin=0, vmax=1,
                           ax=ax, cbar=True, square=False,
                           cbar_kws={'shrink': 0.8, 'aspect': 20})
            else:
                sns.heatmap(data_matrix, mask=mask_matrix,
                           xticklabels=display_features, yticklabels=display_pathology,
                           cmap='viridis_r', vmin=0, vmax=1,
                           ax=ax, cbar=True, square=False,
                           cbar_kws={'shrink': 0.8, 'aspect': 20})
        else:
            # Correlation plot with grey squares for non-significant values
            custom_cmap, custom_norm = self._create_correlation_colormap_with_grey(cmap_colors)
            
            # Use imshow for better control over the colormap
            im = ax.imshow(data_matrix, cmap=custom_cmap, norm=custom_norm, aspect='auto')
            
            # Set up ticks and labels based on transpose
            if transpose:
                # Features as rows (Y-axis), pathology as columns (X-axis)
                ax.set_xticks(range(len(display_pathology)))
                ax.set_yticks(range(len(display_features)))
                ax.set_xticklabels(display_pathology, rotation=45, ha='right')
                ax.set_yticklabels(display_features)
            else:
                # Original: pathology as rows, features as columns
                ax.set_xticks(range(len(display_features)))
                ax.set_yticks(range(len(display_pathology)))
                ax.set_xticklabels(display_features, rotation=90)
                ax.set_yticklabels(display_pathology)
            
            # Add colorbar manually
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Spearman ρ (Grey = p ≥ 0.05)', rotation=270, labelpad=15)
            
            # Set colorbar ticks to only show correlation range
            cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        
        # Add colored Y-axis labels if requested (when transposed)
        if color_ylabels and transpose:
            label_colors = self._get_feature_colors(ordered_features)
            self._apply_colored_ylabels(ax, display_features, label_colors)
        elif color_ylabels and not transpose:
            # Original layout: color X-axis labels
            label_colors = self._get_feature_colors(ordered_features)
            self._apply_colored_xlabels(ax, display_features, label_colors)
        
        # Add grid lines if requested
        if add_grid:
            ax.grid(True, linestyle='-', alpha=0.3, color='white', linewidth=0.5)
            ax.set_axisbelow(True)
        
        # Set title
        if title is None:
            title = f"{severity_label} - {plot_type.upper()}"
        ax.set_title(title, fontsize=14, pad=25)
        
        # Set axis labels based on transpose
        if transpose:
            ax.set_xlabel("Pathology measure", fontsize=12)
            ax.set_ylabel("In vitro feature", fontsize=12)
        else:
            ax.set_xlabel("In vitro feature", fontsize=12)
            ax.set_ylabel("Pathology measure", fontsize=12)
        
        # Add category boundaries and labels based on transpose
        if transpose:
            # Horizontal lines between main classifications (features are rows)
            for start, end, _ in merged_boundaries[:-1]:  # Skip the last boundary
                ax.axhline(y=end-0.5, color="black", linewidth=3, xmin=0, xmax=1)
            
            # Add category labels to the right of the plot
            for start, end, category in merged_boundaries:
                mid_pos = (start + end - 1) / 2
                ax.text(len(pathology_rows) + 0.3, mid_pos, category,
                       ha="left", va="center", fontsize=10, weight="bold", rotation=0)
        else:
            # Original: vertical lines between main classifications (features are columns)
            for start, end, _ in merged_boundaries[:-1]:  # Skip the last boundary
                ax.axvline(x=end-0.5, color="black", linewidth=3, ymin=0, ymax=1)
            
            # Add category labels below the plot
            for start, end, category in merged_boundaries:
                mid_pos = (start + end - 1) / 2
                ax.text(mid_pos, -len(pathology_rows) * 0.15, category,
                       ha="center", va="top", fontsize=10, weight="bold")
        
        plt.tight_layout()
        
        # Save if requested
        if save_to is not None:
            self._save_figure(fig, save_to, dpi=dpi, debug=debug)
        
        return fig, ax
    
    def create_legend(self, 
                     cmap_colors: Tuple[str, str] = ("#d73027", "#4575b4"),
                     figsize: Tuple[float, float] = (8, 1),
                     orientation: str = 'horizontal',
                     include_metric_legend: bool = True,
                     improved_colormap: bool = True,
                     show_grey_square: bool = True,
                     save_to: Optional[str] = None,
                     dpi: int = 300,
                     debug: bool = False) -> plt.Figure:
        """Create a separate legend figure for the correlation heatmaps with optional saving."""
        
        if improved_colormap:
            cmap = self._create_improved_colormap(cmap_colors)
        else:
            colors = [cmap_colors[0], 'grey', cmap_colors[1]]
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
        
        if include_metric_legend:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*2))
        else:
            fig, ax1 = plt.subplots(figsize=figsize)
        
        # Create a dummy heatmap just for the colorbar
        dummy_data = np.linspace(-1, 1, 100).reshape(1, -1)
        
        im = ax1.imshow(dummy_data, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        ax1.set_visible(False)  # Hide the dummy plot
        
        if orientation == 'horizontal':
            cbar = fig.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1, shrink=0.8)
            cbar.set_label('Spearman ρ (Significant correlations)', fontsize=12)
            cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        else:
            cbar = fig.colorbar(im, ax=ax1, orientation='vertical', pad=0.1, shrink=0.8)
            cbar.set_label('Spearman ρ (Significant correlations)', fontsize=12, rotation=270, labelpad=20)
            cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
        
        # Add metric color legend and grey square explanation
        if include_metric_legend:
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            
            # Create metric legend
            legend_elements = []
            for metric, color in self.metric_colors.items():
                if metric != 'default':  # Skip default color
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, 
                                 markersize=10, label=metric, markeredgecolor='black')
                    )
            
            # Add grey square for non-significant correlations
            if show_grey_square:
                legend_elements.append(
                    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#D3D3D3', 
                             markersize=10, label='p ≥ 0.05', markeredgecolor='black')
                )
            
            ax2.legend(handles=legend_elements, loc='center', ncol=len(legend_elements), 
                      title='Metric Types & Significance', title_fontsize=12, fontsize=10)
        
        plt.tight_layout()
        
        # Save if requested
        if save_to is not None:
            self._save_figure(fig, save_to, dpi=dpi, debug=debug)
        
        return fig
    
    def _get_value(self, df: pd.DataFrame, feature: str, column: str):
        """Get a single value from dataframe for given feature and column."""
        if column not in df.columns:
            return None
        
        row = df[df['feature'] == feature]
        if row.empty:
            return None
        
        val = row[column].iloc[0]
        return val if pd.notna(val) else None


def load_correlation_plotter(less_severe_rho_file: str = "less_severe_rho_values_mean.csv",
                           less_severe_p_file: str = "less_severe_p_values_mean.csv", 
                           severe_rho_file: str = "severe_rho_values_mean.csv",
                           severe_p_file: str = "severe_p_values_mean.csv",
                           alpha: float = 0.05) -> CorrelationHeatmapPlotter:
    """Convenience function to load correlation plotter with default files."""
    return CorrelationHeatmapPlotter(
        less_severe_rho_file, less_severe_p_file,
        severe_rho_file, severe_p_file, alpha
    )