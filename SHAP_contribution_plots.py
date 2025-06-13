import os
from pathlib import Path
import datetime
import polars as pl
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from data.MarmosetData import MarmosetData
from data.InVitroData import DiamondData
from helpers.ModelingFunctions import impute_within_compound, split_by_compound
from helpers.FeatureAnalysisFunctions import calculate_shap_values

# ----------------------
# CONFIGURATION
# ----------------------
# Choose which timepoint model to analyze: tp3, tp4, tp5, tp6
TP_KEY = "tp6"
# Filter phrases for in-vitro features (e.g., ["AcidicN","AcidicH"]), or [] for all non-TP features
FILTER_PHRASES = []
# Exclude specific feature names entirely (e.g., ["featA","featB"])
EXCLUDE_FEATURES = []
# Directory containing saved models (pattern: model_{tp_key}_*.joblib)
MODEL_DIR = Path("modeling/models")
# Data directory
DATA_DIR = Path("data")
# Output directory for plots
OUTPUT_DIR = Path("shap_contrib_plots")
# Max SHAP samples
SHAP_SAMPLE_SIZE = 1000

# Create output dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for filenames
NOW = datetime.datetime.now().strftime("%Y%m%d")

# ----------------------
# LOAD AND PREPARE DATA
# ----------------------
marmoset_data = MarmosetData(str(DATA_DIR / "marm_data_wide_clustered_classif.csv"))
diamond_data = DiamondData(str(DATA_DIR / "in_vitro_diamond_data.csv"))

# Select severe lesions and target columns
sev = marmoset_data.get_severe_lesions().data
y_feats = ["MeanHU", "MeanSUV"]
meta_feats = ["Compound", "Lesion"]
y_cols = [c for c in sev.columns if any(f in c for f in y_feats)]
sev_df = sev.select(y_cols + meta_feats)

# Prepare in-vitro data
diamond_df = (
    diamond_data.data
    .filter(pl.col("NumbDrugs") > 1)
    .drop("NumbDrugs")
)
combo_feats = [c for c in diamond_df.columns if "Drug" not in c]

# Merge
X = sev_df.join(other=diamond_df, left_on="Compound", right_on="Drug", how="inner")

# Timepoint feature dict
timepoints = {f"tp{tp}": [c for c in sev_df.columns if f"TP{tp}" in c] for tp in range(2, 7)}

# Base features for modeling
base_feats = timepoints["tp2"] + combo_feats

# Imputation and split
X_imp = impute_within_compound(X, columns=base_feats, grouping_column="Compound")
train_pl, _ = split_by_compound(X_imp, grouping_column="Compound", test_size=0.2, random_state=42)
train_df = train_pl.to_pandas().reset_index(drop=True)

# ----------------------
# CONFIGURE FEATURES FOR TP_MODEL
# ----------------------
tp_num = int(TP_KEY.replace("tp", ""))
all_feats = base_feats.copy()
for prev in range(3, tp_num):
    all_feats += timepoints[f"tp{prev}"]

# ----------------------
# LOAD MODEL
# ----------------------
model_files = list(MODEL_DIR.glob(f"*_{TP_KEY}_*.joblib"))
if not model_files:
    raise FileNotFoundError(f"No model file found for {TP_KEY}")
model = joblib.load(model_files[0])

# ----------------------
# SAMPLE DATA FOR SHAP
# ----------------------
samp_df = train_df.sample(n=min(SHAP_SAMPLE_SIZE, len(train_df)), random_state=42)
X_shap = samp_df[all_feats]

# ----------------------
# CALCULATE SHAP VALUES
# ----------------------
shap_values, _ = calculate_shap_values(model=model, X_data=X_shap, approximate=False)
if shap_values is None:
    raise RuntimeError("SHAP calculation failed")
outs = timepoints[TP_KEY]

# ----------------------
# PLOTTING
# ----------------------
for target_output in outs:
    output_idx = outs.index(target_output)
    sv2d = shap_values[:, :, output_idx]

    # Filter in-vitro features by phrases & exclusion list
    candidates = [f for f in all_feats if not f.startswith("TP")]
    if FILTER_PHRASES:
        candidates = [f for f in candidates if any(ph in f for ph in FILTER_PHRASES)]
    invitro_feats = [f for f in candidates if f not in EXCLUDE_FEATURES]
    idx_feats = [all_feats.index(f) for f in invitro_feats]

    # Compute per-sample SHAP sum
    group_shap = sv2d[:, idx_feats].sum(axis=1)
    df_inv = pd.DataFrame({"Compound": samp_df["Compound"].values, "invitro_SHAP": group_shap})

    # Summary sorted ascending (best first)
    summary = df_inv.groupby("Compound")["invitro_SHAP"].agg(["mean","median","std","count"]).reset_index()
    summary = summary.sort_values("median", ascending=True)
    summary.to_csv(OUTPUT_DIR / f"invitro_shap_summary_{TP_KEY}_{target_output}_{NOW}.csv", index=False)

    # Box-and-whisker plot ordered by median
    order = summary["Compound"].tolist()
    plt.figure(figsize=(10,5))
    sns.boxplot(
        data=df_inv, x="Compound", y="invitro_SHAP", order=order,
        showfliers=True, boxprops=dict(linewidth=1.2), whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2), flierprops=dict(marker='o', markersize=4, alpha=0.6),
        medianprops=dict(color='firebrick', linewidth=2), palette='vlag'
    )
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Compound')
    plt.ylabel(f'In-vitro SHAP ({target_output})\n(negative = improvement)')
    plt.title(f'In-vitro SHAP per Compound (best first) — {target_output}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"invitro_shap_boxwhisker_{TP_KEY}_{target_output}_{NOW}.png", dpi=150)
    plt.close()

print("Plots saved to", OUTPUT_DIR)
