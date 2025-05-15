import warnings
import os
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel

import helpers.constants as C
import helpers.FeatureAnalysisFunctions as FAF
import helpers.ModelingFunctions as MF
import helpers.PlottingFunctions as PF
import sequential_regression as SR
from data import MarmosetData

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# -------------------------
# Load and pre-process data
# -------------------------

marm_df = MarmosetData("data/marm_data_wide_clustered_classif.csv")
marm_df = marm_df.get_severe_lesions().data

invitro_df = pl.read_csv("data/in_vitro_diamond_data.csv")
invitro_df = invitro_df.filter(pl.col('NumbDrugs') > 1).drop('NumbDrugs')

merged_df = marm_df.merge(
    other=invitro_df,
    left_on="Compound",
    right_on="Drug",
    how="inner"
)

train_idx, test_idx = MF.make_stratified_split(merged_df, strat_col="Drug", test_size=0.20, random_state=42)
train_df,  test_df = merged_df.loc[train_idx].reset_index(drop=True), merged_df.loc[test_idx].reset_index(drop=True)

# ------------
# Feature Sets
# ------------

marm_feats = [f for f in merged_df.columns if f.startswith("TP")]
