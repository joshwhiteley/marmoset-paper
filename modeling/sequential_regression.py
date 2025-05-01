"""Sequential regression model for predicting lesion progression.

This module implements a sequential regression approach using Random Forest
to predict lesion progression across multiple timepoints.
"""

import polars as pl
import numpy as np
import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


from data.MarmosetData import MarmosetData
from data.InVitroData import DiamondData
from helpers.PlottingFunctions import (
    plot_lesion_progressions,
    plot_mse_results
)
from helpers.ModelingFunctions import (
    impute_within_compound,
    split_by_compound,
)


def main():
    # Configuration
    MODEL_CONFIG = {
        "diamond_only": False,
        "marm_only": True,
        "test_size": 0.2,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 10,
        "save_figures": True,
        "figure_save_dir": "figures/predictions/marm_only",
        "save_model_results": True,
        "model_results_dir": "modeling/results",
        "figure_save_format": "svg",
        "debug": False
    }
    
    # Set global random seed for reproducibility
    np.random.seed(MODEL_CONFIG["random_state"])
    
    # Load data
    marmoset_data = MarmosetData("data/marm_data_wide_clustered_classif.csv")
    diamond_data = DiamondData("data/in_vitro_diamond_data.csv")
    
    # Get severe lesions data
    severe_lesions = marmoset_data.get_severe_lesions().data
    
    # Print diagnostic info about severe lesions
    print("\nDiagnostic Info:")
    print(f"Number of severe lesions: {len(severe_lesions)}")
    print(f"First few severe lesions Compound values: {severe_lesions['Compound'].head(5).to_list()}")
    print(f"First few severe lesions Lesion values: {severe_lesions['Lesion'].head(5).to_list()}")
    
    # Feature selection
    y_features = ["MeanHU", "MeanSUV"]
    metadata_features = ["Compound", "Lesion"]
    
    # Select target features
    y_cols = [col for col in severe_lesions.columns 
              if any(f in col for f in y_features)]
    severe_data = severe_lesions.select(y_cols + metadata_features)
    
    # Organize timepoint features
    timepoints = {}
    for tp in range(2, 7):
        timepoints[f"tp{tp}"] = [
            col for col in severe_data.columns if f"TP{tp}" in col
        ]
    
    # Prepare in vitro features
    diamond_df = diamond_data.data.filter(
        pl.col('NumbDrugs') > 1
    ).drop('NumbDrugs')
    combo_features = [col for col in diamond_df.columns if "Drug" not in col]
    
    # Merge data
    X = severe_data.join(
        other=diamond_df,
        left_on="Compound",
        right_on="Drug",
        how="inner"
    )
    
    # Print diagnostic info about merged data
    print(f"\nNumber of rows after merge: {len(X)}")
    print(f"First few Compound values after merge: {X['Compound'].head(5).to_list()}")
    print(f"First few Lesion values after merge: {X['Lesion'].head(5).to_list()}")
    
    # Select input features based on configuration
    if MODEL_CONFIG["diamond_only"]:
        base_input_features = combo_features
        print("Using Diamond data only")
    elif MODEL_CONFIG["marm_only"]:
        base_input_features = timepoints["tp2"]
        print("Using Marmoset data only")   
    else:
        base_input_features = timepoints["tp2"] + combo_features
        print("Using combined Diamond and Marmoset data")
    
    # Prepare data
    X_imputed = impute_within_compound(
        X,
        columns=base_input_features,
        grouping_column='Compound'
    )
    
    # Split data
    train, test = split_by_compound(
        X_imputed,
        grouping_column='Compound',
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    if MODEL_CONFIG["debug"]:
        print(f"\nTrain/Test Split Info:")
        print(f"Number of train samples: {len(train)}")
        print(f"Number of test samples: {len(test)}")
        print(f"Train Compounds: {sorted(train['Compound'].unique().to_list())}")
        print(f"Test Compounds: {sorted(test['Compound'].unique().to_list())}")
    
    # Initialize models
    models = {}
    for tp_num in range(3, 7):
        tp_key = f"tp{tp_num}"
        models[tp_key] = RandomForestRegressor(
            n_estimators=MODEL_CONFIG["n_estimators"],
            max_depth=MODEL_CONFIG["max_depth"],
            random_state=MODEL_CONFIG["random_state"],
            n_jobs=1
        )
    
    # Train models sequentially
    for tp_num in range(3, 7):
        tp_key = f"tp{tp_num}"
        model = models[tp_key]
        
        # Collect features for this timepoint
        features = base_input_features.copy()
        for prev_tp in range(3, tp_num):
            features += timepoints[f"tp{prev_tp}"]
        
        # Train model
        model.fit(
            train.select(features),
            train.select(timepoints[tp_key])
        )
    
    # Initialize prediction dataframe
    test_pred_df = pl.DataFrame()
    test_pred_df = test_pred_df.with_columns(
        test["Compound"],
        test["Lesion"]
    )
    
    # Make predictions
    for tp_num in range(3, 7):
        tp_key = f"tp{tp_num}"
        model = models[tp_key]
        features = base_input_features.copy()
        
        # Add previous timepoint features
        for prev_tp in range(3, tp_num):
            features += timepoints[f"tp{prev_tp}"]
        
        # Get predictions
        predictions = model.predict(test.select(features))
        
        # Add predictions to dataframe
        for i, col_name in enumerate(timepoints[tp_key]):
            test_pred_df = test_pred_df.with_columns(
                pl.Series(col_name, predictions[:, i])
            )
    
    # Calculate MSE for each timepoint
    mse_results = []
    for tp_num in range(3, 7):
        tp_key = f"tp{tp_num}"
        mse = mean_squared_error(
            test_pred_df.select(timepoints[tp_key]),
            test.select(timepoints[tp_key])
        )
        mse_results.append({
            "Timepoint": f"TP{tp_num}",
            "MSE": mse
        })
    
    # create, display, and save MSE + models
    mse_df = pl.DataFrame(mse_results)
    print("\nMean Squared Error by Timepoint:")
    print(mse_df)
    
    mse_prefix = ""
    if MODEL_CONFIG["diamond_only"]:
        mse_prefix = "d"
    elif MODEL_CONFIG["marm_only"]:
        mse_prefix = "m"
    elif ~MODEL_CONFIG["diamond_only"] & ~MODEL_CONFIG["marm_only"]:
        mse_prefix = "b"

    now = datetime.datetime.now().strftime("%Y%m%d")
    
    if MODEL_CONFIG["save_model_results"]:
        model_save_path = Path(MODEL_CONFIG["model_results_dir"])
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        mse_df.write_csv(f"{now}_{mse_prefix}_{MODEL_CONFIG['random_state']}_MSE.csv")
        
        #TODO: SAVE MODELS
    
    plot_mse_results(
        mse_df=mse_df,
        save_path=MODEL_CONFIG["model_results_dir"],
        save_prefix=mse_prefix,
        save_format=MODEL_CONFIG["figure_save_format"],
        save_bool=True
    )
    
    # generate and save plots for each compound
    compounds = test_pred_df["Compound"].unique().to_list()
    for compound in compounds:
        compound_test = test.filter(pl.col("Compound") == compound)
        compound_pred = test_pred_df.filter(pl.col("Compound") == compound)
        
        plot_lesion_progressions(
            compound=compound,
            num_lesions=len(compound_test),
            test_df=compound_test,
            pred_df=compound_pred,
            save_dir=MODEL_CONFIG["figure_save_dir"],
            save_bool=MODEL_CONFIG["save_figures"],
            random_state=MODEL_CONFIG["random_state"]
        )


if __name__ == "__main__":
    main()
