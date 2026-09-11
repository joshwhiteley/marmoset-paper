from marmoset_paper.helpers.ShapFunctions import calculate_shap_values
"""
Sequential regression model for predicting lesion progression.

This module implements a sequential regression approach using Random Forest
to predict lesion progression across multiple timepoints.
"""

import polars as pl
import numpy as np
import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import joblib
import warnings

# Suppress FutureWarnings from SHAP and other libraries
warnings.filterwarnings('ignore', category=FutureWarning)


from marmoset_paper.data.MarmosetData import MarmosetData
from marmoset_paper.data.InVitroData import DiamondData

from legacy_plotting import (
    plot_lesion_progressions,
    plot_performance_metrics,
    plot_feature_importance_bars,
    plot_shap_summary_dot
)
from marmoset_paper.helpers.ModelingFunctions import (
    impute_within_compound,
    split_by_compound,
)
from marmoset_paper.helpers.FeatureAnalysisFunctions import (
    get_rf_feature_importance,
    get_permutation_importance
)


def main():
    MODEL_CONFIG = {
        # data setup
        "diamond_only": False,                                  # bool
        "marm_only": False,                                     # bool
        "test_size": 0.2,                                       # float

        # model setup
        "random_state": 42,                                     # int
        "n_estimators": 100,                                    # int
        "max_depth": 10,                                        # int

        # output
        "save_models": True,                                    # bool
        "save_model_results": True,                             # bool
        "model_results_dir": "radiodensity-only/modeling/results",                # str [dir]
        "save_model_dir": "radiodensity-only/modeling/models/",                   # str [dir]

        # other
        "debug": False                                          # bool
    }
    
    ANALYSIS_CONFIG = {
        "save_figures": True,                                   # bool
        "perform_feature_analysis": True,                       # bool
        "figure_save_dir": "radiodensity-only/figures/predictions/",              # str [dir]
        "figure_save_format": "svg",                            # str [png, jpeg, svg, ...]
        "feature_analysis_dir": "radiodensity-only/modeling/feature_analysis",    # str [dir]
        "shap_sample_size": 1000,                               # int
    }
    
    # ------------------------------
    # Data Initialization
    # ------------------------------
    
    # global random seed
    np.random.seed(MODEL_CONFIG["random_state"])
    
    # data loading
    marmoset_data = MarmosetData("data/marm_data_wide_clustered_classif.csv")
    diamond_data = DiamondData("data/in_vitro_diamond_data.csv")
    
    # severe lesions only
    severe_lesions = marmoset_data.get_severe_lesions().data
    
    # model typing
    model_prefix = ""
    if MODEL_CONFIG["diamond_only"]:
        model_prefix = "d"
    elif MODEL_CONFIG["marm_only"]:
        model_prefix = "m"
    else:
        model_prefix = "b"

    now = datetime.datetime.now().strftime("%Y%m%d")
    
    if MODEL_CONFIG["debug"]:
        print("\nDiagnostic Info:")
        print(f"Number of severe lesions: {len(severe_lesions)}")
        print(f"First few severe lesions Compound values: {severe_lesions['Compound'].head(5).to_list()}")
        print(f"First few severe lesions Lesion values: {severe_lesions['Lesion'].head(5).to_list()}")
    
    # feature selection
    y_features = ["MeanHU"]
    metadata_features = ["Compound", "Lesion"]
    
    # feature formatting
    y_cols = [col for col in severe_lesions.columns
              if any(f in col for f in y_features)]
    severe_data = severe_lesions.select(y_cols + metadata_features)

    # timepoint organization (MeanHU only)
    timepoints = {
        f"tp{tp}": [col for col in severe_data.columns if f"TP{tp}" in col and "MeanHU" in col]
        for tp in range(2, 7)
    }

    # in vitro feature preparation
    diamond_df = diamond_data.data.filter(
        pl.col('NumbDrugs') > 1
    ).filter(
        ~pl.col('Drug').str.contains('QBS') # these drugs are missing too many measurements!
    ).drop('NumbDrugs')
    combo_features = [col for col in diamond_df.columns if "Drug" not in col]

    # create merged dataframe
    X = severe_data.join(
        other=diamond_df,
        left_on="Compound",
        right_on="Drug",
        how="inner"
    )
    
    # diagnostics output
    if MODEL_CONFIG["debug"]:
        print(f"\nNumber of rows after merge: {len(X)}")
        print(f"First few Compound values after merge: {X['Compound'].head(5).to_list()}")
        print(f"First few Lesion values after merge: {X['Lesion'].head(5).to_list()}")
    
    # more diagnostics output
    print("\n----------")
    if MODEL_CONFIG["diamond_only"]:
        base_input_features = combo_features
        print("Using Diamond data only")
    elif MODEL_CONFIG["marm_only"]:
        base_input_features = timepoints["tp2"]
        print("Using Marmoset data only")   
    else:
        base_input_features = timepoints["tp2"] + combo_features
        print("Using combined Diamond and Marmoset data")
    print("----------")
    
    # imputation of holey data
    X_imputed = impute_within_compound(
        X,
        columns=base_input_features,
        grouping_column='Compound'
    )
    
    # train test splitting
    train, test = split_by_compound(
        X_imputed,
        grouping_column='Compound',
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"]
    )
    
    # train/test split diagnostics
    if MODEL_CONFIG["debug"]:
        print(f"\nTrain/Test Split Info:")
        print(f"Number of train samples: {len(train)}")
        print(f"Number of test samples: {len(test)}")
        print(f"Train Compounds: {sorted(train['Compound'].unique().to_list())}")
        print(f"Test Compounds: {sorted(test['Compound'].unique().to_list())}")
    
    # ------------------------------
    # Modeling
    # ------------------------------

    # initialize models
    models = {}
    feature_names_per_model = {}
    
    if MODEL_CONFIG["save_models"]:
        model_save_path = Path(MODEL_CONFIG["save_model_dir"])
        model_save_path.mkdir(parents=True, exist_ok=True)
    
    print("\n----------\nTraining models...\n")
    all_training_features = base_input_features.copy()
    
    # creation of each TP model
    for tp_num in range(3, 7):
        tp_key = f"tp{tp_num}"
        model = RandomForestRegressor(
            n_estimators=MODEL_CONFIG["n_estimators"],
            max_depth=MODEL_CONFIG["max_depth"],
            random_state=MODEL_CONFIG["random_state"],
            n_jobs=-1
        )

        # features for a specific timepoints training
        features_for_training = base_input_features.copy()
        for prev_tp_num in range(3, tp_num):
            prev_tp_key = f"tp{prev_tp_num}"
            features_for_training.extend(timepoints[prev_tp_key])

        feature_names_per_model[tp_key] = features_for_training
        all_training_features.extend([f for f in features_for_training if f not in all_training_features])

        if MODEL_CONFIG["debug"]:
            print(f"- Training model for {tp_key} using {len(features_for_training)} features.")

        target_col = timepoints[tp_key][0]  # Single MeanHU column per timepoint

        missing_train_features = [f for f in features_for_training if f not in train.columns]
        if target_col not in train.columns:
            print(f"Error: Missing target column {target_col} in training data for {tp_key}.")
            return
        if missing_train_features:
            print(f"Error: Missing feature columns in training data for {tp_key}.")
            return

        # data preparation
        X_train_df = train.select(features_for_training)
        y_train_df = train.select([target_col]).to_numpy().flatten()
        
        # initialize predictions df
        test_pred_df = pl.DataFrame()
        test_pred_df = test_pred_df.with_columns(
            test["Compound"],
            test["Lesion"]
        )

        # model fitting
        try:
            model.fit(X_train_df, y_train_df)
            models[tp_key] = model

            # save models
            if MODEL_CONFIG["save_models"]:
                model_filename = f"model_{tp_key}_{model_prefix}_seed{MODEL_CONFIG['random_state']}_{now}.joblib"
                full_model_path = model_save_path / model_filename
                joblib.dump(model, full_model_path)
                if MODEL_CONFIG["debug"]:
                    print(f"Model for {tp_key} saved to: {full_model_path}")

        except Exception as e:
            print(f"Error during model fitting for {tp_key}: {e}")
            return

    print("..........\nModel training completed.\n----------")
    
    # ------------------------------
    # Model Predictions
    # ------------------------------

    print("\n----------\nBeginning predictions...\n")
    for tp_num in range(3, 7):
        tp_key = f"tp{tp_num}"
        model = models[tp_key]

        features = base_input_features + [
            feature
            for prev_tp in range(3, tp_num)
            for feature in timepoints[f"tp{prev_tp}"]
        ]

        # predict (single output per timepoint)
        predictions = model.predict(test.select(features))
        target_col = timepoints[tp_key][0]

        # append to predictions df
        test_pred_df = test_pred_df.with_columns(
            pl.Series(target_col, predictions)
        )
    
    print("..........\nPredictions completed.\n----------")
    
    # ------------------------------
    # Model Performance Analysis
    # ------------------------------

    # calculation of performance metrics for MeanHU predictions
    metrics_results = []
    for tp_num in range(3, 7):
        tp_key = f"tp{tp_num}"
        output_name = timepoints[tp_key][0]  # Single MeanHU column per timepoint

        y_true = test.select(output_name).to_numpy().flatten()
        y_pred = test_pred_df.select(output_name).to_numpy().flatten()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        metrics_results.append({
            "timepoint": tp_num,
            "output": output_name,
            "mse": mse,
            "mae": mae,
            "r2": r2
        })
    
    print("Error metrics by timepoint:\n")
    metrics_df = pl.DataFrame(metrics_results)
    print(metrics_df)
            
    if MODEL_CONFIG["save_model_results"]:
        model_save_path = Path(MODEL_CONFIG["model_results_dir"])
        model_save_path.mkdir(parents=True, exist_ok=True)
        
        metrics_df.write_csv(model_save_path / f"{now}_{model_prefix}_{MODEL_CONFIG['random_state']}_performance.csv")
        
    outputs = metrics_df['output'].unique().to_list()
    plot_performance_metrics(
        metrics_df=metrics_df,
        save_path=MODEL_CONFIG["model_results_dir"],
        outputs=outputs,
        metrics=["mse", "mae", "r2"],
        save_prefix=model_prefix,
        save_format=ANALYSIS_CONFIG["figure_save_format"],
        save_bool=True
    )
    
    # ------------------------------
    # Lesion Progression Plotting
    # ------------------------------
    
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
            save_dir=ANALYSIS_CONFIG["figure_save_dir"],
            save_bool=ANALYSIS_CONFIG["save_figures"],
            random_state=MODEL_CONFIG["random_state"],
            model_prefix=model_prefix,
            output_format=ANALYSIS_CONFIG["figure_save_format"]
        )
    
    # ------------------------------
    # Feature Importance Analysis
    # ------------------------------

    if ANALYSIS_CONFIG["perform_feature_analysis"]:
        print("\n----------\nBeginning feature analysis...\n----------\n")
        all_importance_results = {}
        analysis_save_path = Path(ANALYSIS_CONFIG["feature_analysis_dir"])
        analysis_save_path.mkdir(parents=True, exist_ok=True)

        for tp_key, model in models.items():
            if MODEL_CONFIG["debug"]: 
                print(f"Analyzing features for model: {tp_key.upper()}")
            current_features = feature_names_per_model[tp_key]
            
            # Random Forest (MDI) Importance
            try:
                csv_path = analysis_save_path / f"{model_prefix}_{tp_key}_feature_importance_MDI_{now}.csv"
                rf_importance_df = get_rf_feature_importance(model, current_features)
                all_importance_results[f"{tp_key}_MDI"] = rf_importance_df
                rf_importance_df.to_csv(csv_path, index=False)
                plot_feature_importance_bars(
                    importance_df=rf_importance_df,
                    tp_key=tp_key,
                    save_path=analysis_save_path,
                    importance_col='importance',
                    top_n=10,
                    title_suffix='MDI',
                    save_bool=True,
                    save_format=ANALYSIS_CONFIG["figure_save_format"],
                    model_prefix=model_prefix
                )
            except Exception as e:
                print(f"Error during RF importance calculation/plotting for {tp_key}: {e}")
                
            # permutation importance
            try:
                target_col = timepoints[tp_key][0]
                test_non_null = test.drop_nulls(subset=current_features + [target_col])
                X_val = test_non_null.select(current_features).to_pandas()
                y_val = test_non_null.select([target_col]).to_pandas().values.flatten()
                perm_importance_df = get_permutation_importance(
                    model=model,
                    X_val=X_val,
                    y_val=y_val,
                    feature_names=current_features,
                    scoring='neg_mean_squared_error',
                    n_repeats=5,
                    random_state=MODEL_CONFIG["random_state"]
                )
                perm_importance_df.to_csv(analysis_save_path / f"{model_prefix}_{tp_key}_feature_permutation_importance_{now}.csv", index=False)
                plot_feature_importance_bars(
                    importance_df=perm_importance_df,
                    tp_key=tp_key,
                    save_path=analysis_save_path,
                    importance_col='importance_mean',
                    title_suffix='permutation',
                    save_bool=True,
                    save_format=ANALYSIS_CONFIG["figure_save_format"],
                    model_prefix=model_prefix
                )
            except Exception as e:
                print(f"Error during permutation importance for {tp_key}: {e}")
                
            # SHAP analysis
            try:
                n_samples = min(1000, len(train))
                X_train_sample_pd = (
                    train.sample(
                        n=n_samples,
                        with_replacement=False,
                        seed=MODEL_CONFIG["random_state"]
                    )
                    .select(current_features)
                    .to_pandas()
                )

                shap_values, explainer = calculate_shap_values(
                    model=model,
                    X_data=X_train_sample_pd,
                    approximate=False
                )

                output_name = timepoints[tp_key][0]  # Single MeanHU output

                if MODEL_CONFIG["debug"]:
                    print(f"shap_values type: {type(shap_values)}")
                    if isinstance(shap_values, np.ndarray):
                        print(f"shap_values shape: {shap_values.shape}")
                    print(f"output_name: {output_name}")

                # For single output regression, shap_values should be 2D
                plot_shap_summary_dot(
                    shap_values=shap_values,
                    X_data_df=X_train_sample_pd,
                    tp_key=tp_key,
                    output_name=output_name,
                    save_path=analysis_save_path,
                    save_bool=True,
                    save_format=ANALYSIS_CONFIG["figure_save_format"],
                    save_shap_df=True,
                    model_prefix=model_prefix
                )
            except Exception as e:
                print(f"Error during SHAP analysis/plotting for {tp_key}: {e}")


if __name__ == "__main__":
    main()
