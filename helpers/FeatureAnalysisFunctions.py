import pandas as pd  # shap/sklearn imports prefer pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from typing import List, Tuple, Dict, Optional, Union

from helpers.constants import LIDS_DELIMITER, META_DELIMITER


def categorize_features(feature_name: str) -> str:
    """ categorize a feature based on naming convention"""
    
    if feature_name.startswith("TP"):
        return "marmoset"
    elif 'Simple' in feature_name:
        return 'simple PK'
    elif any(delim in feature_name for delim in LIDS_DELIMITER):
        return 'LIDS'
    elif any(delim in feature_name for delim in META_DELIMITER):
        return 'metadata'
    else:
        return 'simple equipotent'


def get_rf_feature_importance(
    model: RandomForestRegressor,
    feature_names: List[str]
) -> pd.DataFrame:
    """extract feature importance from a trianed RF model
    (mean decrease in impurity)

    args:
    -----
    - model: __trained__ scikit-learn rf regressor model
    - feature_names: list of feature names corresponding to model training

    returns:
    -----
    pd.DataFrame with cols ['feature', 'importance'] (sorted desc.)
    """

    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature importances attribute.")
    if len(feature_names) != len(model.feature_importances_):
        raise ValueError(f"Number of feature names ({len(feature_names)}) doesn't correspond to model's number of features.")
    
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    return importance_df

def get_permutation_importance(
    model: RandomForestRegressor,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    feature_names: List[str],
    scoring: str = 'neg_mean_squared_error',
    n_repeats: int = 5,
    random_state: int = 42,
    debug: bool = False
) -> pd.DataFrame:
    """calculates permutation importance for features.
    
    args:
    ------
    - model: trained rf regressor model
    - X_val: validation data features (pd.DataFrame)
    - y_val: validation data targets
    - feature_names: list of feature names
    - scoring: score to use (r2, neg_mean_squared_error)
    - n_repeats: number of times to permutate a feature
    - random_state: random seed
    
    returns:
    ------
    -> pd.DataFrame with ['feature', 'importance_mean', 'importance_std'], sorted descending
    """
    if debug:
        print(f"calculating permutation feature importance... \n")
        print(f"scoring metric: {scoring}")
    
    result = permutation_importance(
        estimator=model,
        X=X_val,
        y=y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std,
    }).sort_values(by='importance_mean', ascending=False).reset_index(drop=True)
    
    if debug:
        print("permutation importance calculations finished!")
        
    return perm_importance_df


def calculate_shap_values(
    model: RandomForestRegressor,
    X_data: pd.DataFrame,
    approximate: bool = False
) -> Tuple[Union[List[np.ndarray], np.ndarray], Optional[shap.TreeExplainer]]:
    """calculate SHAP values using tree explainer
    
    args:
    ------
    model: trained rf regressor model
    X_data: pd.DataFrame sampled for SHAP calculation
    approximate: true/false of whether to use approximate SHAP values
    
    return:
    ------
    -> tuple[shap_values, shap explainer object]
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_data,
                                            approximate=approximate, 
                                            check_additivity=False)
        print(f"SHAP calculation finished.")
        return shap_values, explainer
    except Exception as e:
        print(f"error calculation SHAP values: \n {e}")
        return None, None