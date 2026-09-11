"""Random-forest feature importance summaries."""

import pandas as pd
from sklearn.inspection import permutation_importance


def get_rf_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return mean-decrease-in-impurity importances in descending order."""
    if len(feature_names) != len(model.feature_importances_):
        raise ValueError("Feature names do not match the fitted model")
    return pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)


def get_permutation_importance(
    model,
    X_val,
    y_val,
    feature_names: list[str],
    scoring: str = "neg_mean_squared_error",
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure importance on held-out rows using repeated feature permutations."""
    if list(X_val.columns) != feature_names:
        raise ValueError("Permutation feature names do not match the input columns")
    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)
