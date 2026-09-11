import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from marmoset_paper.helpers.ShapFunctions import (
    calculate_shap_values,
    export_feature_data,
    select_output,
)


@pytest.mark.parametrize("classifier", [False, True])
def test_tree_shap_additivity_and_output_selection(classifier):
    X = pd.DataFrame({"a": range(12), "b": [0, 1] * 6})
    y = (X.a > 5).astype(int) if classifier else X.a * 2 + X.b
    cls = RandomForestClassifier if classifier else RandomForestRegressor
    model = cls(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
    values, explainer = calculate_shap_values(model, X)
    index = 1 if classifier else 0
    selected = select_output(values, index)
    expected = model.predict_proba(X)[:, index] if classifier else model.predict(X)
    baseline = np.asarray(explainer.expected_value).ravel()[index]
    np.testing.assert_allclose(selected.sum(axis=1) + baseline, expected, atol=1e-12)
    with pytest.raises(ValueError, match="column order"):
        calculate_shap_values(model, X[["b", "a"]])


def test_repeated_feature_values_do_not_duplicate_sample_identity(tmp_path):
    features = pd.DataFrame({"a": [1, 1]}, index=[8, 3])
    metadata = pd.DataFrame({"MarmID": ["M1", "M2"], "Lesion": [1, 1]}, index=[8, 3])
    destination = tmp_path / "features.csv"
    export_feature_data(np.array([[2], [3]]), features, metadata, destination)
    result = pd.read_csv(destination)
    assert result.MarmID.tolist() == ["M1", "M2"]
    assert result.shap_value.tolist() == [2, 3]
    with pytest.raises(ValueError, match="aligned"):
        export_feature_data(np.zeros((2, 1)), features, metadata.iloc[::-1], destination)


@pytest.mark.parametrize(
    "values,index", [(np.zeros((3, 2)), 1), (np.zeros(3), 0), (np.zeros((3, 2, 2)), -1)]
)
def test_invalid_shap_output(values, index):
    with pytest.raises(ValueError, match="Cannot select"):
        select_output(values, index)
