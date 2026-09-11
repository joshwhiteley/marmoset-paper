from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from marmoset_paper.analysis.modeling import ModelConfig, train_models
from marmoset_paper.analysis.shap import run_shap

ROOT = Path(__file__).resolve().parents[1]


def test_bundles_preserve_observed_history_sample_identity_and_feature_order(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    counts = train_models(ROOT / "data", model_dir, ModelConfig(n_estimators=5))
    assert counts["b"] == {"train": 212, "test": 49}
    predictions = pd.read_csv(model_dir / "predictions.csv")
    assert len(predictions) == 49 * 8
    bundle = joblib.load(model_dir / "b_tp6.joblib")
    assert bundle["features"][-3:] == ["TP3_MeanHU", "TP4_MeanHU", "TP5_MeanHU"]
    assert bundle["model"].max_depth == 10
    assert bundle["model"].random_state == 42
    source = pd.read_csv(ROOT / "data/marm_data_wide_clustered_classif.csv")
    keys = ["Compound", "MarmID", "Lesion"]
    aligned = bundle["test"][keys].merge(source, on=keys, validate="1:1")
    for tp in [3, 4, 5]:
        np.testing.assert_allclose(bundle["test"][f"TP{tp}_MeanHU"], aligned[f"TP{tp}_MeanHU"])
    rows = predictions[(predictions.strategy == "b") & (predictions.timepoint == 6)]
    np.testing.assert_allclose(
        rows.predicted, bundle["model"].predict(bundle["test"][bundle["features"]])
    )
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    run_shap(model_dir, shap_dir, sample_size=10)
    sample_ids = pd.read_csv(shap_dir / "b_tp6_samples.csv")
    feature_rows = pd.read_csv(shap_dir / "b_tp6_feature_values.csv")
    assert not sample_ids.duplicated(keys).any()
    assert len(feature_rows) == 10 * len(bundle["features"])
    assert not feature_rows.duplicated(keys + ["feature"]).any()
