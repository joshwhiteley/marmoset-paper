from dataclasses import asdict
from pathlib import Path
from shutil import copytree

import joblib
import numpy as np
import pandas as pd
import pytest

from marmoset_paper.analysis.modeling import ModelConfig, train_models
from marmoset_paper.analysis.shap import load_combined_models, run_shap

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["Compound", "MarmID", "Lesion"]


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    directory = tmp_path_factory.mktemp("models")
    train_models(ROOT / "data", directory, ModelConfig(n_estimators=5))
    return directory


def test_publication_defaults_are_explicit():
    assert asdict(ModelConfig()) == {
        "seed": 42,
        "test_size": 0.2,
        "n_estimators": 100,
        "max_depth": 10,
    }


def test_all_bundles_preserve_observed_history_and_identical_strategy_splits(models, tmp_path):
    predictions = pd.read_csv(models / "predictions.csv")
    assignments = pd.read_csv(models / "split_assignments.csv")
    assert len(predictions) == 49 * 8
    pd.testing.assert_frame_equal(
        assignments[assignments.strategy == "m"].drop(columns="strategy").reset_index(drop=True),
        assignments[assignments.strategy == "b"].drop(columns="strategy").reset_index(drop=True),
    )
    source = pd.read_csv(ROOT / "data/marm_data_wide_clustered_classif.csv")
    invitro = pd.read_csv(ROOT / "data/in_vitro_diamond_data.csv", nrows=0)
    invitro_features = [column for column in invitro if column not in {"Drug", "NumbDrugs"}]
    for strategy in ["m", "b"]:
        baseline = ["TP2_MeanHU"] + (invitro_features if strategy == "b" else [])
        for tp in range(3, 7):
            bundle = joblib.load(models / f"{strategy}_tp{tp}.joblib")
            assert (len(bundle["train"]), len(bundle["test"])) == (212, 49)
            assert bundle["features"] == baseline + [
                f"TP{previous}_MeanHU" for previous in range(3, tp)
            ]
            assert bundle["model"].max_depth == 10
            assert bundle["model"].random_state == 42
            aligned = bundle["test"][KEYS].merge(source, on=KEYS, validate="1:1")
            for previous in range(2, tp):
                np.testing.assert_allclose(
                    bundle["test"][f"TP{previous}_MeanHU"], aligned[f"TP{previous}_MeanHU"]
                )
            rows = predictions[(predictions.strategy == strategy) & (predictions.timepoint == tp)]
            np.testing.assert_allclose(
                rows.predicted, bundle["model"].predict(bundle["test"][bundle["features"]])
            )
    shap_dir = tmp_path / "shap"
    shap_dir.mkdir()
    run_shap(models, shap_dir, sample_size=10)
    sample_ids = pd.read_csv(shap_dir / "b_tp6_samples.csv")
    feature_rows = pd.read_csv(shap_dir / "b_tp6_feature_values.csv")
    assert not sample_ids.duplicated(KEYS).any()
    assert len(feature_rows) == 10 * (len(invitro_features) + 4)
    assert not feature_rows.duplicated(KEYS + ["feature"]).any()


@pytest.mark.parametrize(
    "change,match",
    [
        ("strategy", "Unexpected combined"),
        ("config", "configuration"),
        ("identity", "lesion identities"),
        ("measurement", "shared train measurements"),
    ],
)
def test_shap_rejects_inconsistent_bundles(models, tmp_path, change, match):
    directory = tmp_path / "copied-models"
    copytree(models, directory)
    path = directory / "b_tp4.joblib"
    bundle = joblib.load(path)
    if change == "strategy":
        bundle["strategy"] = "m"
    elif change == "config":
        bundle["config"]["seed"] = 7
    elif change == "identity":
        bundle["train"] = bundle["train"].iloc[::-1]
    else:
        bundle["train"].iloc[0, bundle["train"].columns.get_loc("TP2_MeanHU")] += 1
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match=match):
        load_combined_models(directory)
