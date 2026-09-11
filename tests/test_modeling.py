from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from marmoset_paper.analysis.modeling import IDENTIFIERS, prepare_model_data
from marmoset_paper.helpers.ModelingFunctions import impute_within_compound, split_by_compound

ROOT = Path(__file__).resolve().parents[1]


def test_split_is_seeded_disjoint_and_does_not_change_global_rng():
    frame = pl.DataFrame({"Compound": ["B"] * 10 + ["A"] * 10, "id": range(20)})
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    train, test = split_by_compound(frame)
    assert np.random.random() == expected_next
    assert len(train) == 16 and len(test) == 4
    assert set(train["id"]).isdisjoint(test["id"])
    assert set(train["id"]) | set(test["id"]) == set(range(20))
    assert train["Compound"].unique().sort().to_list() == ["A", "B"]
    assert split_by_compound(frame)[1].equals(test)
    assert not split_by_compound(frame, random_state=43)[1].equals(test)


def test_split_is_stable_across_separate_processes():
    import subprocess
    import sys

    code = """
import polars as pl
from marmoset_paper.helpers.ModelingFunctions import split_by_compound
frame = pl.DataFrame({'Compound': ['B'] * 10 + ['A'] * 10, 'id': range(20)})
print(split_by_compound(frame)[1]['id'].to_list())
"""
    results = [subprocess.check_output([sys.executable, "-c", code]) for _ in range(3)]
    assert len(set(results)) == 1


@pytest.mark.parametrize("size", [0, 1, -0.1, 1.1])
def test_invalid_split_size(size):
    with pytest.raises(ValueError, match="test_size"):
        split_by_compound(pl.DataFrame({"Compound": ["A"]}), test_size=size)


def test_imputation_preserves_identity_and_all_missing_groups():
    frame = pl.DataFrame(
        {"Compound": ["A", "B", "A", "B"], "id": [4, 3, 2, 1], "x": [2.0, None, None, None]}
    )
    result = impute_within_compound(frame, ["x"])
    assert result["id"].to_list() == [4, 3, 2, 1]
    assert result["x"].to_list() == [2.0, None, 2.0, None]


def test_deposited_cohort_matches_updated_methods():
    data, features, counts = prepare_model_data(ROOT / "data")
    train, test = split_by_compound(data)
    assert (len(train), len(test), len(features)) == (212, 49, 104)
    assert counts["severe_lesions"] == 566
    assert len(counts["modeled_regimens"]) == 10
    assert not data.select(IDENTIFIERS).is_duplicated().any()
    # This is intentionally not a regimen holdout.
    assert set(train["Compound"]) == set(test["Compound"])
    assert set(train["MarmID"]) & set(test["MarmID"])


def test_imputation_does_not_change_the_deposited_modeling_measurements():
    data, invitro, _ = prepare_model_data(ROOT / "data")
    for features in [["TP2_MeanHU"], ["TP2_MeanHU"] + invitro]:
        original = data.select(features)
        imputed = impute_within_compound(data, features).select(features)
        assert original.null_count().row(0) == imputed.null_count().row(0)
        np.testing.assert_array_equal(original.to_numpy(), imputed.to_numpy())


def test_duplicate_regimen_join_is_rejected(tmp_path):
    for name in ["marm_data_wide_clustered_classif.csv", "in_vitro_diamond_data.csv"]:
        frame = pd.read_csv(ROOT / "data" / name)
        if name.startswith("in_vitro"):
            frame = pd.concat([frame, frame.iloc[[0]]])
        frame.to_csv(tmp_path / name, index=False)
    with pytest.raises(pl.exceptions.ComputeError, match="m:1"):
        prepare_model_data(tmp_path)
