import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from marmoset_paper.analysis.correlations import paired_correlations
from marmoset_paper.helpers.CorrelationPlottingFunctions import select_phases


def test_correlations_align_by_regimen_and_count_finite_pairs():
    invitro = pd.DataFrame({"f": [1, 2, 3, 4, np.inf]}, index=list("ABCDE"))
    pathology = pd.DataFrame({"outcome": [8, 1, 3, 2, 4]}, index=list("EADBC"))
    result = paired_correlations(invitro, pathology)
    expected = spearmanr([1, 2, 3, 4], [1, 2, 4, 3])
    assert result["n_pairs"].loc["f", "outcome"] == 4
    assert result["rho_values"].loc["f", "outcome"] == pytest.approx(expected.statistic)
    assert result["p_values"].loc["f", "outcome"] == pytest.approx(expected.pvalue)


def test_constant_or_small_pairs_remain_unestimated():
    result = paired_correlations(pd.DataFrame({"f": [1, 1]}), pd.DataFrame({"y": [2, 3]}))
    assert result["n_pairs"].iloc[0, 0] == 2
    assert np.isnan(result["rho_values"].iloc[0, 0])


def test_duplicate_regimen_is_rejected():
    with pytest.raises(ValueError, match="one row per regimen"):
        paired_correlations(
            pd.DataFrame({"f": [1, 2]}, index=["A", "A"]), pd.DataFrame({"y": [1, 2]})
        )


def test_phase_selection_uses_significant_mean_absolute_correlation():
    rho = pd.DataFrame(
        {"HU": [0.8, -0.7], "volume": [0.2, 0.9]}, index=["FxC50_Constant", "FxC50_Terminal"]
    )
    p = pd.DataFrame({"HU": [0.01, 0.01], "volume": [0.5, 0.01]}, index=rho.index)
    # Constant and Terminal both have a score of .8: original input order wins.
    assert select_phases(rho, p) == ["FxC50_Constant"]
    p.loc["FxC50_Terminal", "volume"] = 0.5
    assert select_phases(rho, p) == ["FxC50_Constant"]
    with pytest.raises(ValueError, match="identical"):
        select_phases(rho, p.iloc[::-1])
