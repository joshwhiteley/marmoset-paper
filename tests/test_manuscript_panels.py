from pathlib import Path

import pandas as pd
import pytest

from marmoset_paper.analysis.correlations import run_correlations
from marmoset_paper.figures.figure3 import SCATTER_PAIRS
from marmoset_paper.figures.figure5 import HIGHLIGHTS
from marmoset_paper.helpers.DataFunctions import normalize_regimen

ROOT = Path(__file__).resolve().parents[1]


def test_opc_qbs_alias_applies_to_whole_drug_tokens():
    assert normalize_regimen("BDQ+OPC") == "BDQ+QBS"
    assert normalize_regimen("del + opc") == "DEL+QBS"
    assert normalize_regimen("OPC_OTHER") == "OPC_OTHER"


def test_figure3_pairs_recover_manuscript_annotations(tmp_path):
    counts = run_correlations(ROOT / "data", tmp_path)
    assert len(counts["severe"]["matched_regimens"]) == 13
    rho = pd.read_csv(tmp_path / "severe_rho_values_mean.csv", index_col="feature")
    p = pd.read_csv(tmp_path / "severe_p_values_mean.csv", index_col="feature")
    n = pd.read_csv(tmp_path / "severe_n_pairs_mean.csv", index_col="feature")
    for (feature, target), r, p_value, count in zip(
        SCATTER_PAIRS, [0.929, -0.867, 0.715], [0.001, 0.002, 0.009], [8, 9, 12]
    ):
        assert rho.loc[feature, target] == pytest.approx(r, abs=0.0005)
        assert p.loc[feature, target] == pytest.approx(p_value, abs=0.0005)
        assert n.loc[feature, target] == count


def test_figure5_highlights_use_author_confirmed_phase_and_horizons():
    assert HIGHLIGHTS == [
        ("tp6", "cellGRinf_dormancySimplePK_Termil"),
        ("tp4", "NeutralNequip_FBC90"),
    ]
    columns = pd.read_csv(ROOT / "data/in_vitro_diamond_data.csv", nrows=0).columns
    assert all(feature in columns for _, feature in HIGHLIGHTS)
