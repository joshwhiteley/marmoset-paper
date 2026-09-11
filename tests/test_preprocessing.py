from pathlib import Path

import numpy as np
import pandas as pd

from marmoset_paper.analysis.preprocessing import classify_deposited, prepare_wide

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["Compound", "MarmID", "Lesion"]


def test_widening_preserves_deposited_data_and_exclusions(tmp_path):
    counts = prepare_wide(
        ROOT / "data/FINALCORRECTED_Merged_CFUandPETCT_20231005.xlsx",
        ROOT / "config/lesion_exclusions.csv",
        tmp_path,
    )
    assert counts["wide_lesions"] == 1193
    for name in ["marm_data_wide.csv", "marm_data_resistant.csv"]:
        expected = pd.read_csv(ROOT / "data" / name).sort_values(KEYS).reset_index(drop=True)
        actual = pd.read_csv(tmp_path / name).sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False)


def test_classification_and_total_volume_match_deposited_values(tmp_path):
    counts = classify_deposited(ROOT / "data", tmp_path)
    assert counts == {"lesions": 1193, "severe": 566}
    reference = pd.read_csv(ROOT / "data/marm_data_wide_clustered_classif.csv").sort_values(KEYS)
    result = pd.read_csv(tmp_path / "marm_data_wide_clustered_classif.csv").sort_values(KEYS)
    assert result.classif.tolist() == reference.classif.tolist()
    for tp in [2, 6]:
        np.testing.assert_allclose(result[f"TP{tp}_TotalVol"], reference[f"TP{tp}_TotalVol"])
