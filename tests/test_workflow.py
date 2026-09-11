import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from marmoset_paper.cli import analysis_main, figures_main
from marmoset_paper.figures.figure1 import cluster_medians
from marmoset_paper.helpers.PlottingFunctions import plot_performance_metrics
from marmoset_paper.provenance import recorded_run

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_records_failures_and_refuses_overwrite(tmp_path):
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="expected"):
        with recorded_run(output, [ROOT / "pyproject.toml"], {"seed": 42}, ROOT):
            raise RuntimeError("expected failure")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["config"] == {"seed": 42}
    assert len(manifest["inputs"][0]["sha256"]) == 64
    with pytest.raises(FileExistsError):
        with recorded_run(output, [], {}, ROOT):
            pass


def test_missing_input_does_not_create_output(tmp_path):
    with pytest.raises(FileNotFoundError):
        with recorded_run(tmp_path / "run", [tmp_path / "missing"], {}, ROOT):
            pass
    assert not (tmp_path / "run").exists()


def test_analysis_preflight_has_no_writes(tmp_path):
    analysis_main(["--check", "--output-dir", str(tmp_path / "analysis")])
    assert not (tmp_path / "analysis").exists()


def test_figure_preflight_fails_for_missing_artifacts(tmp_path):
    with pytest.raises(SystemExit) as error:
        figures_main(["--check", "--figures", "4", "--analysis-dir", str(tmp_path)])
    assert error.value.code == 2


def test_cluster_medians_match_historical_polars_calculation():
    import numpy as np
    import polars as pl

    from marmoset_paper.figures.figure1 import FEATURES

    data = pd.read_csv(ROOT / "data/marm_data_wide_clustered.csv")
    frame = pl.from_pandas(data)
    expected = (
        frame.select(
            pl.col("cluster"),
            *[((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in FEATURES],
        )
        .group_by("cluster")
        .agg(pl.col(FEATURES).median())
        .sort("cluster")
        .select(FEATURES)
        .to_numpy()
        .T
    )
    np.testing.assert_allclose(cluster_medians(data).to_numpy(), expected)


def test_performance_plot_rejects_missing_scores():
    frame = pd.DataFrame(
        {"strategy": ["m"], "timepoint": [3], "mse": [0.0], "mae": [0.0], "r2": [1.0]}
    )
    with pytest.raises(ValueError, match="exactly one"):
        plot_performance_metrics(frame, [])
