# Publication cleanup record

The starting point was commit `bc4cd45`. The cleanup was done on `publication-cleanup`. Existing local manuscript, revision, and other untracked files were not included in the cleanup commits. The pre-existing `.gitignore` edit was left unstaged.

## What changed

- Packaged the supported code under `src/marmoset_paper/` and added three `uv run` commands.
- Separated data preparation, statistical analysis, model fitting, SHAP calculation, and figure rendering.
- Replaced duplicated plotting implementations with one figure module per manuscript figure and a small set of shared functions.
- Removed import-time analyses, first-match model discovery, broad catch-and-continue behavior, unused data wrapper classes, warning suppression, and obsolete debug scaffolding from the executable workflow.
- Replaced the long exclusion-mask expression with the same applied exclusions in a reviewable CSV.
- Added a deterministic treatment traversal, persisted split identities, model configuration checks, and explicit feature order.
- Retained marmoset identity when selecting trajectories or exporting SHAP values. Equal measurement values are not join keys.
- Added input/output hashes, source hashes, completed-stage checks, and upstream manifests to the provenance chain.
- Added tests and CI. No raw data or existing scientific output was deleted or rewritten.
- Restricted Python distribution contents to code and setup files. Local cached environments, manuscript drafts, and revision work are not packaged.

The scientific changes were deliberately limited. The observed-history forests, within-regimen lesion split, and pre-split median imputation were retained. OPC/QBS normalization for correlations and the terminal dormancy phase for Figure 5 were confirmed with the author. Other manuscript/code differences are listed in [reproducibility.md](reproducibility.md).

## Where the old entry points went

| Earlier code | Supported replacement |
| --- | --- |
| `data/widen_data.py` | `prepare-manuscript-data --stage widen`; `analysis/preprocessing.py` |
| `data/cluster_lesions.py` | `prepare-manuscript-data --stage cluster`; `analysis/preprocessing.py` |
| `modeling/hot_cool_classification.py` | `prepare-manuscript-data --stage classify-deposited`; `figures/figure1.py` |
| `calculate_fxc_correlations.py` | `run-manuscript-analysis --steps fxc`; `analysis/fxc.py` |
| `correlations_agg.ipynb` calculations | `run-manuscript-analysis --steps correlations`; `analysis/correlations.py` |
| `modeling/sequential_regression.py` | `run-manuscript-analysis --steps models`; `analysis/modeling.py` |
| `radiodensity_fig3.py` and its manuscript-folder copy | `figures/figure3.py` |
| `create_final_radiodensity_figure.py` and its manuscript-folder copy | `figures/figure4.py` |
| `create_manuscript_shap_plots.py`, `create_shap_highlights.py` | `analysis/shap.py`, `figures/figure5.py` |
| `SHAP_contribution_plots.py`, `shap_feature_contribution.py`, and the old Figure 5 copy | Identified SHAP exports and per-regimen contribution summaries from Figure 5 |
| Three exploratory lesion-change scripts | `scripts/exploratory/lesion_changes.py` |
| Old `ModelingFunctions`/`PlottingFunctions` SHAP helpers | `helpers/ShapFunctions.py` and `helpers/PlottingFunctions.py` |

The earlier standalone ground-truth trajectory script, incomplete model-analysis driver, duplicate SHAP styling variants, and unused data wrappers were retired. Their individual exploratory plot variants are not all maintained as new CLI options. The original exports remain available, and the code is recoverable from Git history:

```bash
git show bc4cd45:modeling/sequential_regression.py
git show bc4cd45:helpers/PlottingFunctions.py
```

The notebooks were moved to `notebooks/` without changing their source cells or saved outputs. They are historical records, not notebooks that are expected to run against the new package API.

## Quality review

The initial scan covered all 33 tracked Python files, the eight notebooks, figure inputs/outputs, dependency declarations, and existing data/results. It included an AST-based scientific-Python audit and manual tracing of the manuscript panels. The main problems were duplicated workflows, missing dependencies and artifacts, hidden failures, ambiguous row identity, and unsupported reproduction claims—not merely comment style.

The supported code now passes Ruff and the test suite. An independent read-only review checked the modeling method, approved feature selections, identity handling, and documentation. Its recommendations on upstream manifest validation, SHAP bundle consistency, and model-invariant tests were implemented.

The historical notebooks, Illustrator compositions, and all external or untracked revision analyses remain outside the automated test surface. No type-checker, dependency-vulnerability, or full notebook execution result is claimed.

For the consolidated exploratory heatmaps:

```bash
uv run python scripts/exploratory/lesion_changes.py \
  --output-dir outputs/lesion-changes
```

This computes TP6−TP2 changes, standardizes them across the input cohort, and exports cluster/regimen medians and severity-stratified heatmaps. It is not part of the numbered manuscript workflow.
