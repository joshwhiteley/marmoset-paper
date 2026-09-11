# LIDs marmoset analysis

Code and data for the manuscript's lesion clustering, regimen-level correlations, sequential random-forest models, and SHAP analyses.

The main workflow starts from the deposited CSV files. It writes new results under `outputs/` and leaves the original data, figures, and modeling results alone. You do not need to run the notebooks or edit paths inside a script.

**A reproduction caveat:** The commands below refit the stated models with a deterministic split. See [reproduction checks and remaining differences](docs/reproducibility.md) before replacing manuscript panels.

## Run the analysis and figures

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the repository, and run these commands from the repository root:

```bash
uv sync --locked
uv run run-manuscript-analysis --check
uv run run-manuscript-analysis
uv run generate-manuscript-figures
```

`uv` manages Python 3.11 and installs the versions in `uv.lock`. Use the Git checkout for reproduction: Python distributions contain code only, not the study data, configuration, or historical artwork. The analysis command calculates the dosing-strategy correlations, lesion-outcome correlations, both model strategies, and SHAP values. The figure command then reads those results. It does not retrain models.

The output directories are:

```text
outputs/analysis/
  fxc/           Pairwise FxC50 comparisons between dosing strategies
  correlations/  Regimen-mean Spearman correlations, p-values, and pair counts
  models/        Fitted model bundles, metrics, predictions, and split assignments
  shap/          SHAP arrays, feature values, and lesion identifiers
outputs/figures/
  1/  2/  3/  4/  5/
```

Existing stage directories are never overwritten. For another run, choose a new pair of locations:

```bash
uv run run-manuscript-analysis --output-dir outputs/rerun/analysis
uv run generate-manuscript-figures \
  --analysis-dir outputs/rerun/analysis \
  --output-dir outputs/rerun/figures
```

## Which command produces each figure?

The numbering here follows the manuscript captions. Earlier scripts called the Figure 1 clustergram `1c`, the severity UMAP `1b`, and the combination heatmaps `2a`. Those names have been corrected.

| Panel | Implementation in `src/marmoset_paper/figures/` | Files under `outputs/figures/` |
| --- | --- | --- |
| 1A: baseline clusters | `figure1.py` | `1/figure_1a.svg` |
| 1B: baseline cluster medians and dendrogram | `figure1.py` | `1/figure_1b.svg`, `1/figure_1b_cluster_medians.csv` |
| 1C: baseline severity classes | `figure1.py` | `1/figure_1c.svg` |
| 1D: end-of-treatment lesion types | `figure1.py` | `1/figure_1d.svg`, counts, and unmapped-label table |
| 2A: single-drug susceptibility comparison | Not present in the deposited scripts | Requires the upstream source data and plotting workflow |
| 2B: combination-response heatmaps | `figure2.py` | `2/figure_2b_equipotent.svg`, `figure_2b_cas.svg`, `figure_2b_cell.svg` |
| 3A: correlation heatmaps | `figure3.py` | `3/figure_3a_severe.svg`, less-severe counterpart, displayed-value tables, and legend |
| 3B: selected correlations | `figure3.py` | `3/figure_3b_1.svg` through `figure_3b_3.svg`, with paired source tables |
| 4A/B: model scores and example lesions | `figure4.py` | `4/figure_4.svg`, `figure_4.png`, and selected lesion identifiers |
| 5A: week-8 SHAP summary | `figure5.py` | `5/figure_5a.svg` |
| 5B: in vitro predictors only | `figure5.py` | `5/figure_5b.svg` |
| 5C: selected features, stratified by regimen | `figure5.py` | `5/figure_5c_tp6_cellGRinf_dormancySimplePK_Termil.svg` and `figure_5c_tp4_NeutralNequip_FBC90.svg`, with source tables |

The Figure 5 selection uses **terminal-phase cellular-PK dormancy GRinf at TP6** and **equipotent neutral-normoxic FBC90 at TP4**. `Termil` is the spelling in the deposited feature names, not a different phase.

The original Illustrator files in `manuscript-figures/` contain the assembled layouts. The commands generate editable panel files, not pixel-identical copies of those manually assembled pages. Import the generated SVGs into the corresponding layout, retain the panel labels and legends, and check labels and source tables before export.

For only Figures 1 and 2B, no model fitting is needed:

```bash
uv run generate-manuscript-figures --figures 1 2 --output-dir outputs/data-only-figures
```

For individual analysis stages:

```bash
uv run run-manuscript-analysis --steps fxc correlations
uv run run-manuscript-analysis --steps models
uv run run-manuscript-analysis --steps shap
uv run generate-manuscript-figures --figures 3 4 5
```

These stage commands are an **alternative** to the full analysis command, not commands to run again over an existing completed run. `shap` reads the model bundles from the same `--output-dir`.

## What the models estimate

The analysis uses 261 severe lesions from 10 regimens. It selects test lesions within each regimen, giving **212 training lesions and 49 test lesions**. Both model strategies use the same split:

- `m`: baseline radiodensity and observed radiodensity at preceding timepoints.
- `b`: the same imaging measurements plus matched in vitro measurements.

There is one random forest for each of TP3–TP6. TP2 is treatment start; TP3, TP4, TP5, and TP6 correspond to weeks 2, 4, 6, and 8. Each forest uses 100 trees, maximum depth 10, and random state 42. These are explicit settings; a maximum depth of 10 is not a scikit-learn default.

Later models use **observed prior measurements**, not earlier model predictions. This is prediction conditional on imaging already collected, not an eight-week forecast from baseline alone. Evaluation concerns held-out lesions from represented regimens. It does not test unseen regimens, and lesions from the same marmoset can occur in both partitions.

The source code's compound-median imputation before splitting is retained. For the deposited modeling cohort it fills no values: baseline HU is complete, and missing in vitro values are missing for the entire regimen. Those values remain missing and are handled by the forest. The procedure is not train-fitted and should be reviewed before reuse on other data.

SHAP uses up to 1,000 stored training rows per combined model, sampled with seed 42. In this cohort there are only 212 training rows, so all are included. The exports retain `Compound`, `MarmID`, and `Lesion`. Feature values are never used as sample identifiers. Additivity is checked against the model's predictions.

`models/metrics.csv` contains MSE, MAE, and R² for every timepoint and strategy. The model bundles also store feature order and the actual training and test rows. Only load `.joblib` files you created or obtained from a trusted source: loading them can execute Python code.

## Rebuild intermediate data, if needed

This is **not required** to generate the figures from the deposited tables.

```bash
# Reproduce the wide lesion tables from the deposited workbook.
uv run prepare-manuscript-data --stage widen

# Rebuild severity labels and total volumes for the deposited cluster solution.
uv run prepare-manuscript-data --stage classify-deposited

# Re-run clustering as a separate analysis, with its original default seed made explicit.
uv run --extra clustering prepare-manuscript-data --stage cluster \
  --input-file outputs/prepared/widen/marm_data_wide.csv
```

Outputs go to separate subdirectories under `outputs/prepared/`. Widening preserves the original exclusions and first-record pivot rule. The 94 applied lesion exclusions are listed in `config/lesion_exclusions.csv`; the two excluded animals remain explicit in the preprocessing code.

A new clustering run does not automatically receive the old severity labels. Review its cluster identities before using it downstream. The deposited clustered and classified tables also contain different UMAP coordinates, so rebuilding classification does not reproduce the classified table's embedding. Details are in [the reproduction notes](docs/reproducibility.md).

## Repository layout

```text
src/marmoset_paper/
  analysis/   Data preparation, correlations, model fitting, and SHAP calculation
  figures/    One module per manuscript figure
  helpers/    DataFunctions, ModelingFunctions, ShapFunctions,
              FeatureAnalysisFunctions, PlottingFunctions, and plot constants
  cli.py      The three uv commands
  provenance.py
config/       Applied exclusions and input checksums
data/         Deposited inputs and historical intermediate tables
modeling/results/   Historical results; not overwritten by new runs
manuscript-figures/  Original manuscript layouts and exported figures
figures/      Earlier figure exports and drafts
notebooks/    Historical exploration; not the reproduction workflow
scripts/exploratory/lesion_changes.py
outputs/      New run products, ignored by Git
tests/        Numerical, identity, and workflow checks
```

The notebooks preserve earlier work, including outputs and old paths. They are not maintained as executable publication pipelines. Superseded scripts remain available in Git history; see [the cleanup record](docs/cleanup.md) for their replacements.

## Checks

A completed local run and its limitations are recorded in [docs/validation.md](docs/validation.md).

```bash
uv run pytest
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
shasum -a 256 -c config/input_checksums.sha256
```

The tests check the deposited cohort counts, raw-data widening, regimen aliases, the three Figure 3 correlations, observed-history model inputs, saved lesion identities, SHAP additivity, and failure/overwrite behavior. CI runs these checks and renders the data-only panels. A passing test suite does not establish biological validity or remove the limitations of lesion-level splitting.
