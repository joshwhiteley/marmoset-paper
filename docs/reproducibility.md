# Reproduction checks and remaining differences

The cleanup preserves the observed-history regression method. It does not change the split unit, tune models to recover a published score, or reinterpret ambiguous pathology labels. The OPC/QBS join correction and Figure 5 dormancy phase were confirmed by the author during cleanup.

## Inputs and cohort

The deposited workbook can be widened back to the tracked tables. Tests compare the reconstructed values by `Compound`, `MarmID`, and `Lesion`, rather than relying on CSV row order or floating-point text formatting.

| Check | Result |
| --- | --- |
| Included wide lesion table | 1,193 lesions; values agree with `data/marm_data_wide.csv` |
| Excluded/resistant wide table | 109 lesions; values agree with `data/marm_data_resistant.csv` |
| Current baseline severity classes | 566 severe, 627 less severe |
| Modeling cohort | 261 severe lesions from 10 regimens |
| Modeling split | 212 training, 49 test lesions, for both strategies |
| Correlation cohort after OPC/QBS normalization | 13 combination regimens per severity class; pair-specific counts vary |

`config/input_checksums.sha256` records the deposited files used by the workflow. Run its checksum command from the repository root. Run manifests also hash the actual inputs, so a later data replacement is visible even if the filename stays the same.

### Exclusions

The wide-table command preserves the original applied exclusions: two animals matched by `BK21` or `BN20`, 94 explicit animal/lesion pairs, and the `BDQ20` regimen filter. It preserves the original `first` aggregation for duplicate lesion/timepoint records and exports any such records for inspection.

The old script defined a `BQ47`, lesion 4 mask but did not include it in the final exclusion expression. The cleanup does not apply that unused mask. Changing it would change the cohort and requires a separate scientific decision.

## Clustering and Figure 1

- The deposited clustered and classified tables have the same lesion and cluster identities but **different UMAP coordinates**. Figure 1A uses the clustered table's coordinates; Figure 1C uses the classified table's coordinates, preserving the original panel inputs. A single common embedding should be selected before final publication if the panels are intended as directly comparable views.
- A local clustering rerun recovered all 16 deposited cluster labels, with adjusted Rand index 1.0. Its UMAP coordinates were not identical to either deposited embedding. The explicit-X graph reproduced the old implicit-X rerun exactly in the local validation. This is not evidence of pixel-level figure reproduction across environments. The local NumPy/Scanpy run emitted matrix-operation warnings and a Leiden future-default warning; these were not suppressed. Coordinates and graph weights are required to be finite before export.
- The historical Scanpy code computed PCA but did not explicitly select it for the neighbor graph. With five input variables, Scanpy 1.10 uses the standardized input matrix `X`. The cleaned command makes `use_rep="X"` explicit and omits the unused PCA calculation. Do not describe this implementation as a graph built from principal components without reviewing that methodological choice.
- The clustergram uses **average linkage and cosine distance**, as in the deposited script. The manuscript's complete-linkage wording should be reconciled with the code, not silently imposed on the analysis.
- The severity map is tied to the deposited cluster identities. `classify-deposited` checks those identities against the deposited classified table. It is not a general classifier for a new clustering solution.
- Figure 1D retains the original lesion-type mapping. It excludes three unmapped labels: two `Fibrotic scar` records and one `cas/fib cavity` record. Their identities are exported in `figure_1d_unmapped_labels.csv`. Percentages use mapped lesions as the denominator. No new pathology classification was inferred during cleanup.

The older `marm_data_wide_clustered_delta_scaled.csv` contains a different hot/cool assignment (599/594). It is not an input to the publication workflow.

## Correlations and Figure 3

The lesion table calls quabodepistat `OPC`; the in vitro table calls it `QBS`. The analysis now normalizes that alias by whole drug token before joining. Raw files are unchanged. A literal-name join would omit three regimens.

Pathology values are averaged within regimen and severity class. Total volume is checked against soft plus hard volume. Deltas are TP6 minus TP2. Spearman correlations use finite pairs and report the number of matched regimens for every test. P-values are two-sided SciPy values, with no multiple-testing adjustment.

The three regenerated scatterplots recover the displayed manuscript annotations to their printed precision:

| In vitro feature | Pathology outcome | ρ | p | Regimens |
| --- | --- | ---: | ---: | ---: |
| `FxC50_cholesterolCellPK_Constant` | `TP6_MeanHU` | 0.929 | 0.001 | 8 |
| `FxC50_butyrateCellPK_Terminal` | `delta_MeanHU` | −0.867 | 0.002 | 9 |
| `FxC50_5NcellPK` | `delta_MeanHU` | 0.715 | 0.009 | 12 |

Both simple-condition examples use **cellular PK**. Some dosing symbols and caseous/cellular wording in the existing artwork/caption disagree with those source features. The new plots show the full source feature names so this can be checked during layout.

The heatmaps retain two display choices from the original script:

1. For features with multiple assay phases, select the phase with the highest mean absolute correlation among nominally significant pathology pairs. If neither phase is significant, retain the first input phase. This selection is not a correction for multiple testing.
2. Reverse the sign of Einf correlations for display. Raw correlation tables retain the actual signs. The displayed-value tables and colorbar identify the reversal.

Grey cells represent non-significant or unavailable correlations. Displayed-value CSVs record exactly which phases and values were plotted. The bootstrap tail fraction in the exploratory notebook was not a valid null-distribution p-value and is not exported by the supported workflow; it was not used by the manuscript figure script.

## Models and Figure 4

The updated methods describe the implemented model accurately: four separate forests use observed prior radiodensity, not recursively propagated predictions. Each uses 100 trees, depth 10, and seed 42. The split is lesion-level within regimen, not a regimen holdout or an animal-grouped split.

There are two remaining reproducibility limits:

- **Historical split identities are missing.** The original splitter consumed a seeded random stream while iterating over unordered Polars treatment groups. The seed and 212/49 counts alone cannot recover the exact assignments. The cleaned splitter sorts treatments, uses a local random generator, and saves all assignments. It preserves the sampling rule but produces a new, reproducible split.
- **Historical forests are missing.** The original figure script expected dated models under `radiodensity-only/modeling/models/`, while another script searched `modeling/models/`. Neither model collection was deposited. No model is selected by taking the first glob match anymore.

The original score tables remain in `modeling/results/`. New scores must not be described as exact reproductions of those tables. The cleaned Figure 4 reads metrics and predictions from one new analysis run, rather than combining old metrics with a differently dated forest.

Permutation importance is calculated on all 49 held-out lesions using the forest's native missing-value support. The old script dropped rows missing any predictor before this diagnostic and could silently omit the calculation after an error. This correction affects the diagnostic, not model fitting or the prediction metrics.

Compound-median imputation before the split is preserved from the original code. For this deposited cohort, a direct comparison confirms that it changes no modeling measurements: baseline HU has no nulls, and the combined feature matrix retains the same 3,178 null entries before and after imputation. Entirely unmeasured regimen/feature groups remain missing and are handled natively by the forest. On other data, pre-split imputation could allow held-out measurements to influence preprocessing. A train-fitted imputer or a marmoset-grouped split should be treated as a separate methodological change, not cosmetic cleanup.

Example trajectories are identified by **regimen, marmoset, and lesion**. The old plotting code selected by lesion number within regimen and then took the first matching row; lesion numbers can recur across animals. The new plots retain full identity and export the selected examples. Their values are predictions conditional on observed history, even when connected by a line.

## SHAP and Figure 5

SHAP calculations use the actual stored model inputs. They do not reconstruct a split from a seed or join treatment labels by coincidentally equal feature values. Both TreeExplainer's additivity check and a direct reconstruction of model predictions are tested.

The figure selections are:

- TP6 combined model, top 30 features for the full plot.
- The same TP6 SHAP values restricted to non-imaging features, top 25 for the zoomed plot. This does not refit a reduced model.
- TP6 `cellGRinf_dormancySimplePK_Termil`, terminal phase, for the dormancy highlight.
- TP4 `NeutralNequip_FBC90` for the neutral-normoxic highlight.

The highlight panels use original lesion identities to color regimens. Missing feature values remain grey in the global view. Summed in vitro contributions are also exported per lesion and summarized per regimen. Contributions describe the fitted forest, not a causal treatment effect.

Because the forests are newly fitted, SHAP values and their rankings need not match the historical manuscript image exactly. Check the new figure and source tables together before updating the manuscript.

## Figure 2 and assembled artwork

Figure 2B can be regenerated from `in_vitro_modeling.csv`. Values outside the display range are marked with an asterisk; missing cells are grey. The source values are exported without clipping. Drug abbreviations use Q for quabodepistat.

Figure 2A's single-drug susceptibility workflow was not present in the deposited scripts. It remains a publication dependency, not a panel filled in with substitute data. The original Illustrator layouts are retained for all figures. Exact layout, panel lettering, and final typographic adjustments remain manual steps.

## Validation scope

The checks cover the supported Python package, its command-line entry points, the consolidated exploratory script, and the tests. The historical notebooks were inventoried and inspected but were not executed or rewritten. Raw data, historical metrics, Illustrator files, and existing figure exports were not modified.

Local validation includes the full analysis, all supported figure panels, workbook widening, deposited classification, a clustering rerun, and the exploratory lesion-change heatmaps. Tests exercise numerical values and sample identities, not just output existence. Ruff checks imports, undefined names, basic errors, and formatting. No type checker or dependency-security audit is configured; passing Ruff is not evidence that either of those checks was performed.
