# Deposited data

These files are inputs and historical intermediate tables. The publication commands read them but do not overwrite them. New tables go under `outputs/`.

## Inputs used by the supported workflow

| File | Contents and use |
| --- | --- |
| `FINALCORRECTED_Merged_CFUandPETCT_20231005.xlsx` | Long-format lesion measurements; used by the optional widening command |
| `marm_data_wide.csv` | 1,193 included lesions; input for a clustering rerun |
| `marm_data_resistant.csv` | 109 excluded/resistant lesions; checked against the widening output |
| `marm_data_wide_clustered.csv` | 1,193 lesions with 16 cluster labels and UMAP coordinates; Figure 1A/B |
| `marm_data_wide_clustered_classif.csv` | 1,193 lesions, severity labels, total volumes, and a second saved embedding; Figure 1C/D, correlations, and modeling |
| `in_vitro_modeling.csv` | 22 drug/regimen rows and 126 measurement columns; Figure 2B, dosing comparisons, and lesion-outcome correlations |
| `in_vitro_diamond_data.csv` | 12 regimen rows, 104 predictor columns, and metadata; the modeling filter retains 10 matched regimens |

Lesion identity is the combination of `Compound`, `MarmID`, and `Lesion`. Lesion number alone is not unique across animals. TP2 is treatment start, and TP3–TP6 are weeks 2, 4, 6, and 8.

The current severity table contains 566 `hot` (severe) and 627 `cool` (less severe) lesions. Total volume is `SoftVol + HardVol`. Correlation deltas are TP6 minus TP2.

Drug names differ between sources. In particular, the lesion table's `OPC` corresponds to `QBS` in the in vitro table. The correlation pipeline applies this confirmed alias during the join. Modeling retains the historical exclusion of QBS-containing regimens. Source labels are not edited in place.

The two in vitro tables also use different feature naming conventions. Do not substitute one for the other when loading a saved model. Feature order and exact names are stored in each new model bundle.

## Historical tables

The `*_delta.csv`, `*_delta_scaled.csv`, and existing correlation CSVs are retained for reference. They are not substituted for freshly generated analysis outputs. The older scaled-delta table has a different hot/cool assignment from the current classified table.

The existing correlation files without a `_mean` suffix are not the missing regimen-mean inputs expected by the old Figure 3 script. Generate the supported correlation outputs instead of renaming those files.

Input checksums are in `config/input_checksums.sha256`. See the main README for commands and `docs/reproducibility.md` for the remaining differences between intermediate tables.
