# Historical notebooks

These notebooks record earlier analyses and plotting experiments. They were moved here without changing their source cells or saved outputs.

They are not part of the publication reproduction workflow. Some contain personal absolute paths, missing intermediate files, or imports from the retired helper API. Installing a notebook kernel does not make them compatible with the cleaned package.

| Notebook | Earlier work |
| --- | --- |
| `correlations_agg.ipynb` | Regimen-aggregated correlations and exploratory scatterplots; the supported calculations are now in `analysis/correlations.py` |
| `correlations.ipynb` | Earlier lesion-level correlation exploration |
| `correlations_plotting.ipynb` | Correlation figure styling |
| `cluster_specific_sequential_regression.ipynb` | Cluster-specific model comparisons |
| `minimal_model.ipynb` | Earlier modeling prototype |
| `SHAP_feature_analysis.ipynb` | Feature-importance exploration |
| `shap_highlighting.ipynb`, `shap_highlight_v2.ipynb` | Earlier SHAP highlight plots |

Use `uv run run-manuscript-analysis` and `uv run generate-manuscript-figures` for the maintained workflow. If revisiting a notebook, work on a copy, inspect every input and output path, and treat the result as a new analysis. The original repository layout is available at commit `bc4cd45`.
