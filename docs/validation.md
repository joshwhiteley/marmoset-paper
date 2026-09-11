# Local validation record

- Date: 2026-09-11
- Branch: `publication-cleanup`
- Analysis/figure source commit: `9c77301b9c4ec6fea36792d138cf47314fc6b481`
- Run directory: `outputs/publication-review/`
- Status: computational workflow validated; manuscript differences still require author review.

## Objective

Check that the documented commands run from the deposited inputs, retain the stated model method, and produce traceable figures without overwriting historical results.

## Commands

From the repository root:

```bash
uv sync --locked
uv run run-manuscript-analysis --output-dir outputs/publication-review/analysis
uv run generate-manuscript-figures \
  --analysis-dir outputs/publication-review/analysis \
  --output-dir outputs/publication-review/figures
uv run --extra clustering prepare-manuscript-data --stage cluster \
  --input-file outputs/prepared/widen/marm_data_wide.csv \
  --output-dir outputs/publication-review/prepared
```

The widening and deposited-classification commands were also run at their documented default locations under `outputs/prepared/`. The consolidated exploratory lesion-change script was run separately. `--check` was tested from outside the repository using `uv run --project`.

## Environment and provenance

The analysis ran with Python 3.11.15 on macOS 26.5, arm64, using `uv.lock`. Main analysis versions include NumPy 2.0.2, pandas 2.2.3, Polars 1.26.0, scikit-learn 1.6.1, SciPy 1.13.1, and SHAP 0.48.0. Clustering used Scanpy 1.10.4. Computation was CPU-only.

The working tree contained documentation work and pre-existing local manuscript/revision files. Source code was committed at the revision above. Each run manifest records the complete Git status and hashes the source files actually present. The local files were not staged into the cleanup commits.

Input checksums are listed in `config/input_checksums.sha256`. The generated manifests include exact package inventories, settings, input/output hashes, and links to upstream manifests. All consumed artifact hashes were rechecked after generation.

## Results

| Check | Outcome |
| --- | --- |
| Unit/integration suite | 41 tests passed |
| Ruff lint and format | Passed for `src`, `tests`, and `scripts` |
| Scientific-Python AST scanner | No remaining heuristic findings in the 32 tracked Python files |
| Input checksum verification | All nine deposited input/reference files matched |
| Wide table reconstruction | 1,193 included and 109 excluded lesions; deposited values matched |
| Current severity classes | 566 severe and 627 less severe |
| Model split | 212 training and 49 test lesions; identical across strategies |
| Pre-split imputation on deposited model inputs | No measurements changed; combined input null count remained 3,178 |
| Regimen-mean correlations | 13 matched combination regimens per severity class |
| Three Figure 3 annotations | Recovered ρ = 0.929, −0.867, 0.715 at the printed precision |
| SHAP | All four combined models passed additivity and identity checks |
| Figure generation | 18 SVGs across the supported panels, plus Figure 4 PNG and source tables |
| Clustering rerun | All 1,193 deposited cluster labels recovered; adjusted Rand index 1.0 |
| Upstream integrity | All outputs of the ten final analysis/figure/clustering stages matched their manifests |
| Package build | Wheel and source distribution built; archive members checked for local caches, drafts, and revision files |

Two independent full analysis processes produced identical metrics, prediction tables, split assignments, and SHAP arrays at all four timepoints. This is a local repeatability check, not a claim of exact agreement with the historical manuscript models or of bitwise equality on every platform.

Selected Figure 3, 4, and 5 exports were rendered for visual inspection. The review caught and corrected overlapping category/colorbar labels and a missing model-strategy legend. Final Illustrator composition remains a manual publication step.

## Limitations

- Figure 2A has no deposited source workflow.
- Original fitted forests and split identities are absent. Historical scores and SHAP rankings are not recovered exactly by the new split.
- The deposited UMAP embeddings differ. Clustering coordinates are not promised to be identical across runs or environments.
- The local clustering run emitted matrix-operation warnings and a Leiden future-default warning. Warnings were not suppressed; the exported coordinates and graph weights passed finite-value checks.
- Three unmapped Figure 1D lesion labels remain excluded and are exported for review.
- The historical notebooks and untracked revision analyses were not executed.
- The GitHub Actions workflow was added, but no remote CI execution or push was performed.

See [reproducibility.md](reproducibility.md) for the scientific interpretation limits and remaining manuscript/code discrepancies. Retain the complete run directories when archiving results; the package distributions alone are not a study-data release.
