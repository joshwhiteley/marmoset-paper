"""Optional reconstruction of intermediate lesion tables.

Publication plots use the deposited clustered tables. Re-running clustering creates
new cluster identities and requires scientific review before assigning severity.
"""

from pathlib import Path

import numpy as np
import polars as pl

from marmoset_paper.helpers.constants import LESS_SEVERE_CLUSTERS, SEVERE_CLUSTERS

MEASUREMENTS = [
    "MeanHU",
    "MedianHU",
    "StandDevHU",
    "MeanSUV",
    "MedianSUV",
    "StandDevSUV",
    "MaxSUV",
    "HighSUV",
    "VOLMORE1p5_SUV",
    "VOLMORE2_SUV",
    "TLG_SUV",
    "TLG_HardPlusSoft_SUV",
    "TLG_HardPlusSoft_SublesionDivMuscleTLG",
    "SoftVol",
    "HardVol",
    "log10CFU",
    "LesionType",
    "FinalLesionType",
    "Consolidate",
    "Contour",
]
CLUSTERING_FEATURES = ["TP2_MeanHU", "TP2_StandDevHU", "TP2_SoftVol", "TP2_HardVol", "TP2_MeanSUV"]
KEYS = ["Compound", "MarmID", "Lesion"]


def widen_table(data: pl.DataFrame) -> pl.DataFrame:
    """Pivot timepoints, retaining the original first-record aggregation rule."""
    wide = data.pivot("Timepoint", index=KEYS, values=MEASUREMENTS, aggregate_function="first")
    return wide.rename(
        {
            column: f"TP{column.rsplit('_', 1)[1]}_{column.rsplit('_', 1)[0]}"
            for column in wide.columns
            if column not in KEYS
        }
    ).sort("MarmID", maintain_order=True)


def prepare_wide(workbook: Path, exclusions_file: Path, output: Path) -> dict:
    raw = pl.read_excel(workbook, infer_schema_length=None)
    exclusions = pl.read_csv(exclusions_file).with_columns(pl.lit(True).alias("excluded_lesion"))
    marked = raw.join(
        exclusions, on=["MarmID", "Lesion"], how="left", validate="m:1", maintain_order="left"
    )
    resistant = pl.col("MarmID").str.contains("BK21|BN20") | pl.col("excluded_lesion").fill_null(
        False
    )
    removed = marked.filter(resistant).select(raw.columns)
    included = marked.filter(~resistant & ~pl.col("Compound").str.contains("BDQ20")).select(
        raw.columns
    )
    duplicates = raw.filter(raw.select(KEYS + ["Timepoint"]).is_duplicated())
    duplicates.write_csv(output / "duplicate_timepoint_rows.csv")
    wide = widen_table(included)
    wide.write_csv(output / "marm_data_wide.csv")
    widen_table(removed).write_csv(output / "marm_data_resistant.csv")
    return {
        "raw_rows": len(raw),
        "included_rows": len(included),
        "wide_lesions": len(wide),
        "resistance_excluded_rows": len(removed),
        "duplicate_timepoint_rows": len(duplicates),
    }


def cluster_table(input_file: Path, output: Path, seed: int = 0) -> dict:
    """Re-run the historical Scanpy calls with their original default seed made explicit."""
    import scanpy as sc
    from anndata import AnnData
    from sklearn.preprocessing import StandardScaler

    data = pl.read_csv(input_file, infer_schema_length=None)
    X = data.select(CLUSTERING_FEATURES).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Clustering features must be finite")
    scaled = StandardScaler().fit_transform(X)
    adata = AnnData(scaled)
    # With five variables, the original Scanpy call selected X, not X_pca.
    # Make that representation explicit and omit the unused PCA calculation.
    sc.pp.neighbors(adata, metric="cosine", use_rep="X", random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    sc.tl.leiden(adata, resolution=1, random_state=seed, flavor="leidenalg")
    result = data.with_columns(
        pl.Series("UMAP1", adata.obsm["X_umap"][:, 0]),
        pl.Series("UMAP2", adata.obsm["X_umap"][:, 1]),
        pl.Series("cluster", adata.obs.leiden.astype(int).to_numpy()),
    )
    result.write_csv(output / "marm_data_wide_clustered.csv")
    return {
        "lesions": len(result),
        "clusters": result["cluster"].n_unique(),
        "severity_assigned": False,
    }


def classify_deposited(data_dir: Path, output: Path) -> dict:
    """Rebuild severity and total volume only for the deposited cluster solution."""
    data = pl.read_csv(data_dir / "marm_data_wide_clustered.csv", infer_schema_length=None)
    if set(data["cluster"]) != set(LESS_SEVERE_CLUSTERS + SEVERE_CLUSTERS):
        raise ValueError("Classification requires the deposited 16-cluster labels")
    classified = data.with_columns(
        pl.when(pl.col("cluster").is_in(LESS_SEVERE_CLUSTERS))
        .then(pl.lit("cool"))
        .otherwise(pl.lit("hot"))
        .alias("classif"),
        *[
            (pl.col(f"TP{tp}_SoftVol") + pl.col(f"TP{tp}_HardVol")).alias(f"TP{tp}_TotalVol")
            for tp in [2, 6]
        ],
    )
    reference = pl.read_csv(
        data_dir / "marm_data_wide_clustered_classif.csv", infer_schema_length=None
    )
    identity_columns = KEYS + ["cluster", "classif"]
    if (
        not classified.select(identity_columns)
        .sort(KEYS)
        .equals(reference.select(identity_columns).sort(KEYS))
    ):
        raise ValueError(
            "Cluster labels do not match the deposited severity assignments; review the new solution"
        )
    classified.write_csv(output / "marm_data_wide_clustered_classif.csv")
    return {
        "lesions": len(classified),
        "severe": classified.filter(pl.col("classif") == "hot").height,
    }
