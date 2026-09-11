"""Command-line entry points for analysis and manuscript panel generation."""

import argparse
import importlib
from dataclasses import asdict
from pathlib import Path

from marmoset_paper.provenance import recorded_run

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "pyproject.toml").is_file():
    ROOT = Path.cwd()
LESIONS = "marm_data_wide_clustered_classif.csv"


def common_parser(description: str, output: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / output,
        help="New output location; existing stage/figure directories are never overwritten",
    )
    parser.add_argument(
        "--check", action="store_true", help="Check required inputs without running"
    )
    return parser


def correlation_inputs(directory: Path):
    return [
        directory / f"{severity}_{metric}_values_mean.csv"
        for severity in ["severe", "less_severe"]
        for metric in ["rho", "p"]
    ]


def check_inputs(parser, inputs):
    missing = sorted({str(path) for path in inputs if not path.is_file()})
    if missing:
        parser.error("Missing inputs:\n  " + "\n  ".join(missing))


def check_outputs(parser, outputs):
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        parser.error(
            "Output directories already exist; use a new --output-dir:\n  " + "\n  ".join(existing)
        )


def analysis_main(argv=None):
    from marmoset_paper.analysis.correlations import run_correlations
    from marmoset_paper.analysis.fxc import run_fxc
    from marmoset_paper.analysis.modeling import ModelConfig, train_models
    from marmoset_paper.analysis.shap import run_shap

    parser = common_parser("Run the supported LIDs manuscript analyses.", "analysis")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["fxc", "correlations", "models", "shap"],
        default=["fxc", "correlations", "models", "shap"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shap-sample-size", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.shap_sample_size < 1:
        parser.error("--shap-sample-size must be positive")
    sources = {
        "fxc": [args.data_dir / "in_vitro_modeling.csv"],
        "correlations": [args.data_dir / LESIONS, args.data_dir / "in_vitro_modeling.csv"],
        "models": [args.data_dir / LESIONS, args.data_dir / "in_vitro_diamond_data.csv"],
        "shap": [args.output_dir / "models" / f"b_tp{tp}.joblib" for tp in range(3, 7)],
    }
    external = [
        path
        for step in args.steps
        for path in sources[step]
        if not (step == "shap" and "models" in args.steps)
    ]
    check_inputs(parser, external)
    if args.check:
        print(
            "Inputs available. Model bundles for SHAP will be generated first."
            if "models" in args.steps
            else "All required inputs are available."
        )
        return
    check_outputs(parser, [args.output_dir / step for step in args.steps])
    config = ModelConfig(seed=args.seed)
    for step in ["fxc", "correlations", "models", "shap"]:
        if step not in args.steps:
            continue
        destination = args.output_dir / step
        settings = (
            asdict(config)
            if step == "models"
            else {
                "seed": args.seed,
                "sample_size": args.shap_sample_size,
            }
            if step == "shap"
            else {
                "aggregation": "regimen mean" if step == "correlations" else None,
                "method": "spearman",
                "p_adjustment": None,
            }
        )
        print(f"Running {step}: {destination}")
        with recorded_run(destination, sources[step], settings, ROOT) as record:
            if step == "fxc":
                record["counts"] = run_fxc(args.data_dir, destination)
            elif step == "correlations":
                record["counts"] = run_correlations(args.data_dir, destination)
            elif step == "models":
                record["counts"] = train_models(args.data_dir, destination, config)
            else:
                record["counts"] = run_shap(
                    args.output_dir / "models", destination, args.shap_sample_size, args.seed
                )


def prepare_main(argv=None):
    from marmoset_paper.analysis.preprocessing import (
        classify_deposited,
        cluster_table,
        prepare_wide,
    )

    parser = common_parser(
        "Rebuild intermediate lesion tables without replacing deposited data.", "prepared"
    )
    parser.add_argument(
        "--stage", choices=["widen", "cluster", "classify-deposited"], required=True
    )
    parser.add_argument("--input-file", type=Path, help="Wide CSV for --stage cluster")
    parser.add_argument(
        "--seed", type=int, default=0, help="Original Scanpy default seed (clustering only)"
    )
    args = parser.parse_args(argv)
    if args.input_file and args.stage != "cluster":
        parser.error("--input-file is only used by --stage cluster")
    sources = {
        "widen": [
            args.data_dir / "FINALCORRECTED_Merged_CFUandPETCT_20231005.xlsx",
            ROOT / "config/lesion_exclusions.csv",
        ],
        "cluster": [args.input_file or args.data_dir / "marm_data_wide.csv"],
        "classify-deposited": [
            args.data_dir / "marm_data_wide_clustered.csv",
            args.data_dir / LESIONS,
        ],
    }
    check_inputs(parser, sources[args.stage])
    if args.check:
        print("All required preparation inputs are available.")
        return
    destination = args.output_dir / args.stage
    check_outputs(parser, [destination])
    with recorded_run(
        destination, sources[args.stage], {"stage": args.stage, "seed": args.seed}, ROOT
    ) as record:
        if args.stage == "widen":
            record["counts"] = prepare_wide(*sources["widen"], destination)
        elif args.stage == "cluster":
            record["counts"] = cluster_table(sources["cluster"][0], destination, args.seed)
        else:
            record["counts"] = classify_deposited(args.data_dir, destination)


def figures_main(argv=None):
    parser = common_parser(
        "Generate manuscript panels from deposited data and a recorded analysis run.", "figures"
    )
    parser.add_argument(
        "--figures", nargs="+", choices=["1", "2", "3", "4", "5"], default=["1", "2", "3", "4", "5"]
    )
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/analysis")
    args = parser.parse_args(argv)
    sources = {
        "1": [args.data_dir / LESIONS, args.data_dir / "marm_data_wide_clustered.csv"],
        "2": [args.data_dir / "in_vitro_modeling.csv"],
        "3": correlation_inputs(args.analysis_dir / "correlations")
        + [
            args.analysis_dir / "correlations/in_vitro_combinations.csv",
            args.analysis_dir / "correlations/severe_compound_means.csv",
        ],
        "4": [args.analysis_dir / "models" / name for name in ["metrics.csv", "predictions.csv"]],
        "5": [
            args.analysis_dir / "shap" / f"b_{timepoint}_{name}"
            for timepoint in ["tp4", "tp6"]
            for name in ["values.npy", "features.csv", "samples.csv"]
        ],
    }
    check_inputs(parser, [path for number in args.figures for path in sources[number]])
    if args.check:
        print(
            "All required figure inputs are available. This does not validate scientific equivalence."
        )
        return
    check_outputs(parser, [args.output_dir / number for number in args.figures])
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "marmoset-paper"
    for number in dict.fromkeys(args.figures):
        destination = args.output_dir / number
        print(f"Generating figure {number}: {destination}")
        module = importlib.import_module(f"marmoset_paper.figures.figure{number}")
        settings = {"figure": number}
        if number == "3":
            settings["scatter_pairs"] = module.SCATTER_PAIRS
        elif number == "5":
            settings["highlights"] = module.HIGHLIGHTS
        with recorded_run(destination, sources[number], settings, ROOT):
            module.generate(
                args.data_dir if number in {"1", "2"} else args.analysis_dir, destination
            )
