from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import copy_raw_dataset, load_raw_data
from src.export_outputs import export_susceptibility_with_coordinates
from src.preprocessing import prepare_dataset
from src.train_models import get_model_spec, train_single_model
from src.utils import ensure_directories, slugify


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an ArcGIS-friendly flood susceptibility export for the selected best model."
    )
    parser.add_argument("--data-path", type=Path, default=config.DEFAULT_RAW_DATA_INPUT_PATH)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional explicit model name. If omitted, the best test ROC-AUC model is used.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "average_precision", "f1", "mcc", "balanced_accuracy", "accuracy"],
        help="Metric used when automatically selecting the export model from test metrics.",
    )
    return parser


def resolve_model_choice(model_name: str | None, selection_metric: str) -> tuple[str, pd.Series]:
    metrics_path = config.METRICS_DIR / "test_model_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics table at {metrics_path}. The repo needs outputs/metrics/test_model_metrics.csv."
        )

    metrics_frame = pd.read_csv(metrics_path)
    if model_name:
        match = metrics_frame.loc[
            metrics_frame["model"].str.casefold() == model_name.casefold()
        ].reset_index(drop=True)
        if match.empty:
            available = ", ".join(metrics_frame["model"].tolist())
            raise KeyError(f"Model {model_name} not found in metrics table. Available models: {available}")
        return str(match.iloc[0]["model"]), match.iloc[0]

    ranked = metrics_frame.sort_values(
        [selection_metric, "average_precision", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    return str(ranked.iloc[0]["model"]), ranked.iloc[0]


def build_arcgis_export(export_frame: pd.DataFrame, threshold: float, file_path: Path) -> pd.DataFrame:
    arcgis_frame = export_frame[
        [
            "sample_id",
            "POINT_X",
            "POINT_Y",
            "split",
            "observed_class",
            "model",
            "predicted_probability",
            "predicted_class",
        ]
    ].copy()
    arcgis_frame.rename(
        columns={
            "POINT_X": "x_coord",
            "POINT_Y": "y_coord",
            "predicted_probability": "susceptibility_value",
            "predicted_class": "susceptibility_class",
        },
        inplace=True,
    )
    arcgis_frame["threshold_used"] = float(threshold)
    ordered_columns = [
        "sample_id",
        "x_coord",
        "y_coord",
        "susceptibility_value",
        "susceptibility_class",
        "threshold_used",
        "split",
        "observed_class",
        "model",
    ]
    arcgis_frame = arcgis_frame[ordered_columns]
    arcgis_frame.to_csv(file_path, index=False)
    return arcgis_frame


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    ensure_directories(
        [
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.MODELS_DIR,
        ]
    )

    selected_model_name, metrics_row = resolve_model_choice(args.model_name, args.selection_metric)

    raw_source_path = args.data_path.resolve()
    raw_data_path = copy_raw_dataset(raw_source_path, config.RAW_DATA_PATH)
    raw_frame = load_raw_data(raw_data_path)

    prepared = prepare_dataset(
        raw_frame=raw_frame,
        target_column=config.TARGET_COLUMN,
        coordinate_columns=config.COORDINATE_COLUMNS,
        categorical_columns=config.CATEGORICAL_COLUMNS,
        cyclical_angle_columns=config.CYCLICAL_ANGLE_COLUMNS,
        spatial_block_size=config.SPATIAL_BLOCK_SIZE,
        spatial_folds=config.SPATIAL_FOLDS,
        validation_fold=config.VALIDATION_FOLD,
        test_fold=config.TEST_FOLD,
        seed=args.seed,
    )

    spec = get_model_spec(selected_model_name, args.n_jobs)
    result = train_single_model(
        prepared=prepared,
        spec=spec,
        models_dir=config.MODELS_DIR,
        seed=args.seed,
    )

    export_path = (
        config.PROCESSED_DATA_DIR / f"{slugify(selected_model_name)}_susceptibility_with_coordinates.csv"
    )
    arcgis_path = config.PROCESSED_DATA_DIR / f"{slugify(selected_model_name)}_arcgis_points.csv"
    export_frame = export_susceptibility_with_coordinates(
        model=result.model,
        feature_frame=prepared.full.features,
        metadata=prepared.full.metadata,
        threshold=result.threshold,
        model_name=result.name,
        file_path=export_path,
    )
    arcgis_frame = build_arcgis_export(export_frame, result.threshold, arcgis_path)

    summary = pd.DataFrame(
        [
            {
                "selected_model": selected_model_name,
                "selection_metric": args.selection_metric,
                "repo_metric_value": float(metrics_row[args.selection_metric]),
                "repo_threshold": float(metrics_row["threshold"]),
                "retrained_threshold": float(result.threshold),
                "full_rows_exported": int(len(arcgis_frame)),
                "detailed_export_path": str(export_path),
                "arcgis_export_path": str(arcgis_path),
            }
        ]
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
