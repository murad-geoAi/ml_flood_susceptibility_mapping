from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import (
    build_dataset_insight_tables,
    copy_raw_dataset,
    load_raw_data,
    summarize_predictor_ranges,
)
from src.evaluate import (
    plot_calibration_curve,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_lulc_flood_rate,
    plot_missingness,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_probability_distribution,
    plot_roc_curve,
    plot_spatial_split_map,
    plot_susceptibility_map,
    plot_threshold_sweep,
)
from src.explainability import compute_tree_shap_outputs, save_feature_importance
from src.export_outputs import (
    create_conference_docx,
    export_susceptibility_with_coordinates,
    save_prediction_frame,
)
from src.preprocessing import export_processed_data, prepare_dataset
from src.train_models import benchmark_models, get_model_result, predict_probabilities
from src.utils import ensure_directories, set_seed, slugify, write_json


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full classical-ML flood susceptibility package."
    )
    parser.add_argument("--data-path", type=Path, default=config.DEFAULT_RAW_DATA_INPUT_PATH)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--shap-samples", type=int, default=512)
    parser.add_argument("--top-feature-count", type=int, default=15)
    return parser


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    parser = build_argument_parser()
    args = parser.parse_args()
    set_seed(args.seed)

    ensure_directories(
        [
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.OUTPUTS_DIR,
            config.FIGURES_DIR,
            config.TABLES_DIR,
            config.METRICS_DIR,
            config.SHAP_DIR,
            config.MAPS_DIR,
            config.MODELS_DIR,
            config.DOCS_DIR,
        ]
    )

    raw_source_path = args.data_path.resolve()
    raw_data_path = copy_raw_dataset(raw_source_path, config.RAW_DATA_PATH)
    raw_frame = load_raw_data(raw_data_path)
    insights = build_dataset_insight_tables(raw_frame)
    labeled_frame = insights["labeled_frame"]
    predictor_ranges = summarize_predictor_ranges(labeled_frame)

    insights["summary"].to_csv(config.TABLES_DIR / "dataset_summary.csv", index=False)
    insights["missingness"].to_csv(config.TABLES_DIR / "missingness_summary.csv", index=False)
    insights["lulc_flood_rate"].to_csv(config.TABLES_DIR / "lulc_flood_rate.csv", index=False)
    insights["correlation"].to_csv(config.TABLES_DIR / "class_correlation_summary.csv", index=False)
    insights["skewness"].to_csv(config.TABLES_DIR / "predictor_skewness.csv", index=False)
    predictor_ranges.to_csv(config.TABLES_DIR / "predictor_range_summary.csv", index=False)

    plot_class_distribution(
        labeled_frame[config.TARGET_COLUMN].to_numpy(),
        config.FIGURES_DIR / "class_distribution.png",
    )
    plot_missingness(insights["missingness"], config.FIGURES_DIR / "missingness_profile.png")
    plot_lulc_flood_rate(
        insights["lulc_flood_rate"],
        config.FIGURES_DIR / "lulc_flood_rate.png",
    )

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
    export_processed_data(prepared, config.PROCESSED_DATA_DIR)

    plot_spatial_split_map(
        prepared.full.metadata[["POINT_X", "POINT_Y", "split"]],
        config.COORDINATE_COLUMNS,
        config.FIGURES_DIR / "spatial_split_map.png",
    )
    plot_correlation_heatmap(
        prepared.train.features,
        config.FIGURES_DIR / "processed_feature_correlation_heatmap.png",
    )

    benchmark = benchmark_models(
        prepared=prepared,
        models_dir=config.MODELS_DIR,
        seed=args.seed,
        n_jobs=args.n_jobs,
        primary_model_name=config.PRIMARY_MODEL_NAME,
        benchmark_model_name=config.BENCHMARK_MODEL_NAME,
    )

    benchmark.validation_frame.to_csv(
        config.METRICS_DIR / "validation_model_metrics.csv",
        index=False,
    )
    benchmark.test_frame.to_csv(
        config.METRICS_DIR / "test_model_metrics.csv",
        index=False,
    )

    paper_validation_table = benchmark.validation_frame[
        ["model", "roc_auc", "average_precision", "f1"]
    ].copy()
    paper_test_table = benchmark.test_frame[
        [
            "model",
            "family",
            "roc_auc",
            "average_precision",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
        ]
    ].copy()
    paper_validation_table.to_csv(
        config.TABLES_DIR / "table_validation_summary.csv",
        index=False,
    )
    paper_test_table.to_csv(
        config.TABLES_DIR / "table_test_summary.csv",
        index=False,
    )

    plot_model_comparison(benchmark.test_frame, config.FIGURES_DIR / "model_comparison.png")

    for result in benchmark.results:
        result.threshold_frame.to_csv(
            config.METRICS_DIR / f"{slugify(result.name)}_validation_threshold_sweep.csv",
            index=False,
        )
        save_prediction_frame(
            prepared.validation.metadata,
            result.validation_probabilities,
            result.threshold,
            result.name,
            config.METRICS_DIR / f"{slugify(result.name)}_validation_predictions.csv",
        )
        save_prediction_frame(
            prepared.test.metadata,
            result.test_probabilities,
            result.threshold,
            result.name,
            config.METRICS_DIR / f"{slugify(result.name)}_test_predictions.csv",
        )

    primary_result = get_model_result(benchmark, benchmark.primary_model_name)
    benchmark_result = get_model_result(benchmark, benchmark.benchmark_model_name)

    for result, label in [
        (primary_result, "primary"),
        (benchmark_result, "benchmark"),
    ]:
        plot_threshold_sweep(
            result.threshold_frame,
            config.FIGURES_DIR / f"{slugify(result.name)}_threshold_sweep.png",
            f"{result.name} Validation Threshold Sweep",
        )
        plot_roc_curve(
            prepared.test.labels,
            result.test_probabilities,
            config.FIGURES_DIR / f"{slugify(result.name)}_roc_curve.png",
            f"{result.name} Test ROC Curve",
        )
        plot_precision_recall_curve(
            prepared.test.labels,
            result.test_probabilities,
            config.FIGURES_DIR / f"{slugify(result.name)}_precision_recall_curve.png",
            f"{result.name} Test Precision-Recall Curve",
        )
        plot_confusion_matrix(
            prepared.test.labels,
            result.test_probabilities,
            result.threshold,
            config.FIGURES_DIR / f"{slugify(result.name)}_confusion_matrix.png",
            f"{result.name} Test Confusion Matrix",
        )
        plot_calibration_curve(
            prepared.test.labels,
            result.test_probabilities,
            config.FIGURES_DIR / f"{slugify(result.name)}_calibration_curve.png",
            f"{result.name} Test Calibration Curve",
        )
        plot_probability_distribution(
            prepared.test.labels,
            result.test_probabilities,
            config.FIGURES_DIR / f"{slugify(result.name)}_probability_distribution.png",
            f"{result.name} Test Probability Distribution",
        )
        plot_susceptibility_map(
            prepared.test.metadata,
            result.test_probabilities,
            config.COORDINATE_COLUMNS,
            config.MAPS_DIR / f"{slugify(result.name)}_test_susceptibility_map.png",
            f"{result.name} Test Susceptibility Map",
        )
        save_feature_importance(
            result.model,
            prepared.feature_names,
            config.SHAP_DIR / f"{slugify(result.name)}_feature_importance.csv",
            config.SHAP_DIR / f"{slugify(result.name)}_feature_importance.png",
            f"{result.name} Feature Importance",
            top_n=args.top_feature_count,
        )
        compute_tree_shap_outputs(
            result.model,
            prepared.test.features,
            config.SHAP_DIR,
            slugify(result.name),
            result.name,
            max_samples=args.shap_samples,
            seed=args.seed,
        )

    full_probabilities = predict_probabilities(primary_result.model, prepared.full.features)
    export_susceptibility_with_coordinates(
        primary_result.model,
        prepared.full.features,
        prepared.full.metadata,
        primary_result.threshold,
        primary_result.name,
        config.PROCESSED_DATA_DIR / "susceptibility_with_coordinates.csv",
    )
    plot_susceptibility_map(
        prepared.full.metadata,
        full_probabilities,
        config.COORDINATE_COLUMNS,
        config.MAPS_DIR / f"{slugify(primary_result.name)}_full_susceptibility_map.png",
        f"{primary_result.name} Full Susceptibility Map",
    )

    study_summary = {
        "dataset_summary_path": str(config.TABLES_DIR / "dataset_summary.csv"),
        "processed_data_dir": str(config.PROCESSED_DATA_DIR),
        "primary_model": primary_result.name,
        "benchmark_model": benchmark_result.name,
        "primary_model_threshold": primary_result.threshold,
        "benchmark_model_threshold": benchmark_result.threshold,
        "primary_model_metrics": primary_result.test_metrics,
        "benchmark_model_metrics": benchmark_result.test_metrics,
        "validation_top_model": str(benchmark.validation_frame.iloc[0]["model"]),
        "test_top_roc_auc_model": str(
            benchmark.test_frame.sort_values("roc_auc", ascending=False).iloc[0]["model"]
        ),
        "test_top_average_precision_model": str(
            benchmark.test_frame.sort_values("average_precision", ascending=False).iloc[0]["model"]
        ),
    }
    write_json(study_summary, config.METRICS_DIR / "study_summary.json")

    create_conference_docx(
        summary_frame=insights["summary"],
        test_metrics_frame=paper_test_table,
        validation_metrics_frame=paper_validation_table,
        docs_dir=config.DOCS_DIR,
        figures_dir=config.FIGURES_DIR,
    )

    print(pd.DataFrame([study_summary]).to_string(index=False))


if __name__ == "__main__":
    main()
