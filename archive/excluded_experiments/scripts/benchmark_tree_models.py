from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ml_flood_susceptibility_mapping.scripts.train_dnn_flood_mapping import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    choose_threshold,
    compute_binary_metrics,
    prepare_dataset,
    seed_everything,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except Exception:
    lgb = None
    LGBMClassifier = None


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "tree_model_benchmark"


@dataclass
class ModelSpec:
    name: str
    family: str
    builder: Callable[[int], Any]
    fit_mode: str = "standard"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark tree-based and boosting classifiers for flood susceptibility "
            "mapping using the same cleaned preprocessing and split policy as the DNN pipeline."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-column", type=str, default="Class")
    parser.add_argument(
        "--coordinate-columns",
        nargs="+",
        default=["POINT_X", "POINT_Y"],
    )
    parser.add_argument("--include-coordinate-features", action="store_true")
    parser.add_argument(
        "--categorical-columns",
        nargs="*",
        default=["LULC"],
    )
    parser.add_argument(
        "--cyclical-angle-columns",
        nargs="*",
        default=["Aspect"],
    )
    parser.add_argument(
        "--split-strategy",
        choices=["spatial", "stratified"],
        default="spatial",
    )
    parser.add_argument("--spatial-block-size", type=float, default=5000.0)
    parser.add_argument("--spatial-folds", type=int, default=10)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Worker count for estimators that support threaded training.",
    )
    parser.add_argument(
        "--top-feature-count",
        type=int,
        default=15,
        help="Number of features to display for the best model importance plot.",
    )
    return parser


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def build_model_specs(seed: int, n_jobs: int) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="Decision Tree",
            family="tree",
            builder=lambda random_state: DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="Random Forest",
            family="tree_ensemble",
            builder=lambda random_state: RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="Extra Trees",
            family="tree_ensemble",
            builder=lambda random_state: ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="AdaBoost",
            family="boosting",
            builder=lambda random_state: AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=20,
                    random_state=random_state,
                ),
                n_estimators=400,
                learning_rate=0.05,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="Gradient Boosting",
            family="boosting",
            builder=lambda random_state: GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=random_state,
            ),
        ),
        ModelSpec(
            name="HistGradientBoosting",
            family="boosting",
            builder=lambda random_state: HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.05,
                max_depth=6,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=random_state,
            ),
        ),
    ]
    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                name="XGBoost",
                family="boosting",
                builder=lambda random_state: XGBClassifier(
                    n_estimators=2000,
                    learning_rate=0.03,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric=["aucpr", "auc"],
                    early_stopping_rounds=80,
                    tree_method="hist",
                    n_jobs=n_jobs,
                    random_state=random_state,
                    verbosity=0,
                ),
                fit_mode="xgboost",
            )
        )
    if LGBMClassifier is not None:
        specs.append(
            ModelSpec(
                name="LightGBM",
                family="boosting",
                builder=lambda random_state: LGBMClassifier(
                    n_estimators=2000,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    objective="binary",
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbosity=-1,
                ),
                fit_mode="lightgbm",
            )
        )
    return specs


def fit_model(
    spec: ModelSpec,
    model: Any,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_validation: pd.DataFrame,
    y_validation: np.ndarray,
    sample_weight_train: np.ndarray,
    sample_weight_validation: np.ndarray,
) -> Any:
    if spec.fit_mode == "xgboost":
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(x_validation, y_validation)],
            sample_weight_eval_set=[sample_weight_validation],
            verbose=False,
        )
        return model
    if spec.fit_mode == "lightgbm":
        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        model.fit(
            x_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(x_validation, y_validation)],
            eval_names=["validation"],
            eval_sample_weight=[sample_weight_validation],
            eval_metric="auc",
            callbacks=callbacks,
        )
        return model
    model.fit(x_train, y_train, sample_weight=sample_weight_train)
    return model


def predict_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        return np.asarray(probabilities, dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {type(model).__name__} does not expose probability predictions.")


def save_predictions(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
    file_path: Path,
) -> None:
    prediction_frame = metadata.copy()
    prediction_frame["model"] = model_name
    prediction_frame["predicted_probability"] = probabilities
    prediction_frame["predicted_class"] = (probabilities >= threshold).astype(int)
    prediction_frame.to_csv(file_path, index=False)


def create_metric_rows(
    model_name: str,
    family: str,
    split_name: str,
    metrics: dict[str, float | int],
) -> dict[str, Any]:
    return {"model": model_name, "family": family, "split": split_name, **metrics}


def plot_model_comparison(metrics_frame: pd.DataFrame, figure_path: Path) -> None:
    test_frame = (
        metrics_frame.loc[metrics_frame["split"] == "test", ["model", "roc_auc", "average_precision"]]
        .sort_values(["roc_auc", "average_precision"], ascending=False)
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    sns.barplot(
        data=test_frame,
        x="roc_auc",
        y="model",
        hue="model",
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Test ROC-AUC")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_ylabel("")

    sns.barplot(
        data=test_frame,
        x="average_precision",
        y="model",
        hue="model",
        palette="flare",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Test Average Precision")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Average precision")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def extract_feature_importance(model: Any, feature_names: list[str]) -> pd.DataFrame | None:
    if not hasattr(model, "feature_importances_"):
        return None
    importance = np.asarray(model.feature_importances_, dtype=float)
    if importance.shape[0] != len(feature_names):
        return None
    importance_frame = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    return importance_frame


def plot_feature_importance(
    importance_frame: pd.DataFrame,
    model_name: str,
    top_feature_count: int,
    figure_path: Path,
) -> None:
    top_frame = importance_frame.head(top_feature_count).iloc[::-1]
    fig, axis = plt.subplots(figsize=(9, 6), dpi=300)
    sns.barplot(
        data=top_frame,
        x="importance",
        y="feature",
        hue="feature",
        palette="crest",
        legend=False,
        ax=axis,
    )
    axis.set_title(f"{model_name} Feature Importance")
    axis.set_xlabel("Importance")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def benchmark_models(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_dataset(args)
    x_train = prepared.train.features_display.copy()
    y_train = prepared.train.labels.copy()
    x_validation = prepared.validation.features_display.copy()
    y_validation = prepared.validation.labels.copy()
    x_test = prepared.test.features_display.copy()
    y_test = prepared.test.labels.copy()

    output_dir = args.output_dir
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    for directory in (reports_dir, predictions_dir, figures_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)
    sample_weight_validation = compute_sample_weight(class_weight="balanced", y=y_validation)

    metric_rows: list[dict[str, Any]] = []
    best_validation_model_name = ""
    best_validation_score = float("-inf")
    best_validation_ap = float("-inf")
    best_model_for_importance: tuple[str, Any] | None = None
    failed_models: list[dict[str, str]] = []

    model_specs = build_model_specs(args.seed, args.n_jobs)
    for spec in model_specs:
        try:
            model = spec.builder(args.seed)
            model = fit_model(
                spec=spec,
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                sample_weight_train=sample_weight_train,
                sample_weight_validation=sample_weight_validation,
            )

            validation_probabilities = predict_probabilities(model, x_validation)
            test_probabilities = predict_probabilities(model, x_test)
            threshold, threshold_frame = choose_threshold(y_validation, validation_probabilities)
            threshold_frame.to_csv(
                reports_dir / f"{slugify(spec.name)}_validation_threshold_sweep.csv",
                index=False,
            )

            validation_metrics = compute_binary_metrics(
                y_validation,
                validation_probabilities,
                threshold=threshold,
            )
            test_metrics = compute_binary_metrics(
                y_test,
                test_probabilities,
                threshold=threshold,
            )

            metric_rows.append(
                create_metric_rows(spec.name, spec.family, "validation", validation_metrics)
            )
            metric_rows.append(
                create_metric_rows(spec.name, spec.family, "test", test_metrics)
            )

            joblib.dump(model, models_dir / f"{slugify(spec.name)}.joblib")
            save_predictions(
                prepared.validation.metadata,
                validation_probabilities,
                threshold,
                spec.name,
                predictions_dir / f"{slugify(spec.name)}_validation_predictions.csv",
            )
            save_predictions(
                prepared.test.metadata,
                test_probabilities,
                threshold,
                spec.name,
                predictions_dir / f"{slugify(spec.name)}_test_predictions.csv",
            )

            validation_score = float(validation_metrics["roc_auc"])
            validation_ap = float(validation_metrics["average_precision"])
            if (validation_score, validation_ap) > (best_validation_score, best_validation_ap):
                best_validation_model_name = spec.name
                best_validation_score = validation_score
                best_validation_ap = validation_ap
                best_model_for_importance = (spec.name, model)
        except Exception as exc:
            failed_models.append({"model": spec.name, "reason": str(exc)})

    if not metric_rows:
        raise RuntimeError("No benchmark models completed successfully.")

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(reports_dir / "metrics_summary.csv", index=False)

    validation_ranking = (
        metrics_frame.loc[metrics_frame["split"] == "validation"]
        .sort_values(["roc_auc", "average_precision", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    test_ranking = (
        metrics_frame.loc[metrics_frame["split"] == "test"]
        .sort_values(["roc_auc", "average_precision", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    validation_ranking.to_csv(reports_dir / "validation_ranking.csv", index=False)
    test_ranking.to_csv(reports_dir / "test_ranking.csv", index=False)
    plot_model_comparison(metrics_frame, figures_dir / "test_model_comparison.png")

    best_model_importance_path = None
    if best_model_for_importance is not None:
        best_name, best_model = best_model_for_importance
        importance_frame = extract_feature_importance(best_model, prepared.feature_names)
        if importance_frame is not None:
            importance_csv_path = figures_dir / "best_model_feature_importance.csv"
            importance_png_path = figures_dir / "best_model_feature_importance.png"
            importance_frame.to_csv(importance_csv_path, index=False)
            plot_feature_importance(
                importance_frame,
                best_name,
                args.top_feature_count,
                importance_png_path,
            )
            best_model_importance_path = {
                "csv": str(importance_csv_path),
                "figure": str(importance_png_path),
            }

    summary = {
        "split_summary": prepared.split_summary,
        "model_count": len(model_specs),
        "models_benchmarked": [spec.name for spec in model_specs],
        "completed_model_count": int(metrics_frame["model"].nunique()),
        "best_validation_model": best_validation_model_name,
        "best_validation_roc_auc": best_validation_score,
        "best_validation_average_precision": best_validation_ap,
        "failed_models": failed_models,
        "reports": {
            "metrics_summary": str(reports_dir / "metrics_summary.csv"),
            "validation_ranking": str(reports_dir / "validation_ranking.csv"),
            "test_ranking": str(reports_dir / "test_ranking.csv"),
        },
        "figures": {
            "comparison": str(figures_dir / "test_model_comparison.png"),
            "best_model_importance": best_model_importance_path,
        },
        "models_dir": str(models_dir),
        "predictions_dir": str(predictions_dir),
    }
    with (reports_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    parser = build_argument_parser()
    args = parser.parse_args()
    args.data_path = args.data_path.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            handle,
            indent=2,
        )

    seed_everything(args.seed)
    summary = benchmark_models(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
