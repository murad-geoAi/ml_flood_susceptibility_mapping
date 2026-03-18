from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.evaluate import choose_threshold, compute_binary_metrics
from src.utils import slugify

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

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


@dataclass
class ModelSpec:
    name: str
    family: str
    builder: Callable[[int], Any]
    fit_mode: str = "standard"


@dataclass
class ModelResult:
    name: str
    family: str
    model: Any
    threshold: float
    threshold_frame: pd.DataFrame
    validation_metrics: dict[str, float | int]
    test_metrics: dict[str, float | int]
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray
    model_path: Path


@dataclass
class BenchmarkResult:
    results: list[ModelResult]
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    primary_model_name: str
    benchmark_model_name: str


def build_model_specs(n_jobs: int) -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="Decision Tree",
            family="tree",
            builder=lambda seed: DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="Random Forest",
            family="tree_ensemble",
            builder=lambda seed: RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=4,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="Extra Trees",
            family="tree_ensemble",
            builder=lambda seed: ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=3,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=n_jobs,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="AdaBoost",
            family="boosting",
            builder=lambda seed: AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=2,
                    min_samples_leaf=20,
                    random_state=seed,
                ),
                n_estimators=400,
                learning_rate=0.05,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="Gradient Boosting",
            family="boosting",
            builder=lambda seed: GradientBoostingClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.8,
                random_state=seed,
            ),
        ),
        ModelSpec(
            name="HistGradientBoosting",
            family="boosting",
            builder=lambda seed: HistGradientBoostingClassifier(
                max_iter=500,
                learning_rate=0.05,
                max_depth=6,
                min_samples_leaf=20,
                l2_regularization=0.1,
                random_state=seed,
            ),
        ),
    ]
    if XGBClassifier is not None:
        specs.append(
            ModelSpec(
                name="XGBoost",
                family="boosting",
                builder=lambda seed: XGBClassifier(
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
                    random_state=seed,
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
                builder=lambda seed: LGBMClassifier(
                    n_estimators=2000,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    objective="binary",
                    random_state=seed,
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
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {type(model).__name__} does not expose probability predictions.")


def benchmark_models(
    prepared: Any,
    models_dir: Path,
    seed: int,
    n_jobs: int,
    primary_model_name: str,
    benchmark_model_name: str,
) -> BenchmarkResult:
    models_dir.mkdir(parents=True, exist_ok=True)
    sample_weight_train = compute_sample_weight(class_weight="balanced", y=prepared.train.labels)
    sample_weight_validation = compute_sample_weight(
        class_weight="balanced", y=prepared.validation.labels
    )

    results: list[ModelResult] = []
    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for spec in build_model_specs(n_jobs):
        model = spec.builder(seed)
        model = fit_model(
            spec=spec,
            model=model,
            x_train=prepared.train.features,
            y_train=prepared.train.labels,
            x_validation=prepared.validation.features,
            y_validation=prepared.validation.labels,
            sample_weight_train=sample_weight_train,
            sample_weight_validation=sample_weight_validation,
        )
        validation_probabilities = predict_probabilities(model, prepared.validation.features)
        test_probabilities = predict_probabilities(model, prepared.test.features)
        threshold, threshold_frame = choose_threshold(
            prepared.validation.labels,
            validation_probabilities,
        )
        validation_metrics = compute_binary_metrics(
            prepared.validation.labels,
            validation_probabilities,
            threshold,
        )
        test_metrics = compute_binary_metrics(
            prepared.test.labels,
            test_probabilities,
            threshold,
        )
        model_path = models_dir / f"{slugify(spec.name)}.joblib"
        joblib.dump(model, model_path)
        results.append(
            ModelResult(
                name=spec.name,
                family=spec.family,
                model=model,
                threshold=threshold,
                threshold_frame=threshold_frame,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                validation_probabilities=validation_probabilities,
                test_probabilities=test_probabilities,
                model_path=model_path,
            )
        )
        validation_rows.append(
            {"model": spec.name, "family": spec.family, "split": "validation", **validation_metrics}
        )
        test_rows.append({"model": spec.name, "family": spec.family, "split": "test", **test_metrics})

    validation_frame = pd.DataFrame(validation_rows).sort_values(
        ["roc_auc", "average_precision", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    test_frame = pd.DataFrame(test_rows).sort_values(
        ["roc_auc", "average_precision", "f1"],
        ascending=False,
    ).reset_index(drop=True)

    available_names = {result.name for result in results}
    if primary_model_name not in available_names:
        primary_model_name = str(validation_frame.iloc[0]["model"])
    if benchmark_model_name not in available_names:
        benchmark_model_name = str(test_frame.iloc[0]["model"])

    return BenchmarkResult(
        results=results,
        validation_frame=validation_frame,
        test_frame=test_frame,
        primary_model_name=primary_model_name,
        benchmark_model_name=benchmark_model_name,
    )


def get_model_result(benchmark: BenchmarkResult, model_name: str) -> ModelResult:
    for result in benchmark.results:
        if result.name == model_name:
            return result
    raise KeyError(f"Model {model_name} not found in benchmark results.")
