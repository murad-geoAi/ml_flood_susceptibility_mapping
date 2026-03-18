from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.spatial_split import build_split_labels, create_spatial_split


@dataclass
class SplitData:
    features: pd.DataFrame
    labels: np.ndarray
    metadata: pd.DataFrame
    raw_frame: pd.DataFrame


@dataclass
class PreparedDataset:
    feature_names: list[str]
    source_feature_names: list[str]
    categorical_feature_names: list[str]
    indicator_feature_names: list[str]
    continuous_feature_names: list[str]
    coordinate_columns: list[str]
    split_summary: dict[str, Any]
    train: SplitData
    validation: SplitData
    test: SplitData
    full: SplitData
    split_labels: pd.Series
    raw_frame: pd.DataFrame
    preprocessor: ColumnTransformer


def engineer_feature_frame(
    source_frame: pd.DataFrame,
    source_feature_names: list[str],
    cyclical_angle_columns: list[str],
) -> pd.DataFrame:
    feature_frame = source_frame[source_feature_names].apply(pd.to_numeric, errors="coerce").copy()
    for column in source_feature_names:
        feature_frame[f"{column}_missing"] = feature_frame[column].isna().astype(np.float32)
    for column in cyclical_angle_columns:
        if column not in feature_frame.columns:
            continue
        radians = np.deg2rad(feature_frame[column] % 360.0)
        feature_frame[f"{column}_sin"] = np.sin(radians)
        feature_frame[f"{column}_cos"] = np.cos(radians)
        feature_frame.drop(columns=column, inplace=True)
    return feature_frame


def format_categorical_feature_name(feature_name: str) -> str:
    if "_" not in feature_name:
        return feature_name
    column_name, category = feature_name.rsplit("_", maxsplit=1)
    if category.endswith(".0"):
        category = category[:-2]
    return f"{column_name}={category}"


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    categorical_feature_names: list[str],
    indicator_feature_names: list[str],
    continuous_feature_names: list[str],
) -> list[str]:
    transformed_names: list[str] = []
    if categorical_feature_names:
        encoder_pipeline = preprocessor.named_transformers_["cat"]
        encoder = encoder_pipeline.named_steps["encoder"]
        encoded_names = encoder.get_feature_names_out(categorical_feature_names)
        transformed_names.extend(format_categorical_feature_name(name) for name in encoded_names)
    transformed_names.extend(indicator_feature_names)
    transformed_names.extend(continuous_feature_names)
    return transformed_names


def build_split_data(
    raw_frame: pd.DataFrame,
    index: np.ndarray,
    split_name: str,
    source_feature_names: list[str],
    target_column: str,
    coordinate_columns: list[str],
    cyclical_angle_columns: list[str],
    feature_names: list[str],
    preprocessor: ColumnTransformer,
) -> SplitData:
    subset = raw_frame.iloc[index].reset_index(drop=True).copy()
    engineered = engineer_feature_frame(
        subset,
        source_feature_names,
        cyclical_angle_columns,
    )
    features = pd.DataFrame(
        preprocessor.transform(engineered),
        columns=feature_names,
    ).astype(np.float32)
    metadata = subset[["sample_id", *coordinate_columns, target_column]].copy()
    metadata.rename(columns={target_column: "observed_class"}, inplace=True)
    metadata["split"] = split_name
    return SplitData(
        features=features,
        labels=subset[target_column].astype(int).to_numpy(),
        metadata=metadata,
        raw_frame=subset,
    )


def prepare_dataset(
    raw_frame: pd.DataFrame,
    target_column: str,
    coordinate_columns: list[str],
    categorical_columns: list[str],
    cyclical_angle_columns: list[str],
    spatial_block_size: float,
    spatial_folds: int,
    validation_fold: int,
    test_fold: int,
    seed: int,
) -> PreparedDataset:
    original_row_count = len(raw_frame)
    working_frame = raw_frame.dropna(subset=[target_column]).copy()
    working_frame[target_column] = working_frame[target_column].astype(int)
    working_frame.insert(0, "sample_id", np.arange(len(working_frame), dtype=int))

    excluded_columns = {"sample_id", target_column, *coordinate_columns}
    source_feature_names = [
        column for column in working_frame.columns if column not in excluded_columns
    ]

    train_idx, validation_idx, test_idx, split_summary, groups = create_spatial_split(
        working_frame,
        working_frame[target_column],
        coordinate_columns,
        spatial_block_size,
        spatial_folds,
        validation_fold,
        test_fold,
        seed,
    )
    split_labels = build_split_labels(len(working_frame), train_idx, validation_idx, test_idx)

    training_features = engineer_feature_frame(
        working_frame.iloc[train_idx].reset_index(drop=True),
        source_feature_names,
        cyclical_angle_columns,
    )
    categorical_feature_names = [
        column for column in categorical_columns if column in training_features.columns
    ]
    indicator_feature_names = [
        column for column in training_features.columns if column.endswith("_missing")
    ]
    continuous_feature_names = [
        column
        for column in training_features.columns
        if column not in categorical_feature_names and column not in indicator_feature_names
    ]

    transformers: list[tuple[str, Any, list[str]]] = []
    if categorical_feature_names:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_feature_names,
            )
        )
    if indicator_feature_names:
        transformers.append(("indicator", "passthrough", indicator_feature_names))
    if continuous_feature_names:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                continuous_feature_names,
            )
        )
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    preprocessor.fit(training_features)
    feature_names = get_transformed_feature_names(
        preprocessor,
        categorical_feature_names,
        indicator_feature_names,
        continuous_feature_names,
    )

    train = build_split_data(
        working_frame,
        train_idx,
        "train",
        source_feature_names,
        target_column,
        coordinate_columns,
        cyclical_angle_columns,
        feature_names,
        preprocessor,
    )
    validation = build_split_data(
        working_frame,
        validation_idx,
        "validation",
        source_feature_names,
        target_column,
        coordinate_columns,
        cyclical_angle_columns,
        feature_names,
        preprocessor,
    )
    test = build_split_data(
        working_frame,
        test_idx,
        "test",
        source_feature_names,
        target_column,
        coordinate_columns,
        cyclical_angle_columns,
        feature_names,
        preprocessor,
    )
    full = build_split_data(
        working_frame,
        np.arange(len(working_frame)),
        "all",
        source_feature_names,
        target_column,
        coordinate_columns,
        cyclical_angle_columns,
        feature_names,
        preprocessor,
    )
    full.metadata["split"] = split_labels.values

    split_summary.update(
        {
            "dropped_rows_with_missing_target": int(original_row_count - len(working_frame)),
            "total_rows_after_target_drop": int(len(working_frame)),
            "source_feature_count": int(len(source_feature_names)),
            "model_feature_count": int(len(feature_names)),
            "features_used": source_feature_names,
            "model_features_used": feature_names,
            "categorical_feature_names": categorical_feature_names,
            "indicator_feature_names": indicator_feature_names,
            "continuous_feature_names": continuous_feature_names,
            "cyclical_angle_columns": [
                column for column in cyclical_angle_columns if column in source_feature_names
            ],
            "coordinates_used_as_features": False,
            "train_rows": int(len(train.labels)),
            "validation_rows": int(len(validation.labels)),
            "test_rows": int(len(test.labels)),
            "train_positive_rate": float(train.labels.mean()),
            "validation_positive_rate": float(validation.labels.mean()),
            "test_positive_rate": float(test.labels.mean()),
            "spatial_group_count_check": int(groups.nunique()),
        }
    )

    return PreparedDataset(
        feature_names=feature_names,
        source_feature_names=source_feature_names,
        categorical_feature_names=categorical_feature_names,
        indicator_feature_names=indicator_feature_names,
        continuous_feature_names=continuous_feature_names,
        coordinate_columns=coordinate_columns,
        split_summary=split_summary,
        train=train,
        validation=validation,
        test=test,
        full=full,
        split_labels=split_labels,
        raw_frame=working_frame,
        preprocessor=preprocessor,
    )


def export_processed_data(prepared: PreparedDataset, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)

    prepared.train.raw_frame.drop(columns=["sample_id"]).to_csv(
        processed_dir / "train.csv",
        index=False,
    )
    prepared.validation.raw_frame.drop(columns=["sample_id"]).to_csv(
        processed_dir / "val.csv",
        index=False,
    )
    prepared.test.raw_frame.drop(columns=["sample_id"]).to_csv(
        processed_dir / "test.csv",
        index=False,
    )

    processed_features = pd.concat(
        [
            prepared.full.metadata[
                ["sample_id", *prepared.coordinate_columns, "split", "observed_class"]
            ],
            prepared.full.features,
        ],
        axis=1,
    )
    processed_features.to_csv(processed_dir / "processed_features.csv", index=False)

    joblib.dump(
        {
            "preprocessor": prepared.preprocessor,
            "feature_names": prepared.feature_names,
            "source_feature_names": prepared.source_feature_names,
            "categorical_feature_names": prepared.categorical_feature_names,
            "indicator_feature_names": prepared.indicator_feature_names,
            "continuous_feature_names": prepared.continuous_feature_names,
            "coordinate_columns": prepared.coordinate_columns,
            "split_summary": prepared.split_summary,
        },
        processed_dir / "preprocessing.joblib",
    )
