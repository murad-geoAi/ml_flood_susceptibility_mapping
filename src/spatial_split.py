from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_spatial_groups(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    block_size: float,
) -> pd.Series:
    grid_x = np.floor((frame[x_column] - frame[x_column].min()) / block_size).astype(int)
    grid_y = np.floor((frame[y_column] - frame[y_column].min()) / block_size).astype(int)
    return grid_x.astype(str) + "_" + grid_y.astype(str)


def create_spatial_split(
    frame: pd.DataFrame,
    labels: pd.Series,
    coordinate_columns: list[str],
    block_size: float,
    spatial_folds: int,
    validation_fold: int,
    test_fold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], pd.Series]:
    if validation_fold == test_fold:
        raise ValueError("Validation fold and test fold must be different.")
    if len(coordinate_columns) < 2:
        raise ValueError("Spatial splitting requires two coordinate columns.")

    groups = make_spatial_groups(
        frame,
        coordinate_columns[0],
        coordinate_columns[1],
        block_size,
    )
    splitter = StratifiedGroupKFold(
        n_splits=spatial_folds,
        shuffle=True,
        random_state=seed,
    )
    fold_indices = [test_index for _, test_index in splitter.split(frame, labels, groups)]
    validation_idx = fold_indices[validation_fold]
    test_idx = fold_indices[test_fold]
    mask = np.ones(len(frame), dtype=bool)
    mask[validation_idx] = False
    mask[test_idx] = False
    train_idx = np.where(mask)[0]

    split_summary = {
        "strategy": "spatial",
        "spatial_block_size": block_size,
        "spatial_folds": spatial_folds,
        "validation_fold": validation_fold,
        "test_fold": test_fold,
        "unique_spatial_groups": int(groups.nunique()),
        "train_groups": int(groups.iloc[train_idx].nunique()),
        "validation_groups": int(groups.iloc[validation_idx].nunique()),
        "test_groups": int(groups.iloc[test_idx].nunique()),
    }
    return train_idx, validation_idx, test_idx, split_summary, groups


def build_split_labels(
    length: int,
    train_idx: np.ndarray,
    validation_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pd.Series:
    labels = np.full(length, "train", dtype=object)
    labels[validation_idx] = "validation"
    labels[test_idx] = "test"
    return pd.Series(labels, name="split")
