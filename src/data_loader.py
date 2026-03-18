from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.config import COORDINATE_COLUMNS, TARGET_COLUMN


def copy_raw_dataset(source_path: Path, destination_path: Path) -> Path:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != destination_path.resolve():
        shutil.copy2(source_path, destination_path)
    return destination_path


def load_raw_data(data_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(data_path)
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def build_dataset_insight_tables(
    raw_frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    coordinate_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    if coordinate_columns is None:
        coordinate_columns = COORDINATE_COLUMNS

    original_rows = len(raw_frame)
    missing_target_rows = int(raw_frame[target_column].isna().sum())
    labeled_frame = raw_frame.dropna(subset=[target_column]).copy()
    labeled_frame[target_column] = labeled_frame[target_column].astype(int)

    summary_frame = pd.DataFrame(
        [
            ("original_rows", int(original_rows)),
            ("missing_target_rows", int(missing_target_rows)),
            ("labeled_rows", int(len(labeled_frame))),
            ("class_0_count", int((labeled_frame[target_column] == 0).sum())),
            ("class_1_count", int((labeled_frame[target_column] == 1).sum())),
            ("positive_rate", float(labeled_frame[target_column].mean())),
            ("duplicate_rows", int(labeled_frame.duplicated().sum())),
            (
                "duplicate_coordinate_pairs",
                int(labeled_frame.duplicated(subset=coordinate_columns).sum()),
            ),
            ("point_x_min", float(labeled_frame[coordinate_columns[0]].min())),
            ("point_x_max", float(labeled_frame[coordinate_columns[0]].max())),
            ("point_y_min", float(labeled_frame[coordinate_columns[1]].min())),
            ("point_y_max", float(labeled_frame[coordinate_columns[1]].max())),
        ],
        columns=["metric", "value"],
    )

    missingness_frame = (
        labeled_frame.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_pct=lambda df: (df["missing_count"] / len(labeled_frame)) * 100.0)
        .reset_index(names="feature")
        .sort_values(["missing_count", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    lulc_flood_rate_frame = (
        labeled_frame.groupby("LULC", dropna=False)[target_column]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "flood_rate"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    correlation_frame = (
        labeled_frame.corr(numeric_only=True)[target_column]
        .drop(target_column)
        .rename("pearson_correlation")
        .to_frame()
        .reset_index(names="feature")
        .assign(abs_correlation=lambda df: df["pearson_correlation"].abs())
        .sort_values(["abs_correlation", "feature"], ascending=[False, True])
        .drop(columns="abs_correlation")
        .reset_index(drop=True)
    )

    skewness_frame = (
        labeled_frame.drop(columns=[target_column])
        .skew(numeric_only=True)
        .rename("skewness")
        .to_frame()
        .reset_index(names="feature")
        .assign(abs_skewness=lambda df: df["skewness"].abs())
        .sort_values(["abs_skewness", "feature"], ascending=[False, True])
        .drop(columns="abs_skewness")
        .reset_index(drop=True)
    )

    return {
        "summary": summary_frame,
        "missingness": missingness_frame,
        "lulc_flood_rate": lulc_flood_rate_frame,
        "correlation": correlation_frame,
        "skewness": skewness_frame,
        "labeled_frame": labeled_frame,
    }


def summarize_predictor_ranges(frame: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    predictor_frame = frame.drop(columns=[target_column], errors="ignore")
    return predictor_frame.describe(include="all").transpose().reset_index(names="feature")
