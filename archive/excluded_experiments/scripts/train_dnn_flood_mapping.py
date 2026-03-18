from __future__ import annotations

import argparse
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import shap
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "Flood_data.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "processed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dnn_flood_mapping"


@dataclass
class SplitBundle:
    features_scaled: np.ndarray
    features_display: pd.DataFrame
    labels: np.ndarray
    metadata: pd.DataFrame


@dataclass
class PreparedData:
    feature_names: list[str]
    source_feature_names: list[str]
    categorical_feature_names: list[str]
    indicator_feature_names: list[str]
    continuous_feature_names: list[str]
    continuous_feature_start_index: int
    coordinate_columns: list[str]
    split_strategy: str
    split_summary: dict[str, Any]
    train: SplitBundle
    validation: SplitBundle
    test: SplitBundle
    feature_transformer: ColumnTransformer
    scaler: StandardScaler


class FloodTensorDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        training: bool = False,
        gaussian_noise_std: float = 0.0,
        continuous_feature_start_index: int | None = None,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.training = training
        self.gaussian_noise_std = gaussian_noise_std
        self.continuous_feature_start_index = continuous_feature_start_index

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[index]
        if self.training and self.gaussian_noise_std > 0.0:
            if self.continuous_feature_start_index is None:
                x = x + torch.randn_like(x) * self.gaussian_noise_std
            else:
                x = x.clone()
                x[self.continuous_feature_start_index :] = (
                    x[self.continuous_feature_start_index :]
                    + torch.randn_like(x[self.continuous_feature_start_index :])
                    * self.gaussian_noise_std
                )
        return x, self.labels[index]


class FloodDataModule(pl.LightningDataModule):
    def __init__(
        self,
        prepared: PreparedData,
        batch_size: int,
        num_workers: int,
        gaussian_noise_std: float,
    ) -> None:
        super().__init__()
        self.prepared = prepared
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.gaussian_noise_std = gaussian_noise_std

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = FloodTensorDataset(
            self.prepared.train.features_scaled,
            self.prepared.train.labels,
            training=True,
            gaussian_noise_std=self.gaussian_noise_std,
            continuous_feature_start_index=self.prepared.continuous_feature_start_index,
        )
        self.validation_dataset = FloodTensorDataset(
            self.prepared.validation.features_scaled,
            self.prepared.validation.labels,
            training=False,
        )
        self.test_dataset = FloodTensorDataset(
            self.prepared.test.features_scaled,
            self.prepared.test.labels,
            training=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class FloodDNN(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        learning_rate: float,
        dropout: float,
        weight_decay: float,
        pos_weight: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.validation_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=inputs.size(0),
        )
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)
        probabilities = torch.sigmoid(logits)
        self.validation_outputs.append(
            (probabilities.detach().cpu(), targets.detach().cpu())
        )
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=inputs.size(0),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.validation_outputs:
            return
        probabilities = torch.cat([chunk[0] for chunk in self.validation_outputs]).numpy()
        targets = torch.cat([chunk[1] for chunk in self.validation_outputs]).numpy()
        metrics = compute_binary_metrics(targets, probabilities, threshold=0.5)
        self.log("val_auc", metrics["roc_auc"], prog_bar=True, sync_dist=False)
        self.log("val_ap", metrics["average_precision"], prog_bar=True, sync_dist=False)
        self.validation_outputs.clear()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc",
            },
        }


class ProbabilityWrapper(nn.Module):
    def __init__(self, model: FloodDNN) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(inputs)).unsqueeze(1)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch Lightning DNN for flood susceptibility mapping with "
            "leakage-safe preprocessing, Gaussian noise augmentation, and paper-ready figures."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-column", type=str, default="Class")
    parser.add_argument(
        "--coordinate-columns",
        nargs="+",
        default=["POINT_X", "POINT_Y"],
        help="Coordinate columns reserved for spatial splitting and map figures.",
    )
    parser.add_argument(
        "--include-coordinate-features",
        action="store_true",
        help="Include coordinates as model inputs. Disabled by default to reduce leakage risk.",
    )
    parser.add_argument(
        "--categorical-columns",
        nargs="*",
        default=["LULC"],
        help="Columns that should be encoded as categorical classes instead of scaled as continuous values.",
    )
    parser.add_argument(
        "--cyclical-angle-columns",
        nargs="*",
        default=["Aspect"],
        help="Angular columns in degrees that should be expanded to sine/cosine features.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["spatial", "stratified"],
        default="spatial",
        help="Spatial blocks are safer for susceptibility mapping because nearby points stay together.",
    )
    parser.add_argument("--spatial-block-size", type=float, default=5000.0)
    parser.add_argument("--spatial-folds", type=int, default=10)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--test-fold", type=int, default=1)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.05)
    parser.add_argument("--background-samples", type=int, default=256)
    parser.add_argument("--explain-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed, workers=True)


def safe_score(metric_fn: Any, *args: Any, default: float = float("nan"), **kwargs: Any) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except ValueError:
        return default


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-7
    clipped_probabilities = np.clip(y_prob, eps, 1.0 - eps)
    return {
        "roc_auc": safe_score(roc_auc_score, y_true, y_prob),
        "average_precision": safe_score(average_precision_score, y_true, y_prob),
        "accuracy": safe_score(accuracy_score, y_true, y_pred),
        "balanced_accuracy": safe_score(balanced_accuracy_score, y_true, y_pred),
        "precision": safe_score(precision_score, y_true, y_pred, zero_division=0),
        "recall": safe_score(recall_score, y_true, y_pred, zero_division=0),
        "f1": safe_score(f1_score, y_true, y_pred, zero_division=0),
        "mcc": safe_score(matthews_corrcoef, y_true, y_pred),
        "brier_score": safe_score(brier_score_loss, y_true, y_prob),
        "log_loss": safe_score(log_loss, y_true, clipped_probabilities),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def make_spatial_groups(
    frame: pd.DataFrame, x_column: str, y_column: str, block_size: float
) -> pd.Series:
    grid_x = np.floor((frame[x_column] - frame[x_column].min()) / block_size).astype(int)
    grid_y = np.floor((frame[y_column] - frame[y_column].min()) / block_size).astype(int)
    return grid_x.astype(str) + "_" + grid_y.astype(str)


def create_split_indices(
    frame: pd.DataFrame,
    labels: pd.Series,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if args.split_strategy == "spatial":
        if len(args.coordinate_columns) < 2:
            raise ValueError("Spatial splitting requires two coordinate columns.")
        if args.validation_fold == args.test_fold:
            raise ValueError("Validation fold and test fold must be different.")
        groups = make_spatial_groups(
            frame,
            args.coordinate_columns[0],
            args.coordinate_columns[1],
            args.spatial_block_size,
        )
        splitter = StratifiedGroupKFold(
            n_splits=args.spatial_folds,
            shuffle=True,
            random_state=args.seed,
        )
        fold_indices = [test_idx for _, test_idx in splitter.split(frame, labels, groups)]
        validation_idx = fold_indices[args.validation_fold]
        test_idx = fold_indices[args.test_fold]
        mask = np.ones(len(frame), dtype=bool)
        mask[validation_idx] = False
        mask[test_idx] = False
        train_idx = np.where(mask)[0]
        split_summary = {
            "strategy": "spatial",
            "spatial_block_size": args.spatial_block_size,
            "spatial_folds": args.spatial_folds,
            "validation_fold": args.validation_fold,
            "test_fold": args.test_fold,
            "unique_spatial_groups": int(groups.nunique()),
            "train_groups": int(groups.iloc[train_idx].nunique()),
            "validation_groups": int(groups.iloc[validation_idx].nunique()),
            "test_groups": int(groups.iloc[test_idx].nunique()),
        }
        return train_idx, validation_idx, test_idx, split_summary

    all_indices = np.arange(len(frame))
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=args.validation_size + args.test_size,
        stratify=labels,
        random_state=args.seed,
    )
    temp_labels = labels.iloc[temp_idx]
    relative_test_size = args.test_size / (args.validation_size + args.test_size)
    validation_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=args.seed,
    )
    split_summary = {
        "strategy": "stratified",
        "validation_size": args.validation_size,
        "test_size": args.test_size,
    }
    return train_idx, validation_idx, test_idx, split_summary


def build_split_bundle(
    source_frame: pd.DataFrame,
    index: np.ndarray,
    source_feature_names: list[str],
    target_column: str,
    coordinate_columns: list[str],
    cyclical_angle_columns: list[str],
    feature_transformer: ColumnTransformer,
    scaler: StandardScaler,
    categorical_feature_names: list[str],
    indicator_feature_names: list[str],
    continuous_feature_names: list[str],
    transformed_feature_names: list[str],
) -> SplitBundle:
    subset = source_frame.iloc[index].reset_index(drop=True)
    engineered_features = engineer_feature_frame(
        subset,
        source_feature_names,
        cyclical_angle_columns,
    )
    unscaled_features = feature_transformer.transform(engineered_features).astype(np.float32)
    scaled = apply_scaler_to_continuous_block(
        unscaled_features,
        scaler,
        continuous_feature_names,
    )
    display_features = pd.DataFrame(unscaled_features, columns=transformed_feature_names)
    metadata_columns = ["sample_id", *coordinate_columns, target_column]
    metadata = subset[metadata_columns].copy()
    metadata.rename(columns={target_column: "observed_class"}, inplace=True)
    return SplitBundle(
        features_scaled=scaled.astype(np.float32),
        features_display=display_features,
        labels=subset[target_column].astype(int).to_numpy(),
        metadata=metadata,
    )


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
    feature_transformer: ColumnTransformer,
    categorical_feature_names: list[str],
    indicator_feature_names: list[str],
    continuous_feature_names: list[str],
) -> list[str]:
    transformed_names: list[str] = []
    if categorical_feature_names:
        categorical_pipeline = feature_transformer.named_transformers_["cat"]
        encoder = categorical_pipeline.named_steps["encoder"]
        encoded_names = encoder.get_feature_names_out(categorical_feature_names)
        transformed_names.extend(
            format_categorical_feature_name(name) for name in encoded_names.tolist()
        )
    transformed_names.extend(indicator_feature_names)
    transformed_names.extend(continuous_feature_names)
    return transformed_names


def apply_scaler_to_continuous_block(
    features: np.ndarray,
    scaler: StandardScaler,
    continuous_feature_names: list[str],
) -> np.ndarray:
    scaled = features.copy()
    if not continuous_feature_names:
        return scaled
    start_index = features.shape[1] - len(continuous_feature_names)
    scaled[:, start_index:] = scaler.transform(features[:, start_index:])
    return scaled


def prepare_dataset(args: argparse.Namespace) -> PreparedData:
    raw_frame = pd.read_csv(args.data_path)
    raw_frame.columns = [column.strip() for column in raw_frame.columns]
    original_row_count = len(raw_frame)
    raw_frame = raw_frame.dropna(subset=[args.target_column]).copy()
    raw_frame[args.target_column] = raw_frame[args.target_column].astype(int)
    raw_frame.insert(0, "sample_id", np.arange(len(raw_frame), dtype=int))

    excluded_columns = {"sample_id", args.target_column}
    if not args.include_coordinate_features:
        excluded_columns.update(args.coordinate_columns)
    source_feature_names = [
        column for column in raw_frame.columns if column not in excluded_columns
    ]
    train_idx, validation_idx, test_idx, split_summary = create_split_indices(
        raw_frame,
        raw_frame[args.target_column],
        args,
    )

    train_frame = raw_frame.iloc[train_idx].reset_index(drop=True)
    training_features = engineer_feature_frame(
        train_frame,
        source_feature_names,
        args.cyclical_angle_columns,
    )
    categorical_feature_names = [
        column for column in args.categorical_columns if column in training_features.columns
    ]
    indicator_feature_names = [
        column for column in training_features.columns if column.endswith("_missing")
    ]
    continuous_feature_names = [
        column
        for column in training_features.columns
        if column not in categorical_feature_names and column not in indicator_feature_names
    ]
    feature_transformers: list[tuple[str, Any, list[str]]] = []
    if categorical_feature_names:
        feature_transformers.append(
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
        feature_transformers.append(
            ("indicator", "passthrough", indicator_feature_names)
        )
    if continuous_feature_names:
        feature_transformers.append(
            ("num", SimpleImputer(strategy="median"), continuous_feature_names)
        )
    feature_transformer = ColumnTransformer(
        transformers=feature_transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )
    scaler = StandardScaler()
    training_transformed = feature_transformer.fit_transform(training_features).astype(np.float32)
    transformed_feature_names = get_transformed_feature_names(
        feature_transformer,
        categorical_feature_names,
        indicator_feature_names,
        continuous_feature_names,
    )
    if continuous_feature_names:
        start_index = len(transformed_feature_names) - len(continuous_feature_names)
        scaler.fit(training_transformed[:, start_index:])
    else:
        start_index = len(transformed_feature_names)
        scaler.fit(np.zeros((len(training_transformed), 1), dtype=np.float32))

    train_bundle = build_split_bundle(
        raw_frame,
        train_idx,
        source_feature_names,
        args.target_column,
        args.coordinate_columns,
        args.cyclical_angle_columns,
        feature_transformer,
        scaler,
        categorical_feature_names,
        indicator_feature_names,
        continuous_feature_names,
        transformed_feature_names,
    )
    validation_bundle = build_split_bundle(
        raw_frame,
        validation_idx,
        source_feature_names,
        args.target_column,
        args.coordinate_columns,
        args.cyclical_angle_columns,
        feature_transformer,
        scaler,
        categorical_feature_names,
        indicator_feature_names,
        continuous_feature_names,
        transformed_feature_names,
    )
    test_bundle = build_split_bundle(
        raw_frame,
        test_idx,
        source_feature_names,
        args.target_column,
        args.coordinate_columns,
        args.cyclical_angle_columns,
        feature_transformer,
        scaler,
        categorical_feature_names,
        indicator_feature_names,
        continuous_feature_names,
        transformed_feature_names,
    )

    split_summary.update(
        {
            "dropped_rows_with_missing_target": int(original_row_count - len(raw_frame)),
            "total_rows_after_target_drop": int(len(raw_frame)),
            "source_feature_count": int(len(source_feature_names)),
            "model_feature_count": int(len(transformed_feature_names)),
            "features_used": source_feature_names,
            "model_features_used": transformed_feature_names,
            "categorical_feature_names": categorical_feature_names,
            "indicator_feature_names": indicator_feature_names,
            "continuous_feature_names": continuous_feature_names,
            "cyclical_angle_columns": [
                column
                for column in args.cyclical_angle_columns
                if column in source_feature_names
            ],
            "coordinates_used_as_features": bool(args.include_coordinate_features),
            "train_rows": int(len(train_bundle.labels)),
            "validation_rows": int(len(validation_bundle.labels)),
            "test_rows": int(len(test_bundle.labels)),
            "train_positive_rate": float(train_bundle.labels.mean()),
            "validation_positive_rate": float(validation_bundle.labels.mean()),
            "test_positive_rate": float(test_bundle.labels.mean()),
        }
    )
    return PreparedData(
        feature_names=transformed_feature_names,
        source_feature_names=source_feature_names,
        categorical_feature_names=categorical_feature_names,
        indicator_feature_names=indicator_feature_names,
        continuous_feature_names=continuous_feature_names,
        continuous_feature_start_index=start_index,
        coordinate_columns=args.coordinate_columns,
        split_strategy=args.split_strategy,
        split_summary=split_summary,
        train=train_bundle,
        validation=validation_bundle,
        test=test_bundle,
        feature_transformer=feature_transformer,
        scaler=scaler,
    )


def save_processed_data(prepared: PreparedData, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        "train": prepared.train,
        "validation": prepared.validation,
        "test": prepared.test,
    }
    for split_name, bundle in split_map.items():
        pd.DataFrame(bundle.features_scaled, columns=prepared.feature_names).to_csv(
            processed_dir / f"x_{split_name}.csv",
            index=False,
        )
        pd.DataFrame({"Class": bundle.labels}).to_csv(
            processed_dir / f"y_{split_name}.csv",
            index=False,
        )
        bundle.features_display.to_csv(
            processed_dir / f"x_{split_name}_imputed_unscaled.csv",
            index=False,
        )
        bundle.metadata.to_csv(
            processed_dir / f"metadata_{split_name}.csv",
            index=False,
        )

    with (processed_dir / "feature_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_names": prepared.feature_names,
                "source_feature_names": prepared.source_feature_names,
                "categorical_feature_names": prepared.categorical_feature_names,
                "indicator_feature_names": prepared.indicator_feature_names,
                "continuous_feature_names": prepared.continuous_feature_names,
                "continuous_feature_start_index": prepared.continuous_feature_start_index,
                "coordinate_columns": prepared.coordinate_columns,
                "split_summary": prepared.split_summary,
            },
            handle,
            indent=2,
        )

    with (processed_dir / "preprocessing.pkl").open("wb") as handle:
        pickle.dump(
            {
                "feature_transformer": prepared.feature_transformer,
                "scaler": prepared.scaler,
                "feature_names": prepared.feature_names,
                "source_feature_names": prepared.source_feature_names,
                "categorical_feature_names": prepared.categorical_feature_names,
                "indicator_feature_names": prepared.indicator_feature_names,
                "continuous_feature_names": prepared.continuous_feature_names,
                "continuous_feature_start_index": prepared.continuous_feature_start_index,
                "coordinate_columns": prepared.coordinate_columns,
                "split_summary": prepared.split_summary,
            },
            handle,
        )


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    threshold_grid = np.linspace(0.05, 0.95, 181)
    rows: list[dict[str, float]] = []
    for threshold in threshold_grid:
        y_pred = (y_prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": safe_score(precision_score, y_true, y_pred, zero_division=0),
                "recall": safe_score(recall_score, y_true, y_pred, zero_division=0),
                "f1": safe_score(f1_score, y_true, y_pred, zero_division=0),
                "balanced_accuracy": safe_score(balanced_accuracy_score, y_true, y_pred),
            }
        )
    threshold_frame = pd.DataFrame(rows)
    best_row = threshold_frame.sort_values(
        by=["f1", "balanced_accuracy", "precision", "recall"],
        ascending=False,
    ).iloc[0]
    return float(best_row["threshold"]), threshold_frame


def predict_probabilities(
    model: FloodDNN, features: np.ndarray, batch_size: int
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            batch = torch.tensor(features[start:end], dtype=torch.float32, device=device)
            batch_probabilities = torch.sigmoid(model(batch)).cpu().numpy()
            probabilities.append(batch_probabilities)
    return np.concatenate(probabilities)


def extract_training_history(csv_logger: CSVLogger) -> pd.DataFrame:
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    metrics_frame = pd.read_csv(metrics_path)
    metrics_frame = metrics_frame.sort_values(["epoch", "step"]).reset_index(drop=True)
    history = metrics_frame.groupby("epoch", as_index=False).last()
    return history


def plot_training_history(history: pd.DataFrame, figure_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    train_loss_column = "train_loss"
    if "train_loss_epoch" in history:
        train_loss_column = "train_loss_epoch"
    if train_loss_column in history:
        axes[0].plot(history["epoch"], history[train_loss_column], label="Train loss", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss", linewidth=2)
    axes[0].set_title("Loss History")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].legend()

    if "val_auc" in history:
        axes[1].plot(history["epoch"], history["val_auc"], label="Validation ROC-AUC", linewidth=2)
    if "val_ap" in history:
        axes[1].plot(
            history["epoch"],
            history["val_ap"],
            label="Validation average precision",
            linewidth=2,
        )
    axes[1].set_title("Validation Discrimination")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, figure_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig, axis = plt.subplots(figsize=(7, 6), dpi=300)
    axis.plot(fpr, tpr, linewidth=2.5, label=f"ROC-AUC = {auc_score:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_title("ROC Curve")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray, figure_path: Path
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))
    fig, axis = plt.subplots(figsize=(7, 6), dpi=300)
    axis.plot(recall, precision, linewidth=2.5, label=f"AP = {average_precision:.3f}")
    axis.axhline(baseline, linestyle="--", color="black", linewidth=1, label="Class prevalence")
    axis.set_title("Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, figure_path: Path
) -> None:
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-flood", "Flood"],
        yticklabels=["Non-flood", "Flood"],
        ax=axis,
    )
    axis.set_title(f"Confusion Matrix at threshold = {threshold:.2f}")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Observed class")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, figure_path: Path
) -> None:
    frac_positive, mean_predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )
    fig, axis = plt.subplots(figsize=(7, 6), dpi=300)
    axis.plot(mean_predicted, frac_positive, marker="o", linewidth=2, label="DNN")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    axis.set_title("Calibration Curve")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed flood frequency")
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distribution(
    y_true: np.ndarray, y_prob: np.ndarray, figure_path: Path
) -> None:
    probability_frame = pd.DataFrame(
        {
            "Predicted probability": y_prob,
            "Observed class": np.where(y_true == 1, "Flood", "Non-flood"),
        }
    )
    fig, axis = plt.subplots(figsize=(8, 6), dpi=300)
    sns.histplot(
        data=probability_frame,
        x="Predicted probability",
        hue="Observed class",
        stat="density",
        common_norm=False,
        bins=30,
        element="step",
        fill=True,
        alpha=0.35,
        ax=axis,
    )
    axis.set_title("Predicted Susceptibility Distribution")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(threshold_frame: pd.DataFrame, figure_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 6), dpi=300)
    for metric_name in ["precision", "recall", "f1", "balanced_accuracy"]:
        axis.plot(
            threshold_frame["threshold"],
            threshold_frame[metric_name],
            linewidth=2,
            label=metric_name.replace("_", " ").title(),
        )
    axis.set_title("Validation Threshold Sweep")
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlation_heatmap(
    features: pd.DataFrame, figure_path: Path
) -> None:
    correlation = features.corr(numeric_only=True)
    fig, axis = plt.subplots(figsize=(12, 10), dpi=300)
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0.0,
        square=True,
        linewidths=0.3,
        ax=axis,
    )
    axis.set_title("Training Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_susceptibility_map(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    figure_path: Path,
    coordinate_columns: list[str],
) -> None:
    if len(coordinate_columns) < 2:
        return
    figure_data = metadata.copy()
    figure_data["predicted_probability"] = probabilities
    figure_data["predicted_class"] = (probabilities >= threshold).astype(int)
    fig, axis = plt.subplots(figsize=(8, 7), dpi=300)
    scatter = axis.scatter(
        figure_data[coordinate_columns[0]],
        figure_data[coordinate_columns[1]],
        c=figure_data["predicted_probability"],
        cmap="viridis",
        s=16,
        alpha=0.85,
        linewidths=0,
    )
    axis.set_title("Test-set Flood Susceptibility Map")
    axis.set_xlabel(coordinate_columns[0])
    axis.set_ylabel(coordinate_columns[1])
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Predicted flood probability")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def select_waterfall_index(y_true: np.ndarray, y_prob: np.ndarray) -> int:
    positive_indices = np.where(y_true == 1)[0]
    if len(positive_indices) > 0:
        return int(positive_indices[np.argmax(y_prob[positive_indices])])
    return int(np.argmax(y_prob))


def compute_shap_outputs(
    model: FloodDNN,
    prepared: PreparedData,
    output_dir: Path,
    batch_size: int,
    background_samples: int,
    explain_samples: int,
    seed: int,
) -> None:
    del batch_size
    rng = np.random.default_rng(seed)
    wrapper = ProbabilityWrapper(model.cpu().eval())

    background_size = min(background_samples, len(prepared.train.features_scaled))
    explain_size = min(explain_samples, len(prepared.test.features_scaled))
    background_index = rng.choice(
        len(prepared.train.features_scaled),
        size=background_size,
        replace=False,
    )
    explain_index = rng.choice(
        len(prepared.test.features_scaled),
        size=explain_size,
        replace=False,
    )

    background_tensor = torch.tensor(
        prepared.train.features_scaled[background_index],
        dtype=torch.float32,
    )
    explain_tensor = torch.tensor(
        prepared.test.features_scaled[explain_index],
        dtype=torch.float32,
    )
    display_frame = prepared.test.features_display.iloc[explain_index].reset_index(drop=True)

    try:
        explainer = shap.DeepExplainer(wrapper, background_tensor)
        shap_values = explainer.shap_values(explain_tensor)
        expected_value = getattr(explainer, "expected_value", None)
    except Exception:
        explainer = shap.GradientExplainer(wrapper, background_tensor)
        shap_values = explainer.shap_values(explain_tensor)
        expected_value = None

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    if expected_value is None:
        with torch.no_grad():
            expected_value = float(wrapper(background_tensor).mean().item())
    elif isinstance(expected_value, list):
        expected_value = expected_value[0]
    elif isinstance(expected_value, np.ndarray) and expected_value.ndim > 0:
        expected_value = expected_value.squeeze()

    shap.summary_plot(
        shap_values,
        features=display_frame,
        feature_names=prepared.feature_names,
        show=False,
        plot_size=(10, 6),
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()

    mean_absolute_shap = np.abs(shap_values).mean(axis=0)
    importance_frame = pd.DataFrame(
        {
            "feature": prepared.feature_names,
            "mean_abs_shap": mean_absolute_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_frame.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    fig, axis = plt.subplots(figsize=(8, 6), dpi=300)
    sns.barplot(
        data=importance_frame.head(15),
        x="mean_abs_shap",
        y="feature",
        color=sns.color_palette("crest", n_colors=1)[0],
        ax=axis,
    )
    axis.set_title("SHAP Feature Importance")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    local_index = select_waterfall_index(
        prepared.test.labels[explain_index],
        predict_probabilities(model.cpu(), prepared.test.features_scaled[explain_index], 512),
    )
    local_base_value = float(np.asarray(expected_value).reshape(-1)[0])
    waterfall = shap.Explanation(
        values=shap_values[local_index],
        base_values=local_base_value,
        data=display_frame.iloc[local_index].to_numpy(),
        feature_names=prepared.feature_names,
    )
    shap.plots.waterfall(waterfall, max_display=12, show=False)
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall.png", bbox_inches="tight", dpi=300)
    plt.close()


def save_predictions(
    metadata: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    file_path: Path,
) -> None:
    prediction_frame = metadata.copy()
    prediction_frame["predicted_probability"] = y_prob
    prediction_frame["predicted_class"] = (y_prob >= threshold).astype(int)
    prediction_frame.to_csv(file_path, index=False)


def train_model(
    args: argparse.Namespace,
    prepared: PreparedData,
    output_dir: Path,
) -> tuple[FloodDNN, CSVLogger]:
    positive_count = int(prepared.train.labels.sum())
    negative_count = int(len(prepared.train.labels) - positive_count)
    pos_weight = negative_count / max(positive_count, 1)

    data_module = FloodDataModule(
        prepared=prepared,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gaussian_noise_std=args.gaussian_noise_std,
    )
    model = FloodDNN(
        input_dim=len(prepared.feature_names),
        hidden_dims=args.hidden_dims,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-flood-dnn",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=args.patience,
    )
    logger = CSVLogger(save_dir=str(output_dir / "logs"), name="flood_dnn")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    trainer.fit(model, datamodule=data_module)

    best_model = FloodDNN.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        map_location="cpu",
    )
    best_model.eval()
    return best_model, logger


def generate_all_outputs(
    args: argparse.Namespace,
    prepared: PreparedData,
    model: FloodDNN,
    logger: CSVLogger,
) -> dict[str, Any]:
    output_dir = args.output_dir
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    validation_probabilities = predict_probabilities(
        model,
        prepared.validation.features_scaled,
        args.batch_size,
    )
    test_probabilities = predict_probabilities(
        model,
        prepared.test.features_scaled,
        args.batch_size,
    )

    threshold, threshold_frame = choose_threshold(
        prepared.validation.labels,
        validation_probabilities,
    )
    threshold_frame.to_csv(reports_dir / "validation_threshold_sweep.csv", index=False)
    plot_threshold_sweep(threshold_frame, figures_dir / "threshold_sweep.png")

    validation_metrics = compute_binary_metrics(
        prepared.validation.labels,
        validation_probabilities,
        threshold=threshold,
    )
    test_metrics = compute_binary_metrics(
        prepared.test.labels,
        test_probabilities,
        threshold=threshold,
    )

    history = extract_training_history(logger)
    history.to_csv(reports_dir / "training_history.csv", index=False)
    plot_training_history(history, figures_dir / "training_history.png")
    plot_roc_curve(prepared.test.labels, test_probabilities, figures_dir / "roc_curve.png")
    plot_precision_recall_curve(
        prepared.test.labels,
        test_probabilities,
        figures_dir / "precision_recall_curve.png",
    )
    plot_confusion_matrix(
        prepared.test.labels,
        test_probabilities,
        threshold,
        figures_dir / "confusion_matrix.png",
    )
    plot_calibration_curve(
        prepared.test.labels,
        test_probabilities,
        figures_dir / "calibration_curve.png",
    )
    plot_probability_distribution(
        prepared.test.labels,
        test_probabilities,
        figures_dir / "probability_distribution.png",
    )
    plot_feature_correlation_heatmap(
        prepared.train.features_display,
        figures_dir / "feature_correlation_heatmap.png",
    )
    plot_susceptibility_map(
        prepared.test.metadata,
        test_probabilities,
        threshold,
        figures_dir / "susceptibility_map.png",
        prepared.coordinate_columns,
    )
    compute_shap_outputs(
        model=model,
        prepared=prepared,
        output_dir=figures_dir,
        batch_size=args.batch_size,
        background_samples=args.background_samples,
        explain_samples=args.explain_samples,
        seed=args.seed,
    )

    save_predictions(
        prepared.validation.metadata,
        validation_probabilities,
        threshold,
        predictions_dir / "validation_predictions.csv",
    )
    save_predictions(
        prepared.test.metadata,
        test_probabilities,
        threshold,
        predictions_dir / "test_predictions.csv",
    )

    report = {
        "split_summary": prepared.split_summary,
        "gaussian_noise_std": args.gaussian_noise_std,
        "selected_threshold_from_validation": threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "output_directories": {
            "figures": str(figures_dir),
            "reports": str(reports_dir),
            "predictions": str(predictions_dir),
        },
    }
    with (reports_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    pd.DataFrame(
        [
            {"split": "validation", **validation_metrics},
            {"split": "test", **test_metrics},
        ]
    ).to_csv(reports_dir / "metrics_summary.csv", index=False)
    return report


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    parser = build_argument_parser()
    args = parser.parse_args()

    args.data_path = args.data_path.resolve()
    args.processed_dir = args.processed_dir.resolve()
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
    prepared = prepare_dataset(args)
    save_processed_data(prepared, args.processed_dir)
    model, logger = train_model(args, prepared, args.output_dir)
    report = generate_all_outputs(args, prepared, model, logger)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
