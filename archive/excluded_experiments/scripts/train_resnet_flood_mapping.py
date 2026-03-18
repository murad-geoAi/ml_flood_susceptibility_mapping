from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch import nn

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from ml_flood_susceptibility_mapping.scripts.train_dnn_flood_mapping import (
    DEFAULT_DATA_PATH,
    DEFAULT_PROCESSED_DIR,
    FloodDataModule,
    PROJECT_ROOT,
    PreparedData,
    compute_binary_metrics,
    generate_all_outputs,
    prepare_dataset,
    save_processed_data,
    seed_everything,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "resnet_flood_mapping"


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion_factor: int, dropout: float) -> None:
        super().__init__()
        expanded_dim = hidden_dim * expansion_factor
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class FloodResNet(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int,
        expansion_factor: int,
        learning_rate: float,
        dropout: float,
        weight_decay: float,
        pos_weight: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    hidden_dim=hidden_dim,
                    expansion_factor=expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
        )
        self.validation_outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.input_projection(inputs)
        for block in self.residual_blocks:
            outputs = block(outputs)
        return self.head(outputs).squeeze(1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        del batch_idx
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
        del batch_idx
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a residual tabular neural network for flood susceptibility "
            "mapping and compare it with the existing repo models."
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--expansion-factor", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gaussian-noise-std", type=float, default=0.05)
    parser.add_argument("--background-samples", type=int, default=256)
    parser.add_argument("--explain-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def train_model(
    args: argparse.Namespace,
    prepared: PreparedData,
    output_dir: Path,
) -> tuple[FloodResNet, CSVLogger]:
    positive_count = int(prepared.train.labels.sum())
    negative_count = int(len(prepared.train.labels) - positive_count)
    pos_weight = negative_count / max(positive_count, 1)

    data_module = FloodDataModule(
        prepared=prepared,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gaussian_noise_std=args.gaussian_noise_std,
    )
    model = FloodResNet(
        input_dim=len(prepared.feature_names),
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        expansion_factor=args.expansion_factor,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-flood-resnet",
        monitor="val_auc",
        mode="max",
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=args.patience,
    )
    logger = CSVLogger(save_dir=str(output_dir / "logs"), name="flood_resnet")
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

    best_model = FloodResNet.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        map_location="cpu",
    )
    best_model.eval()
    return best_model, logger


def load_existing_model_metrics() -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    dnn_metrics_path = PROJECT_ROOT / "artifacts" / "dnn_flood_mapping" / "reports" / "metrics_summary.csv"
    if dnn_metrics_path.exists():
        dnn_frame = pd.read_csv(dnn_metrics_path)
        dnn_frame = dnn_frame.loc[dnn_frame["split"] == "test"].copy()
        dnn_frame.insert(0, "model", "DNN")
        dnn_frame.insert(1, "family", "neural_net")
        dnn_frame["source"] = "dnn_flood_mapping"
        frames.append(dnn_frame)

    tree_metrics_path = (
        PROJECT_ROOT / "artifacts" / "tree_model_benchmark" / "reports" / "test_ranking.csv"
    )
    if tree_metrics_path.exists():
        tree_frame = pd.read_csv(tree_metrics_path).copy()
        tree_frame["source"] = "tree_model_benchmark"
        frames.append(tree_frame)

    return frames


def plot_repo_model_comparison(comparison_frame: pd.DataFrame, figure_path: Path) -> None:
    display_frame = comparison_frame.sort_values(
        ["roc_auc", "average_precision", "f1"],
        ascending=False,
    ).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    sns.barplot(
        data=display_frame,
        x="roc_auc",
        y="model",
        hue="model",
        palette="crest",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Repo Model Test ROC-AUC")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_ylabel("")

    sns.barplot(
        data=display_frame,
        x="average_precision",
        y="model",
        hue="model",
        palette="flare",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Repo Model Test Average Precision")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Average precision")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def build_repo_comparison(output_dir: Path) -> pd.DataFrame:
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"
    resnet_metrics_path = reports_dir / "metrics_summary.csv"
    if not resnet_metrics_path.exists():
        raise FileNotFoundError(f"Missing ResNet metrics file: {resnet_metrics_path}")

    resnet_frame = pd.read_csv(resnet_metrics_path)
    resnet_frame = resnet_frame.loc[resnet_frame["split"] == "test"].copy()
    resnet_frame.insert(0, "model", "ResNet")
    resnet_frame.insert(1, "family", "residual_nn")
    resnet_frame["source"] = "resnet_flood_mapping"

    comparison_frame = pd.concat(
        [resnet_frame, *load_existing_model_metrics()],
        ignore_index=True,
        sort=False,
    )
    comparison_frame = comparison_frame.sort_values(
        ["roc_auc", "average_precision", "f1"],
        ascending=False,
    ).reset_index(drop=True)
    comparison_frame.to_csv(reports_dir / "repo_model_comparison.csv", index=False)
    plot_repo_model_comparison(
        comparison_frame,
        figures_dir / "repo_model_comparison.png",
    )
    return comparison_frame


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
    comparison_frame = build_repo_comparison(args.output_dir)
    report["repo_model_comparison"] = {
        "path": str(args.output_dir / "reports" / "repo_model_comparison.csv"),
        "best_model": str(comparison_frame.iloc[0]["model"]),
        "best_model_roc_auc": float(comparison_frame.iloc[0]["roc_auc"]),
        "best_model_average_precision": float(comparison_frame.iloc[0]["average_precision"]),
    }

    with (args.output_dir / "reports" / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
