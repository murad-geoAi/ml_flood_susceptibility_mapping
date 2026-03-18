from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
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

from src.utils import safe_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    eps = 1e-7
    clipped = np.clip(y_prob, eps, 1.0 - eps)
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
        "log_loss": safe_score(log_loss, y_true, clipped),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


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
        ["f1", "balanced_accuracy", "precision", "recall"],
        ascending=False,
    ).iloc[0]
    return float(best_row["threshold"]), threshold_frame


def plot_class_distribution(y_true: np.ndarray, figure_path: Path) -> None:
    class_frame = pd.DataFrame(
        {"Class": np.where(np.asarray(y_true) == 1, "Flood-prone", "Non-flood-prone")}
    )
    fig, axis = plt.subplots(figsize=(7, 5), dpi=300)
    sns.countplot(data=class_frame, x="Class", hue="Class", legend=False, palette="crest", ax=axis)
    axis.set_title("Class Distribution")
    axis.set_xlabel("")
    axis.set_ylabel("Sample count")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_missingness(missingness_frame: pd.DataFrame, figure_path: Path) -> None:
    figure_frame = missingness_frame.loc[missingness_frame["missing_count"] > 0].copy()
    fig, axis = plt.subplots(figsize=(10, 6), dpi=300)
    sns.barplot(
        data=figure_frame,
        x="missing_pct",
        y="feature",
        hue="feature",
        legend=False,
        palette="flare",
        ax=axis,
    )
    axis.set_title("Missingness Profile")
    axis.set_xlabel("Missing values (%)")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_lulc_flood_rate(lulc_frame: pd.DataFrame, figure_path: Path) -> None:
    figure_frame = lulc_frame.copy()
    figure_frame["LULC"] = figure_frame["LULC"].fillna("Missing").astype(str)
    fig, axis = plt.subplots(figsize=(9, 5), dpi=300)
    sns.barplot(
        data=figure_frame,
        x="LULC",
        y="flood_rate",
        hue="LULC",
        legend=False,
        palette="crest",
        ax=axis,
    )
    axis.set_title("Flood Rate by LULC Class")
    axis.set_xlabel("LULC category")
    axis.set_ylabel("Flood rate")
    axis.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_split_map(
    metadata_frame: pd.DataFrame,
    coordinate_columns: list[str],
    figure_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(8, 7), dpi=300)
    sns.scatterplot(
        data=metadata_frame,
        x=coordinate_columns[0],
        y=coordinate_columns[1],
        hue="split",
        palette={"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"},
        s=18,
        linewidth=0,
        alpha=0.8,
        ax=axis,
    )
    axis.set_title("Spatial Train / Validation / Test Split")
    axis.set_xlabel(coordinate_columns[0])
    axis.set_ylabel(coordinate_columns[1])
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(features: pd.DataFrame, figure_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 10), dpi=300)
    correlation = features.corr(numeric_only=True)
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0.0,
        square=True,
        linewidths=0.3,
        ax=axis,
    )
    axis.set_title("Processed Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics_frame: pd.DataFrame, figure_path: Path) -> None:
    figure_frame = metrics_frame.sort_values(["roc_auc", "average_precision"], ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    sns.barplot(
        data=figure_frame,
        x="roc_auc",
        y="model",
        hue="model",
        legend=False,
        palette="crest",
        ax=axes[0],
    )
    axes[0].set_title("Test ROC-AUC by Model")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_ylabel("")

    sns.barplot(
        data=figure_frame,
        x="average_precision",
        y="model",
        hue="model",
        legend=False,
        palette="flare",
        ax=axes[1],
    )
    axes[1].set_title("Test Average Precision by Model")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Average precision")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(threshold_frame: pd.DataFrame, figure_path: Path, title: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 6), dpi=300)
    for metric_name in ["precision", "recall", "f1", "balanced_accuracy"]:
        axis.plot(
            threshold_frame["threshold"],
            threshold_frame[metric_name],
            linewidth=2,
            label=metric_name.replace("_", " ").title(),
        )
    axis.set_title(title)
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.05)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, figure_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    fig, axis = plt.subplots(figsize=(7, 6), dpi=300)
    axis.plot(fpr, tpr, linewidth=2.5, label=f"ROC-AUC = {auc_score:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axis.set_title(title)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figure_path: Path,
    title: str,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    average_precision = average_precision_score(y_true, y_prob)
    baseline = float(np.mean(y_true))
    fig, axis = plt.subplots(figsize=(7, 6), dpi=300)
    axis.plot(recall, precision, linewidth=2.5, label=f"AP = {average_precision:.3f}")
    axis.axhline(baseline, linestyle="--", color="black", linewidth=1, label="Class prevalence")
    axis.set_title(title)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    figure_path: Path,
    title: str,
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
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Observed class")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figure_path: Path,
    title: str,
) -> None:
    frac_positive, mean_predicted = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )
    fig, axis = plt.subplots(figsize=(7, 6), dpi=300)
    axis.plot(mean_predicted, frac_positive, marker="o", linewidth=2, label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration")
    axis.set_title(title)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed flood frequency")
    axis.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figure_path: Path,
    title: str,
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
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_susceptibility_map(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    coordinate_columns: list[str],
    figure_path: Path,
    title: str,
) -> None:
    figure_data = metadata.copy()
    figure_data["predicted_probability"] = probabilities
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
    axis.set_title(title)
    axis.set_xlabel(coordinate_columns[0])
    axis.set_ylabel(coordinate_columns[1])
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("Predicted flood probability")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
