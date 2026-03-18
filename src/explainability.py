from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap


def save_feature_importance(
    model: object,
    feature_names: list[str],
    csv_path: Path,
    figure_path: Path,
    title: str,
    top_n: int = 15,
) -> pd.DataFrame | None:
    if not hasattr(model, "feature_importances_"):
        return None
    importance = np.asarray(model.feature_importances_, dtype=float)
    if importance.shape[0] != len(feature_names):
        return None
    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_frame.to_csv(csv_path, index=False)

    plot_frame = importance_frame.head(top_n).iloc[::-1]
    fig, axis = plt.subplots(figsize=(9, 6), dpi=300)
    sns.barplot(
        data=plot_frame,
        x="importance",
        y="feature",
        hue="feature",
        legend=False,
        palette="crest",
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Importance")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    return importance_frame


def compute_tree_shap_outputs(
    model: object,
    feature_frame: pd.DataFrame,
    output_dir: Path,
    model_slug: str,
    title_prefix: str,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_frame = feature_frame.sample(
        n=min(max_samples, len(feature_frame)),
        random_state=seed,
    ).reset_index(drop=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_frame)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]
        elif shap_values.shape[-1] == 1:
            shap_values = shap_values[:, :, 0]

    shap.summary_plot(
        shap_values,
        sample_frame,
        show=False,
        plot_size=(10, 6),
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_slug}_shap_summary.png", bbox_inches="tight", dpi=300)
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_frame = (
        pd.DataFrame(
            {"feature": sample_frame.columns.tolist(), "mean_abs_shap": mean_abs_shap}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_frame.to_csv(output_dir / f"{model_slug}_shap_importance.csv", index=False)

    plot_frame = importance_frame.head(15).iloc[::-1]
    fig, axis = plt.subplots(figsize=(9, 6), dpi=300)
    sns.barplot(
        data=plot_frame,
        x="mean_abs_shap",
        y="feature",
        hue="feature",
        legend=False,
        palette="flare",
        ax=axis,
    )
    axis.set_title(f"{title_prefix} SHAP Importance")
    axis.set_xlabel("Mean absolute SHAP value")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_slug}_shap_importance.png", bbox_inches="tight")
    plt.close(fig)
    return importance_frame
