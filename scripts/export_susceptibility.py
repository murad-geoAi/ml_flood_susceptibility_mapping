from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.export_outputs import export_susceptibility_with_coordinates
from src.utils import slugify


def main() -> None:
    processed_features_path = config.PROCESSED_DATA_DIR / "processed_features.csv"
    study_summary_path = config.METRICS_DIR / "study_summary.json"
    if not processed_features_path.exists():
        raise FileNotFoundError(
            f"Missing processed features export at {processed_features_path}. Run scripts/run_all.py first."
        )
    if not study_summary_path.exists():
        raise FileNotFoundError(
            f"Missing study summary at {study_summary_path}. Run scripts/run_all.py first."
        )

    with study_summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    model_name = summary["primary_model"]
    threshold = float(summary["primary_model_threshold"])
    model_path = config.MODELS_DIR / f"{slugify(model_name)}.joblib"
    model = joblib.load(model_path)

    processed_frame = pd.read_csv(processed_features_path)
    metadata = processed_frame[
        ["sample_id", *config.COORDINATE_COLUMNS, "split", "observed_class"]
    ].copy()
    feature_frame = processed_frame.drop(
        columns=["sample_id", *config.COORDINATE_COLUMNS, "split", "observed_class"]
    )
    export_susceptibility_with_coordinates(
        model=model,
        feature_frame=feature_frame,
        metadata=metadata,
        threshold=threshold,
        model_name=model_name,
        file_path=config.PROCESSED_DATA_DIR / "susceptibility_with_coordinates.csv",
    )


if __name__ == "__main__":
    main()
