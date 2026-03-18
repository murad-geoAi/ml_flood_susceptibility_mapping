from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"
SHAP_DIR = OUTPUTS_DIR / "shap"
MAPS_DIR = OUTPUTS_DIR / "maps"
MODELS_DIR = OUTPUTS_DIR / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
ARCHIVE_DIR = PROJECT_ROOT / "archive" / "excluded_experiments"

LEGACY_RAW_DATA_PATH = PROJECT_ROOT / "Flood_data.csv"
RAW_DATA_PATH = RAW_DATA_DIR / "Flood_data.csv"
DEFAULT_RAW_DATA_INPUT_PATH = RAW_DATA_PATH if RAW_DATA_PATH.exists() else LEGACY_RAW_DATA_PATH
TARGET_COLUMN = "Class"
COORDINATE_COLUMNS = ["POINT_X", "POINT_Y"]
SOURCE_COLUMNS = [
    "POINT_X",
    "POINT_Y",
    "LULC",
    "NDVI",
    "Elevation",
    "Slope",
    "Curvature",
    "Aspect",
    "TWI",
    "Drainage_Density",
    "Precipitation",
    "Distance_to_Road",
    "Distance_to_River",
    "SPI",
    "Class",
]
CATEGORICAL_COLUMNS = ["LULC"]
CYCLICAL_ANGLE_COLUMNS = ["Aspect"]
SPATIAL_BLOCK_SIZE = 5000.0
SPATIAL_FOLDS = 10
VALIDATION_FOLD = 0
TEST_FOLD = 1
SEED = 42
PRIMARY_MODEL_NAME = "XGBoost"
BENCHMARK_MODEL_NAME = "Random Forest"
MODEL_REPORT_ORDER = [
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "Extra Trees",
    "HistGradientBoosting",
    "Gradient Boosting",
    "AdaBoost",
    "Decision Tree",
]
