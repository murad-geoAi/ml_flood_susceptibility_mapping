# Flood Susceptibility ML

Leakage-aware classical machine learning for flood susceptibility mapping under a strict spatial holdout.

## Scope

This repository packages a conference-oriented flood susceptibility mapping study using only classical machine learning in the main workflow:

- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM

Neural-network experiments are excluded from the main study and treated as archived side experiments only.

## Scientific Focus

The repository is organized around five methodological principles:

1. Leakage-aware spatial evaluation
2. Reproducible preprocessing for mixed geospatial predictors
3. Classical ML comparison under the same spatial split
4. Explainability for selected models
5. Publication-ready outputs and documentation

## Dataset

Raw source:

- [data/raw/Flood_data.csv](/h:/Flood Susceptibility/ml_flood_susceptibility_mapping/data/raw/Flood_data.csv)

Key dataset facts:

- Original rows: `15,000`
- Rows with missing target removed: `12`
- Final labeled rows: `14,988`
- Positive rate: `27.42%`
- No duplicate rows
- No duplicate coordinate pairs

## Preprocessing

The final classical pipeline uses:

- removal of rows with missing target
- coordinate reservation for spatial splitting and export only
- one-hot encoding of `LULC`
- cyclic transformation of `Aspect` into `Aspect_sin` and `Aspect_cos`
- missing-indicator variables
- median imputation of continuous predictors
- reproducible feature assembly without coordinate leakage

## Spatial Evaluation

Main split design:

- strategy: `spatial`
- block size: `5000`
- spatial folds: `10`
- validation fold: `0`
- test fold: `1`

This strict spatial holdout is used to reduce leakage from nearby samples and better reflect real flood susceptibility mapping use cases.

## Current Main Findings

- `Random Forest` achieved the highest test ROC-AUC: `0.6270`
- `XGBoost` achieved the highest test average precision: `0.4024`
- `XGBoost` achieved the highest test F1: `0.4516`
- Tree/boosting models are the strongest family under spatial evaluation

Recommended primary model:

- `XGBoost`

Recommended benchmark baseline:

- `Random Forest`

## Repository Layout

```text
ml_flood_susceptibility_mapping/
|- README.md
|- requirements.txt
|- environment.yml
|- .gitignore
|- LICENSE
|- CITATION.cff
|- data/
|- notebooks/
|- src/
|- outputs/
|- docs/
|- scripts/
`- archive/
```

## Quick Start

Create the environment and run the full classical package:

```bash
python scripts/run_all.py
```

This produces:

- processed splits in `data/processed/`
- metrics and tables in `outputs/metrics/` and `outputs/tables/`
- publication figures in `outputs/figures/`
- explainability outputs in `outputs/shap/`
- susceptibility maps in `outputs/maps/`
- paper draft files in `docs/`

## Key Output Files

- [table_test_summary.csv](/h:/Flood Susceptibility/ml_flood_susceptibility_mapping/outputs/tables/table_test_summary.csv)
- [model_comparison.png](/h:/Flood Susceptibility/ml_flood_susceptibility_mapping/outputs/figures/model_comparison.png)
- [conference_paper.md](/h:/Flood Susceptibility/ml_flood_susceptibility_mapping/docs/conference_paper.md)
- [conference_paper.docx](/h:/Flood Susceptibility/ml_flood_susceptibility_mapping/docs/conference_paper.docx)

## Archived Material

Neural-network experiments are not part of the main paper package. See:

- [README.md](/h:/Flood Susceptibility/ml_flood_susceptibility_mapping/archive/excluded_experiments/README.md)
