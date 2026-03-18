# Methods

## Study Design

This study evaluates classical machine learning models for flood susceptibility mapping from tabular geospatial predictors under a leakage-aware spatial holdout. The analytical workflow was designed to prioritize reproducibility, defensible preprocessing, and realistic evaluation.

## Dataset

The raw dataset contains point-based observations with binary flood labels and environmental predictors:

- coordinates: `POINT_X`, `POINT_Y`
- land cover: `LULC`
- vegetation: `NDVI`
- terrain: `Elevation`, `Slope`, `Curvature`, `Aspect`
- hydrologic indicators: `TWI`, `Drainage_Density`, `SPI`
- climatic and proximity variables: `Precipitation`, `Distance_to_Road`, `Distance_to_River`

Rows with missing target values were removed prior to modeling.

## Preprocessing

The final preprocessing workflow included:

1. Removing rows with missing target values.
2. Reserving coordinates for spatial splitting and mapping only.
3. Treating `LULC` as a categorical predictor and one-hot encoding it.
4. Transforming `Aspect` into `Aspect_sin` and `Aspect_cos`.
5. Creating missing-indicator variables for source predictors.
6. Applying median imputation to continuous variables.
7. Preserving a fixed feature layout for all classical models without using coordinates as predictors.

This yielded a 31-feature modeling matrix composed of one-hot land-cover indicators, missingness flags, continuous predictors, and cyclic aspect features.

## Spatial Holdout

A strict spatial split was used to reduce information leakage:

- block size: `5000`
- folds: `10`
- validation fold: `0`
- test fold: `1`

This approach keeps nearby observations in the same fold and better reflects real deployment conditions for flood susceptibility mapping.

## Models

The main comparison included:

- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM

All models were trained on the same processed feature matrix and evaluated under the same spatial split.

## Evaluation

Because the dataset is moderately imbalanced, multiple metrics were reported:

- ROC-AUC
- Average Precision
- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- MCC

Decision thresholds were selected from the validation set and then applied unchanged to the test set.

## Explainability

Explainability outputs were generated for the primary candidate model (`XGBoost`) and the strongest benchmark (`Random Forest`) using built-in feature importance and TreeSHAP-based global explanations.
