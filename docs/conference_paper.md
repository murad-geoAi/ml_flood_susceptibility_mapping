# Leakage-Aware Classical Machine Learning for Flood Susceptibility Mapping Under a Strict Spatial Holdout

## Abstract

Flood susceptibility mapping often suffers from optimistic evaluation when nearby observations are allowed to leak between training and testing subsets. This study develops a classical machine learning workflow for flood susceptibility mapping using a tabular geospatial dataset and a strict spatial holdout. The dataset contains 14,988 labeled observations after removing rows with missing target values. A reproducible preprocessing pipeline was constructed to handle mixed geospatial predictors by one-hot encoding land use/land cover (`LULC`), applying cyclic encoding to `Aspect`, adding missing-indicator variables, median-imputing continuous predictors, and preserving a fixed feature matrix without coordinate leakage. Eight classical models were evaluated under the same spatial split: Decision Tree, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, and LightGBM. Random Forest achieved the highest test ROC-AUC (0.6270), while XGBoost achieved the best average precision (0.4024) and F1 score (0.4516). Explainability analyses consistently identified land-cover classes, vegetation, hydrologic, and terrain variables as dominant drivers of flood susceptibility. The results show that tree-based ensemble methods provide the strongest and most defendable performance under leakage-aware spatial evaluation.

## Keywords

Flood susceptibility mapping; spatial holdout; leakage-aware evaluation; XGBoost; Random Forest; explainable machine learning

## 1. Introduction

Flood susceptibility mapping is commonly used to identify areas that are likely to experience flooding based on environmental, hydrologic, and anthropogenic predictors. Many flood susceptibility studies report strong performance, but these results can be overly optimistic when spatial dependence is not properly controlled during evaluation. If nearby observations appear in both training and testing subsets, a model may learn local spatial patterns that do not generalize to new geographic areas.

This study addresses that issue by framing the problem as a leakage-aware geospatial susceptibility study. The main contribution is not simply testing multiple algorithms, but building a reproducible classical machine learning pipeline under a strict spatial holdout. The workflow emphasizes correct handling of mixed predictor types, interpretable tree-based models, publication-quality explainability outputs, and clean repository organization for reuse and review.

## 2. Materials and Methods

### 2.1 Dataset

The raw dataset contains 15 environmental and spatial columns, including binary flood class labels. After removing 12 rows with missing target values, the final labeled dataset contained 14,988 observations, of which 4,109 were flood-prone and 10,879 were non-flood-prone. The flood class prevalence was 27.42%.

### 2.2 Preprocessing

The preprocessing workflow was designed to preserve methodological defensibility:

1. Rows with missing target values were removed.
2. `POINT_X` and `POINT_Y` were reserved for spatial splitting and export only.
3. `LULC` was treated as a categorical predictor and one-hot encoded.
4. `Aspect` was transformed into `Aspect_sin` and `Aspect_cos`.
5. Missing-indicator variables were created for source predictors.
6. Continuous predictors were median-imputed.

This produced a 31-feature modeling matrix.

### 2.3 Spatial Holdout

A strict spatial split was used in the main study to limit leakage:

- block size: 5,000
- spatial folds: 10
- validation fold: 0
- test fold: 1

The resulting split contained 202 train groups, 27 validation groups, and 32 test groups.

### 2.4 Classical Machine Learning Models

The following classical models were evaluated:

- Decision Tree
- Random Forest
- Extra Trees
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM

All models were trained on the same processed features and evaluated using the same spatial split.

### 2.5 Evaluation Metrics

Because the dataset is moderately imbalanced, evaluation included ROC-AUC, average precision, accuracy, balanced accuracy, precision, recall, F1 score, and Matthews correlation coefficient. Decision thresholds were selected on the validation set and then applied to the test set.

### 2.6 Explainability

Model interpretation focused on the primary candidate (`XGBoost`) and the strongest benchmark (`Random Forest`). Global explainability was assessed using built-in feature importance and TreeSHAP outputs.

## 3. Results

### 3.1 Validation Results

The strongest validation performance was observed for XGBoost, LightGBM, and Random Forest, with XGBoost producing the highest validation ROC-AUC (0.6692) and average precision (0.4679).

### 3.2 Test Results

The main test-set results are summarized below.

| Model | ROC-AUC | Average Precision | F1 |
|---|---:|---:|---:|
| Random Forest | 0.6270 | 0.3879 | 0.4470 |
| XGBoost | 0.6256 | 0.4024 | 0.4516 |
| LightGBM | 0.6254 | 0.3971 | 0.3854 |
| Extra Trees | 0.6227 | 0.3738 | 0.4387 |
| HistGradientBoosting | 0.6208 | 0.3686 | 0.3639 |
| Gradient Boosting | 0.6207 | 0.3707 | 0.4275 |
| AdaBoost | 0.6193 | 0.3749 | 0.4313 |
| Decision Tree | 0.5689 | 0.3410 | 0.4163 |

Random Forest achieved the highest overall discrimination according to ROC-AUC, while XGBoost achieved the strongest imbalance-sensitive performance through average precision and F1 score.

### 3.3 Explainability

Across feature importance and TreeSHAP analyses, the most influential predictors consistently included:

- `LULC=5`
- `LULC=2`
- `LULC=7`
- `NDVI`
- `Precipitation`
- `Drainage_Density`
- `Elevation`
- `TWI`
- `Distance_to_Road`

These findings suggest that land-cover conditions dominate the predictive signal, while vegetation, hydrologic, and terrain variables provide important supporting information.

## 4. Discussion

Three main findings emerge from this study. First, preprocessing quality was more important than increasing model complexity. Correct treatment of categorical land cover, cyclic aspect, and informative missingness materially improved the study design. Second, the strict spatial holdout produced a challenging but realistic evaluation setting, which explains why even the strongest models remained in the 0.62-0.63 ROC-AUC range. Third, tree-based ensembles and gradient boosting methods were the strongest model family under this geospatial evaluation design.

From a practical standpoint, XGBoost is the strongest candidate when the priority is retrieving flood-prone cases under class imbalance, whereas Random Forest is the strongest benchmark when overall ranking ability is the main priority. This pairing provides a strong basis for conference reporting and supervisor discussion.

## 5. Conclusion

This study presents a reproducible classical machine learning framework for flood susceptibility mapping under leakage-aware spatial evaluation. The results show that classical tree-based ensembles outperform simpler baselines and remain highly competitive for interpretable, defendable geospatial susceptibility modeling. XGBoost is recommended as the primary candidate model, while Random Forest is recommended as the strongest benchmark baseline.

## Acknowledged Limitations

The current study uses one main spatial validation/test configuration. Future work should evaluate repeated spatial cross-validation to assess fold-to-fold stability and strengthen claims about generalization.
