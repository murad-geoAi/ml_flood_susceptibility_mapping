# Figure Captions

1. `class_distribution.png`
Class distribution after removing rows with missing target labels.

2. `missingness_profile.png`
Percentage of missing values across source predictors in the labeled dataset.

3. `lulc_flood_rate.png`
Observed flood rate across land-use / land-cover classes, highlighting the strong effect of categorical land cover.

4. `spatial_split_map.png`
Spatial arrangement of training, validation, and test samples under the leakage-aware spatial holdout.

5. `processed_feature_correlation_heatmap.png`
Correlation structure of the processed feature matrix used for classical model training.

6. `model_comparison.png`
Test-set comparison of model ROC-AUC and average precision across all classical ML models.

7. `xgboost_roc_curve.png`
Receiver operating characteristic curve for the primary candidate model.

8. `xgboost_precision_recall_curve.png`
Precision-recall curve for the primary candidate model under class imbalance.

9. `xgboost_confusion_matrix.png`
Confusion matrix for XGBoost using the threshold selected from validation data.

10. `xgboost_calibration_curve.png`
Calibration performance of the primary candidate model on the spatial test set.

11. `xgboost_feature_importance.png`
Built-in feature importance ranking for XGBoost.

12. `xgboost_shap_summary.png`
Global TreeSHAP summary plot for XGBoost.

13. `random_forest_shap_summary.png`
Global TreeSHAP summary plot for the strongest benchmark model.

14. `xgboost_full_susceptibility_map.png`
Spatial flood susceptibility estimates produced by the primary candidate model across all labeled samples.
