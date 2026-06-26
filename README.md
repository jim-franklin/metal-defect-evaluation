# metal-defect-eval

A data analysis pipeline built on the UCI Steel Plates Faults dataset (1,941 plates, 27 features, 7 fault types). The project mirrors the core workflow of in-line inspection data analysis: assess the data, prepare and integrate supplementary records, then categorize and evaluate defect features using machine learning.

## Background

In pipeline and industrial inspection, raw sensor data goes through a structured evaluation process before a result is reported. Features are detected, classified by type, and assessed for severity. This project applies that same workflow to open steel defect data as a demonstration of the underlying analytical process.

## What the notebooks cover

| Notebook | Stage |
|---|---|
| 01_exploration | Data profiling, class balance, correlation analysis, feature vs target relationships |
| 02_cleaning | Target encoding, supplementary data integration (steel type), feature reduction based on correlation findings, SMOTE balancing, train/val/test split |
| 03_classification | Random Forest and XGBoost classifiers, model comparison, confusion matrix, feature importance |

- XGBoost outperformed Random Forest (78.7% vs 74.9% validation accuracy), reaching 79% accuracy on the held-out test set with a weighted F1 of 0.80

- K_Scratch and Z_Scratch were the most separable classes (F1 0.95 and 0.96), consistent with their distinct size and geometric signatures identified in exploration

- Pastry was the hardest class (F1 0.59) due to heavy overlap with Other_Faults in feature space

- LogOfAreas was the strongest predictive feature, followed by steel type and Outside_X_Index, suggesting defect size and material grade drive classification more than position or luminosity

## Data

UCI Machine Learning Repository: Steel Plates Faults (dataset ID 198). Loaded directly via `ucimlrepo` for reproducibility.

## Future extensions

The current pipeline covers data assessment, preparation, and feature classification. Three analytical stages were scoped but not implemented in this version:

**Feature sizing**: a regression model to predict defect area from geometric features, demonstrating how classified defects would be dimensioned before reporting.

**Anomaly interaction rules**: proximity-based clustering (DBSCAN) to identify defects sitting close enough together to be assessed as a single combined feature, mirroring the interaction criteria applied in fitness-for-service assessment.

**Non-standard analysis**: novelty detection using Isolation Forest to flag defects that do not match any known fault class and would require manual review before a result is reported.

These stages follow naturally from the classification output and represent the next layer of the ILI evaluation workflow.