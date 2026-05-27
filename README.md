# Retail Sales and Housing Price Prediction with Tree-Based Models

## Project Overview

This project applies tree-based machine learning models to two business-style prediction problems:

1. **Retail sales classification**: predicting whether child car seat sales at a store are high or low.
2. **Housing price prediction**: predicting median housing values using regression trees and ensemble methods.

The project demonstrates interpretable machine learning, model pruning, cross-validation, bagging, random forests, boosting, feature importance, and predictive model evaluation.

## Business Questions

### Retail Sales Classification
Can store-level characteristics such as pricing, advertising, shelf location, and customer demographics be used to identify high-performing retail locations?

### Housing Price Prediction
Can tree-based models improve housing price prediction compared with a single regression tree?

## Datasets

The project uses datasets from the `ISLR2` R package:

- `Carseats`: sales data from 400 retail stores
- `Boston`: housing data used for regression modeling

No external data files are required.

## Methods Used

### Retail Sales Classification
- Binary outcome creation: High vs. Low sales
- Classification tree
- Train/test split
- Confusion matrix
- Cross-validation
- Tree pruning
- Accuracy comparison

### Housing Price Prediction
- Regression tree
- Cross-validation and pruning
- Bagging
- Random forest
- Boosting
- Feature importance
- Partial dependence plots
- Test MSE and RMSE comparison

## Repository Structure

```text
retail-and-housing-tree-models/
│
├── scripts/
│   └── tree_ensemble_models.R
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── README.md
└── .gitignore
```

## Outputs

Running the script saves:

### Tables
- Retail dataset overview
- Classification tree test performance
- Pruned tree performance
- Cross-validation results
- Housing model comparison
- Random forest feature importance
- Boosting feature importance

### Figures
- Classification tree plot
- Pruned classification tree plot
- Cross-validation error plot
- Regression tree plot
- Pruned regression tree plot
- Predicted vs actual plots
- Random forest feature importance
- Boosting partial dependence plots

## Tools Used

- R
- ISLR2
- tree
- randomForest
- gbm
- ggplot2
- dplyr

## How to Run

1. Download or clone this repository.
2. Open RStudio.
3. Set your working directory to the project folder.
4. Run:

```r
source("scripts/tree_ensemble_models.R")
```

The script automatically creates output folders and saves all figures and tables.

## Portfolio Relevance

This project demonstrates applied machine learning skills relevant to data analyst and junior data science roles:

- business-style prediction problems
- interpretable machine learning
- classification and regression modeling
- model evaluation
- cross-validation
- ensemble learning
- communication of model performance and business insights
