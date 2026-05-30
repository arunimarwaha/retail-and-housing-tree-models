# Retail Sales and Housing Price Prediction with Tree-Based Models

## Project Overview

This project applies tree-based machine learning models to two business-style prediction problems:

1. Predicting whether retail stores have high or low child car seat sales
2. Predicting housing prices using regression trees and ensemble models

The project demonstrates how interpretable machine learning and ensemble methods can be used for business decision-making, model comparison, and performance evaluation.

## Business Questions

### Retail Sales Classification

Can store-level characteristics such as pricing, advertising, shelf location, and customer demographics be used to identify high-performing retail locations?

### Housing Price Prediction

Can ensemble tree-based models improve housing price prediction compared with a single regression tree?

## Datasets

The project uses two datasets from the `ISLR2` R package:

* `Carseats`: retail store sales data
* `Boston`: housing market data

No external data files are required.

## Methods Used

### Retail Sales Classification

* Created a binary sales outcome: high sales vs. low sales
* Built a classification tree to identify high-performing stores
* Used a train/test split to evaluate performance
* Evaluated results with accuracy and confusion matrices
* Applied cross-validation and pruning to simplify the model
* Compared full and pruned classification trees

### Housing Price Prediction

* Built a regression tree to predict median housing value
* Used cross-validation and pruning to improve interpretability
* Compared regression tree performance against ensemble methods
* Applied bagging, random forest, and boosting
* Evaluated models using test MSE and RMSE
* Used feature importance to identify key housing price drivers

## What the Project Does

The first part of the project focuses on retail sales classification. It uses store-level information to classify whether a store has high or low sales. A classification tree is used because it produces business-friendly decision rules that are easy to interpret.

The model is then evaluated on test data, and cross-validation is used to decide whether pruning improves the balance between accuracy and simplicity.

The second part of the project focuses on housing price prediction. It starts with a single regression tree and then compares it with more advanced ensemble models, including bagging, random forest, and boosting.

The housing models are evaluated using test RMSE, which makes it easier to compare prediction accuracy across models. Feature importance outputs are also created to show which variables are most important for predicting housing values.

## Outputs

Running the script automatically saves key results in the `outputs/` folder.

### Figures

* High vs. low sales distribution
* Full and pruned classification tree plots
* Classification tree cross-validation error plot
* Full and pruned regression tree plots
* Predicted vs. actual housing value plots
* Random forest feature importance
* Boosting feature importance
* Housing model RMSE comparison

### Tables

* Retail and housing dataset overview
* Sales classification distribution
* Classification tree performance
* Confusion matrices
* Cross-validation results
* Housing model comparison
* Random forest feature importance
* Boosting feature importance
* Business interpretation summary

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

## Tools and Packages Used

* R
* ISLR2
* tree
* randomForest
* gbm
* tidyverse
* ggplot2
* dplyr
* broom

## How to Run the Project

1. Clone or download this repository.

2. Open the project in RStudio or another R environment.

3. Run:

```r
source("scripts/tree_ensemble_models.R")
```

The script will load the datasets from `ISLR2`, run the analysis, and save all figures and tables in the `outputs/` folder.

## Key Insights

Classification trees provide an interpretable way to identify retail stores that are likely to have high sales.

Pruning can simplify a tree while keeping similar predictive performance, making the model easier to explain to business stakeholders.

For housing price prediction, a single regression tree is easy to interpret but may be less accurate than ensemble methods.

Bagging, random forests, and boosting improve prediction by combining many trees.

Feature importance helps translate complex models into business insights by showing which variables contribute most to predictions.

## Business Value

This project shows how tree-based models can support both classification and regression business problems.

For retail analytics, classification trees can help identify the store characteristics associated with high sales performance. This can support merchandising, pricing, and store strategy decisions.

For housing analytics, ensemble models can improve price prediction accuracy while feature importance helps explain key drivers of housing value.

The project demonstrates how model performance, interpretability, and business usefulness can be evaluated together.

## Skills Demonstrated

This project demonstrates skills relevant to data analyst, business analyst, and junior data science roles, including:

* Framing business problems as prediction tasks
* Preparing data for machine learning
* Building classification and regression trees
* Evaluating model performance on test data
* Using confusion matrices, MSE, and RMSE
* Applying cross-validation
* Pruning decision trees
* Comparing single-tree and ensemble models
* Using bagging, random forests, and boosting
* Interpreting feature importance
* Communicating model results in business terms
* Building reproducible analysis outputs in R

## Conclusion

This project demonstrates how tree-based machine learning models can be used for practical business prediction problems.

It shows the tradeoff between interpretability and predictive performance: single trees are easier to explain, while ensemble methods often provide stronger accuracy.

Overall, the project highlights how machine learning can support retail performance analysis, housing price prediction, and data-driven business decision-making.
