# Retail Sales and Housing Price Prediction with Tree-Based Models
# Author: Arunima Marwaha
#
# Purpose:
# This script creates a portfolio-ready machine learning project using:
# 1. Classification trees for retail sales prediction
# 2. Regression trees, bagging, random forests, and boosting for housing price prediction
#
# Datasets:
# - Carseats from ISLR2
# - Boston from ISLR2
#
# Outputs:
# - figures saved in outputs/figures/
# - tables saved in outputs/tables/

# -----------------------------
# 1. Setup
# -----------------------------

required_packages <- c(
  "ISLR2",
  "tree",
  "randomForest",
  "gbm",
  "tidyverse",
  "broom"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ISLR2)
library(tree)
library(randomForest)
library(gbm)
library(tidyverse)
library(broom)

dir.create("outputs", showWarnings = FALSE)
dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables", recursive = TRUE, showWarnings = FALSE)

# -----------------------------
# 2. Retail Sales Classification: Carseats
# -----------------------------

data("Carseats", package = "ISLR2")

# Create binary sales outcome:
# High = Yes if Sales > 8, otherwise No
carseats <- Carseats %>%
  dplyr::mutate(
    High = factor(ifelse(Sales > 8, "Yes", "No"))
  )

carseats_overview <- data.frame(
  metric = c("Number of observations", "Number of variables", "Outcome variable"),
  value = c(nrow(carseats), ncol(carseats), "High sales indicator")
)

write.csv(carseats_overview,
          "outputs/tables/01_carseats_dataset_overview.csv",
          row.names = FALSE)

sales_distribution <- carseats %>%
  dplyr::count(High) %>%
  dplyr::mutate(share = n / sum(n))

write.csv(sales_distribution,
          "outputs/tables/02_high_sales_distribution.csv",
          row.names = FALSE)

p_sales_dist <- ggplot(sales_distribution, aes(x = High, y = n)) +
  geom_col(fill = "steelblue") +
  labs(
    title = "Distribution of High vs Low Store Sales",
    x = "High sales",
    y = "Number of stores"
  ) +
  theme_minimal()

ggsave("outputs/figures/01_high_sales_distribution.png",
       plot = p_sales_dist, width = 7, height = 5, dpi = 300)

# -----------------------------
# 3. Classification Tree
# -----------------------------

# Fit full classification tree using all predictors except Sales
tree_carseats_full <- tree::tree(High ~ . - Sales, data = carseats)

capture.output(
  summary(tree_carseats_full),
  file = "outputs/tables/03_full_classification_tree_summary.txt"
)

png("outputs/figures/02_full_classification_tree.png",
    width = 1200, height = 900, res = 150)
plot(tree_carseats_full)
text(tree_carseats_full, pretty = 0)
dev.off()

# Train/test split
set.seed(123)
train_index <- sample(1:nrow(carseats), 200)

carseats_train <- carseats[train_index, ]
carseats_test <- carseats[-train_index, ]
high_test <- carseats_test$High

tree_carseats <- tree::tree(High ~ . - Sales, data = carseats_train)

tree_predictions <- predict(tree_carseats,
                            newdata = carseats_test,
                            type = "class")

tree_confusion <- table(Predicted = tree_predictions, Actual = high_test)
tree_accuracy <- mean(tree_predictions == high_test)
tree_error <- 1 - tree_accuracy

tree_performance <- data.frame(
  model = "Full classification tree",
  accuracy = tree_accuracy,
  error_rate = tree_error
)

write.csv(tree_performance,
          "outputs/tables/04_classification_tree_test_performance.csv",
          row.names = FALSE)

write.csv(as.data.frame(tree_confusion),
          "outputs/tables/05_classification_tree_confusion_matrix.csv",
          row.names = FALSE)

# -----------------------------
# 4. Cross-Validation and Pruning
# -----------------------------

set.seed(123)
cv_carseats <- tree::cv.tree(tree_carseats, FUN = prune.misclass)

cv_carseats_table <- data.frame(
  tree_size = cv_carseats$size,
  cross_validation_error = cv_carseats$dev,
  complexity_parameter = cv_carseats$k
)

write.csv(cv_carseats_table,
          "outputs/tables/06_classification_tree_cv_results.csv",
          row.names = FALSE)

p_cv_size <- ggplot(cv_carseats_table,
                    aes(x = tree_size, y = cross_validation_error)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Classification Tree Cross-Validation Error",
    x = "Tree size / terminal nodes",
    y = "Cross-validation error"
  ) +
  theme_minimal()

ggsave("outputs/figures/03_classification_tree_cv_error.png",
       plot = p_cv_size, width = 7, height = 5, dpi = 300)

# Select a 9-node tree for interpretability
pruned_carseats <- tree::prune.misclass(tree_carseats, best = 9)

png("outputs/figures/04_pruned_classification_tree.png",
    width = 1200, height = 900, res = 150)
plot(pruned_carseats)
text(pruned_carseats, pretty = 0)
dev.off()

pruned_predictions <- predict(pruned_carseats,
                              newdata = carseats_test,
                              type = "class")

pruned_confusion <- table(Predicted = pruned_predictions, Actual = high_test)
pruned_accuracy <- mean(pruned_predictions == high_test)
pruned_error <- 1 - pruned_accuracy

pruned_performance <- data.frame(
  model = "Pruned classification tree",
  accuracy = pruned_accuracy,
  error_rate = pruned_error
)

write.csv(pruned_performance,
          "outputs/tables/07_pruned_classification_tree_performance.csv",
          row.names = FALSE)

write.csv(as.data.frame(pruned_confusion),
          "outputs/tables/08_pruned_classification_tree_confusion_matrix.csv",
          row.names = FALSE)

classification_model_comparison <- dplyr::bind_rows(
  tree_performance,
  pruned_performance
)

write.csv(classification_model_comparison,
          "outputs/tables/09_classification_model_comparison.csv",
          row.names = FALSE)

# -----------------------------
# 5. Housing Price Prediction: Boston
# -----------------------------

data("Boston", package = "ISLR2")

boston_overview <- data.frame(
  metric = c("Number of observations", "Number of variables", "Outcome variable"),
  value = c(nrow(Boston), ncol(Boston), "medv")
)

write.csv(boston_overview,
          "outputs/tables/10_boston_dataset_overview.csv",
          row.names = FALSE)

set.seed(123)
boston_train_index <- sample(1:nrow(Boston), nrow(Boston) / 2)

boston_train <- Boston[boston_train_index, ]
boston_test <- Boston[-boston_train_index, ]
boston_test_y <- boston_test$medv

# -----------------------------
# 6. Regression Tree
# -----------------------------

tree_boston <- tree::tree(medv ~ ., data = boston_train)

capture.output(
  summary(tree_boston),
  file = "outputs/tables/11_regression_tree_summary.txt"
)

png("outputs/figures/05_full_regression_tree.png",
    width = 1200, height = 900, res = 150)
plot(tree_boston)
text(tree_boston, pretty = 0)
dev.off()

set.seed(123)
cv_boston <- tree::cv.tree(tree_boston)

cv_boston_table <- data.frame(
  tree_size = cv_boston$size,
  cross_validation_error = cv_boston$dev,
  complexity_parameter = cv_boston$k
)

write.csv(cv_boston_table,
          "outputs/tables/12_regression_tree_cv_results.csv",
          row.names = FALSE)

p_boston_cv <- ggplot(cv_boston_table,
                      aes(x = tree_size, y = cross_validation_error)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Regression Tree Cross-Validation Error",
    x = "Tree size / terminal nodes",
    y = "Cross-validation error"
  ) +
  theme_minimal()

ggsave("outputs/figures/06_regression_tree_cv_error.png",
       plot = p_boston_cv, width = 7, height = 5, dpi = 300)

pruned_boston <- tree::prune.tree(tree_boston, best = 5)

png("outputs/figures/07_pruned_regression_tree.png",
    width = 1200, height = 900, res = 150)
plot(pruned_boston)
text(pruned_boston, pretty = 0)
dev.off()

yhat_tree <- predict(tree_boston, newdata = boston_test)
yhat_pruned_tree <- predict(pruned_boston, newdata = boston_test)

mse_tree <- mean((yhat_tree - boston_test_y)^2)
rmse_tree <- sqrt(mse_tree)

mse_pruned_tree <- mean((yhat_pruned_tree - boston_test_y)^2)
rmse_pruned_tree <- sqrt(mse_pruned_tree)

pred_tree_df <- data.frame(
  actual = boston_test_y,
  predicted = yhat_tree
)

p_tree_pred <- ggplot(pred_tree_df, aes(x = predicted, y = actual)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1) +
  labs(
    title = "Regression Tree: Predicted vs Actual Housing Values",
    x = "Predicted median value",
    y = "Actual median value"
  ) +
  theme_minimal()

ggsave("outputs/figures/08_regression_tree_predicted_vs_actual.png",
       plot = p_tree_pred, width = 7, height = 5, dpi = 300)

# -----------------------------
# 7. Bagging and Random Forest
# -----------------------------

set.seed(123)

bag_boston <- randomForest::randomForest(
  medv ~ .,
  data = boston_train,
  mtry = ncol(boston_train) - 1,
  importance = TRUE
)

yhat_bag <- predict(bag_boston, newdata = boston_test)
mse_bag <- mean((yhat_bag - boston_test_y)^2)
rmse_bag <- sqrt(mse_bag)

rf_boston <- randomForest::randomForest(
  medv ~ .,
  data = boston_train,
  mtry = 6,
  importance = TRUE
)

yhat_rf <- predict(rf_boston, newdata = boston_test)
mse_rf <- mean((yhat_rf - boston_test_y)^2)
rmse_rf <- sqrt(mse_rf)

pred_rf_df <- data.frame(
  actual = boston_test_y,
  predicted = yhat_rf
)

p_rf_pred <- ggplot(pred_rf_df, aes(x = predicted, y = actual)) +
  geom_point(alpha = 0.7) +
  geom_abline(intercept = 0, slope = 1) +
  labs(
    title = "Random Forest: Predicted vs Actual Housing Values",
    x = "Predicted median value",
    y = "Actual median value"
  ) +
  theme_minimal()

ggsave("outputs/figures/09_random_forest_predicted_vs_actual.png",
       plot = p_rf_pred, width = 7, height = 5, dpi = 300)

rf_importance <- as.data.frame(randomForest::importance(rf_boston))
rf_importance$variable <- rownames(rf_importance)

write.csv(rf_importance,
          "outputs/tables/13_random_forest_feature_importance.csv",
          row.names = FALSE)

rf_importance_plot_data <- rf_importance %>%
  dplyr::arrange(desc(`%IncMSE`)) %>%
  dplyr::slice(1:10)

p_rf_importance <- ggplot(rf_importance_plot_data,
                          aes(x = reorder(variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Random Forest Feature Importance",
    x = "Variable",
    y = "Increase in MSE"
  ) +
  theme_minimal()

ggsave("outputs/figures/10_random_forest_feature_importance.png",
       plot = p_rf_importance, width = 7, height = 5, dpi = 300)

# -----------------------------
# 8. Boosting
# -----------------------------

set.seed(123)

boost_boston <- gbm::gbm(
  medv ~ .,
  data = boston_train,
  distribution = "gaussian",
  n.trees = 5000,
  interaction.depth = 4,
  verbose = FALSE
)

yhat_boost <- predict(boost_boston,
                      newdata = boston_test,
                      n.trees = 5000)

mse_boost <- mean((yhat_boost - boston_test_y)^2)
rmse_boost <- sqrt(mse_boost)

boost_importance <- summary(boost_boston, plotit = FALSE)

write.csv(boost_importance,
          "outputs/tables/14_boosting_feature_importance.csv",
          row.names = FALSE)

p_boost_importance <- ggplot(boost_importance,
                             aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_col(fill = "coral") +
  coord_flip() +
  labs(
    title = "Boosting Feature Importance",
    x = "Variable",
    y = "Relative influence"
  ) +
  theme_minimal()

ggsave("outputs/figures/11_boosting_feature_importance.png",
       plot = p_boost_importance, width = 7, height = 5, dpi = 300)

png("outputs/figures/12_boosting_partial_dependence_rm.png",
    width = 900, height = 700, res = 150)
plot(boost_boston, i = "rm")
dev.off()

png("outputs/figures/13_boosting_partial_dependence_lstat.png",
    width = 900, height = 700, res = 150)
plot(boost_boston, i = "lstat")
dev.off()

set.seed(123)

boost_boston_shrinkage <- gbm::gbm(
  medv ~ .,
  data = boston_train,
  distribution = "gaussian",
  n.trees = 5000,
  interaction.depth = 4,
  shrinkage = 0.02,
  verbose = FALSE
)

yhat_boost_shrinkage <- predict(boost_boston_shrinkage,
                                newdata = boston_test,
                                n.trees = 5000)

mse_boost_shrinkage <- mean((yhat_boost_shrinkage - boston_test_y)^2)
rmse_boost_shrinkage <- sqrt(mse_boost_shrinkage)

# -----------------------------
# 9. Housing Model Comparison
# -----------------------------

housing_model_comparison <- data.frame(
  model = c(
    "Regression tree",
    "Pruned regression tree",
    "Bagging",
    "Random forest",
    "Boosting",
    "Boosting with shrinkage = 0.02"
  ),
  test_mse = c(
    mse_tree,
    mse_pruned_tree,
    mse_bag,
    mse_rf,
    mse_boost,
    mse_boost_shrinkage
  ),
  test_rmse = c(
    rmse_tree,
    rmse_pruned_tree,
    rmse_bag,
    rmse_rf,
    rmse_boost,
    rmse_boost_shrinkage
  )
)

write.csv(housing_model_comparison,
          "outputs/tables/15_housing_model_comparison.csv",
          row.names = FALSE)

p_model_comparison <- housing_model_comparison %>%
  ggplot(aes(x = reorder(model, test_rmse), y = test_rmse)) +
  geom_col(fill = "grey50") +
  coord_flip() +
  labs(
    title = "Housing Price Model Comparison",
    subtitle = "Lower RMSE indicates better predictive performance",
    x = "Model",
    y = "Test RMSE"
  ) +
  theme_minimal()

ggsave("outputs/figures/14_housing_model_comparison_rmse.png",
       plot = p_model_comparison, width = 8, height = 5, dpi = 300)

# -----------------------------
# 10. Business Interpretation
# -----------------------------

business_interpretation <- data.frame(
  section = c(
    "Retail classification",
    "Retail classification",
    "Housing regression",
    "Housing regression",
    "Housing regression"
  ),
  insight = c(
    "Classification trees provide an interpretable way to identify high-performing stores.",
    "Pruning can simplify the model while retaining similar predictive accuracy.",
    "Single regression trees are interpretable but can be less accurate than ensemble methods.",
    "Random forests and boosting improve prediction by combining many trees.",
    "Feature importance and partial dependence plots help translate complex models into business insights."
  )
)

write.csv(business_interpretation,
          "outputs/tables/16_business_interpretation.csv",
          row.names = FALSE)

# -----------------------------
# 11. Console Summary
# -----------------------------

cat("\\nAnalysis complete.\\n")
cat("Working directory:", getwd(), "\\n")
cat("Figures saved in:", file.path(getwd(), "outputs/figures"), "\\n")
cat("Tables saved in:", file.path(getwd(), "outputs/tables"), "\\n\\n")

cat("Classification model comparison:\\n")
print(classification_model_comparison)

cat("\\nHousing model comparison:\\n")
print(housing_model_comparison)
