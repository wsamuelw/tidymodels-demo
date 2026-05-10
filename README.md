# Tidymodels

Progressive comparison of tree-based models for customer churn prediction — from single decision trees to tuned XGBoost — using the tidymodels framework in R.

## Problem

Which tree-based algorithm best predicts customer churn, and how much does hyperparameter tuning and ensemble methods improve performance? This project builds 6 models of increasing complexity on the same dataset and compares them head-to-head.

## Approach

The scripts progress from simple to complex, each building on the last:

| # | Script | What It Does |
|---|--------|-------------|
| 1 | `tidymodels 001.R` | Decision tree on diabetes data — baseline |
| 2 | `tidymodels 002.R` | Decision tree on bank churners — same model, bigger dataset |
| 3 | `tidymodels 003.R` | Decision tree + hyperparameter tuning + 3-fold CV |
| 4 | `tidymodels 004.R` | XGBoost + 5-fold CV — direct comparison against decision tree |
| 5 | `tidymodels 005.R` | Bagged trees (100 bootstrap samples) — middle ground |
| 6 | `tidymodels 006.R` | Head-to-head: Decision Tree vs Random Forest vs XGBoost |

## Results

### Churn Prediction — Model Comparison (Script 006)

| Model | Accuracy | Precision | Recall | F1 | AUC | Log Loss |
|-------|----------|-----------|--------|-----|-----|----------|
| Decision Tree | 0.934 | 0.960 | 0.961 | 0.961 | 0.935 | 0.233 |
| Random Forest | 0.960 | 0.962 | 0.991 | 0.976 | 0.989 | 0.123 |
| **XGBoost** | **0.966** | **0.972** | **0.989** | **0.980** | **0.990** | **0.100** |

### Key Findings

- **XGBoost wins** across every metric — accuracy, precision, recall, F1, AUC, and log loss
- **Tuning helps**: tuned decision tree (Script 003) improved accuracy from 0.934 → 0.943 via hyperparameter search
- **Ensembles beat single trees**: bagged trees (AUC 0.887) outperform a single tree (0.935), but XGBoost (0.990) dominates both
- **`total_trans_amt`** is the most important feature across all models — recent transaction behaviour is the strongest predictor of churn
- **Cross-validation** gives more reliable AUC estimates than a single train/test split

## Setup

```bash
git clone https://github.com/wsamuelw/tidymodels.git
cd tidymodels
```

```r
install.packages(c("tidymodels", "rpart.plot", "vip", "baguette", "xgboost"))
source("tidymodels 006.R")  # the final comparison
```

## Data

**Bank Churners** — 10,127 customers with 19 features (demographics, transaction history, credit behaviour). Target: `still_customer` (yes/no).

| Feature Group | Examples |
|--------------|---------|
| Demographics | age, gender, education, marital status, income |
| Account | credit limit, months on book, card category |
| Behaviour | total transaction amount/count, months inactive, contacts |
| Credit | revolving balance, utilisation ratio, open to buy |

The diabetes dataset (Script 001) uses `mlbench::PimaIndiansDiabetes` — 768 rows, 8 medical features.

## Tech Stack

- **tidymodels** — unified modelling framework (splitting, fitting, evaluating)
- **rpart** — decision tree engine
- **ranger** — random forest engine
- **xgboost** — gradient boosting engine
- **vip** — variable importance plots
- **rpart.plot** — tree visualisation

## Key Tidymodels Concepts

**Workflow** — tidymodels wraps model specification + formula into a consistent interface:

```r
model <- decision_tree() %>%
  set_engine("rpart") %>%
  set_mode("classification") %>%
  fit(outcome ~ ., data = train)
```

**Hyperparameter tuning** — mark parameters with `tune()` then search over a grid:

```r
tune_spec <- decision_tree(
  tree_depth = tune(),
  cost_complexity = tune()
) %>% set_mode("classification") %>% set_engine("rpart")
```

**Cross-validation** — `vfold_cv()` splits training data into folds for reliable performance estimates:

```r
folds <- vfold_cv(train, v = 5)
cv_results <- fit_resamples(spec, outcome ~ ., resamples = folds)
```

## References

- [tidymodels documentation](https://www.tidymodels.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Bank Churners dataset](https://www.kaggle.com/sakshigoyal7/credit-card-customers)

## License

MIT
