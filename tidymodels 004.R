# create a simple xgboost model using tidymodels to predict customer churn
# with cross validation

# load package
librarylibrary(tidymodels)
library(rpart.plot) # plot tree
library(vip) # feature importance

# import data
customers <- read.csv('/Users/samuelwong/Desktop/Work/Data/datacamp/bank_churners.csv', stringsAsFactors = T)

glimpse(customers)

# Rows: 10,127
# Columns: 20
# $ still_customer           <fct> no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, no, yes, no…
# $ customer_age             <int> 45, 49, 51, 40, 40, 44, 51, 32, 37, 48, 42, 65, 56, 35, 57, 44, 48, 41, 61, 45, 47, 62, 41,…
# $ gender                   <fct> M, F, M, F, M, M, M, M, M, M, M, M, M, M, F, M, M, M, M, F, M, F, M, F, M, F, M, M, F, M, M…
# $ dependent_count          <int> 3, 5, 3, 4, 3, 2, 4, 0, 3, 2, 5, 1, 1, 3, 2, 4, 4, 3, 1, 2, 1, 0, 3, 4, 2, 3, 1, 1, 3, 4, 3…
# $ education_level          <fct> High School, Graduate, Graduate, High School, Uneducated, Graduate, Unknown, High School, U…
# $ marital_status           <fct> Married, Single, Married, Unknown, Married, Married, Married, Unknown, Single, Single, Unkn…
# $ income_category          <fct> $60K - $80K, Less than $40K, $80K - $120K, Less than $40K, $60K - $80K, $40K - $60K, $120K …
# $ card_category            <fct> Blue, Blue, Blue, Blue, Blue, Blue, Gold, Silver, Blue, Blue, Blue, Blue, Blue, Blue, Blue,…
# $ months_on_book           <int> 39, 44, 36, 34, 21, 36, 46, 27, 36, 36, 31, 54, 36, 30, 48, 37, 36, 34, 56, 37, 42, 49, 33,…
# $ total_relationship_count <int> 5, 6, 4, 3, 5, 3, 6, 2, 5, 6, 5, 6, 3, 5, 5, 5, 6, 4, 2, 6, 5, 2, 4, 3, 4, 6, 4, 3, 5, 6, 3…
# $ months_inactive_12_mon   <int> 1, 1, 1, 4, 1, 1, 1, 2, 2, 3, 3, 2, 6, 1, 2, 1, 2, 4, 2, 1, 2, 3, 2, 3, 2, 1, 1, 3, 2, 0, 2…
# $ contacts_count_12_mon    <int> 3, 2, 0, 1, 0, 2, 3, 2, 0, 3, 2, 3, 0, 3, 2, 2, 3, 1, 3, 2, 0, 3, 1, 2, 3, 2, 2, 2, 2, 0, 3…
# $ credit_limit             <dbl> 12691.0, 8256.0, 3418.0, 3313.0, 4716.0, 4010.0, 34516.0, 29081.0, 22352.0, 11656.0, 6748.0…
# $ total_revolving_bal      <int> 777, 864, 0, 2517, 0, 1247, 2264, 1396, 2517, 1677, 1467, 1587, 0, 1666, 680, 972, 2362, 12…
# $ avg_open_to_buy          <dbl> 11914.0, 7392.0, 3418.0, 796.0, 4716.0, 2763.0, 32252.0, 27685.0, 19835.0, 9979.0, 5281.0, …
# $ total_amt_chng_q4_q1     <dbl> 1.335, 1.541, 2.594, 1.405, 2.175, 1.376, 1.975, 2.204, 3.355, 1.524, 0.831, 1.433, 3.397, …
# $ total_trans_amt          <int> 1144, 1291, 1887, 1171, 816, 1088, 1330, 1538, 1350, 1441, 1201, 1314, 1539, 1311, 1570, 13…
# $ total_trans_ct           <int> 42, 33, 20, 20, 28, 24, 31, 36, 24, 32, 42, 26, 17, 33, 29, 27, 27, 21, 30, 21, 27, 16, 18,…
# $ total_ct_chng_q4_q1      <dbl> 1.625, 3.714, 2.333, 2.333, 2.500, 0.846, 0.722, 0.714, 1.182, 0.882, 0.680, 1.364, 3.250, …
# $ avg_utilization_ratio    <dbl> 0.061, 0.105, 0.000, 0.760, 0.000, 0.311, 0.066, 0.048, 0.113, 0.144, 0.217, 0.174, 0.000, …

# split data 
set.seed(222) 
customers_split <- initial_split(customers, prop = 0.8, strata = still_customer) # enforce similar distributions

# <Analysis/Assess/Total>
# <8101/2026/10127>

customers_train <- training(customers_split); nrow(customers_train) # 8101
customers_test <- testing(customers_split); nrow(customers_test) # 2026

boost_spec <- boost_tree() %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")

boost_spec

# Train the model on the training set
boost_model <- fit(boost_spec, still_customer ~., customers_train)

# Create CV folds
set.seed(99)
folds <- vfold_cv(customers_train, v = 5)

# Fit and evaluate models for all folds
cv_results <- fit_resamples(boost_spec,
                            still_customer ~ .,
                            resamples = folds,
                            metrics = metric_set(roc_auc))

# Collect cross-validated metrics
collect_metrics(cv_results)

# .metric .estimator  mean     n std_err .config             
# roc_auc binary     0.989     5 0.00127 Preprocessor1_Model1

# compare AUC between xgboost vs. decision tree
# using xgboost
# Specify, fit, predict, and combine with training data
set.seed(100)

predictions_xgb <- boost_tree() %>%
  set_mode("classification") %>%
  set_engine("xgboost") %>% 
  fit(still_customer ~ ., data = customers_train) %>%
  predict(new_data = customers_test, type = "prob") %>% 
  bind_cols(customers_test)

# Calculate AUC
roc_auc(predictions_xgb, 
        truth = still_customer, 
        estimate = .pred_no) # 0.990

# using decision tree
# Specify, fit, predict and combine with training data
set.seed(100)

predictions_dtree <- decision_tree() %>%
  set_mode("classification") %>%
  set_engine("rpart") %>% 
  fit(still_customer ~ ., data = customers_train) %>%
  predict(new_data = customers_test, type = "prob") %>% 
  bind_cols(customers_test)

# Calculate AUC
roc_auc(predictions_dtree, 
        truth = still_customer,
        estimate = .pred_no) # 0.935

# the result = 0.990 vs. 0.935




