# load packages
library(tidymodels) # modeling
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

# split the data 
set.seed(222) 
customers_split <- initial_split(customers, prop = 0.8, strata = still_customer) # enforce similar distributions
customers_split

# <Analysis/Assess/Total>
# <8101/2026/10127>

customers_train <- training(customers_split); nrow(customers_train) # 8101
customers_test <- testing(customers_split); nrow(customers_test) # 2026

# create decision tree model
decision_tree_model <- decision_tree() %>% 
  set_engine("rpart") %>%
  set_mode("classification") %>% 
  fit(formula = still_customer ~ ., data = customers_train)

# create random forest model
random_forest_model <- rand_forest() %>% 
  set_engine("ranger", importance = "impurity") %>% # impurity or premutation 
  set_mode("classification") %>% 
  fit(formula = still_customer ~ ., data = customers_train)

# create xgboost model
xgboost_model <- boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>% 
  fit(formula = still_customer ~ ., data = customers_train)

# feature importance for each model
vip(decision_tree_model)
vip(random_forest_model)
vip(xgboost_model)

# predicting on test data using decision tree
dt_prediction_class <- predict(decision_tree_model, new_data = customers_test, type = "class")
dt_prediction_prob <- predict(decision_tree_model, new_data = customers_test, type = "prob")
dt_prediction_all <- cbind(customers_test, dt_prediction_class, dt_prediction_prob)

# predicting on test data using random forest
rf_prediction_class <- predict(random_forest_model, new_data = customers_test, type = "class")
rf_prediction_prob <- predict(random_forest_model, new_data = customers_test, type = "prob")
rf_prediction_all <- cbind(customers_test, rf_prediction_class, rf_prediction_prob)

# predicting on test data using xgboost
xgb_prediction_class <- predict(xgboost_model, new_data = customers_test, type = "class")
xgb_prediction_prob <- predict(xgboost_model, new_data = customers_test, type = "prob")
xgb_prediction_all <- cbind(customers_test, xgb_prediction_class, xgb_prediction_prob)

# evaluation 
# confusion matric
conf_mat(data = dt_prediction_all, estimate = .pred_class, truth = still_customer)

# Truth
# Prediction   no  yes
# no  1634   68
# yes   66  258

conf_mat(data = rf_prediction_all, estimate = .pred_class, truth = still_customer)

# Truth
# Prediction   no  yes
# no  1684   66
# yes   16  260

conf_mat(data = xgb_prediction_all, estimate = .pred_class, truth = still_customer)

# Truth
# Prediction   no  yes
# no  1681   49
# yes   19  277

# accuracy 
accuracy(dt_prediction_all, estimate = .pred_class, truth = still_customer) # 0.934
accuracy(rf_prediction_all, estimate = .pred_class, truth = still_customer) # 0.960
accuracy(xgb_prediction_all, estimate = .pred_class, truth = still_customer) # 0.966

# precision
precision(dt_prediction_all, still_customer, .pred_class) # 0.960
precision(rf_prediction_all, still_customer, .pred_class) # 0.962
precision(xgb_prediction_all, still_customer, .pred_class) # 0.972

# recall
recall(dt_prediction_all, still_customer, .pred_class) # 0.961
recall(rf_prediction_all, still_customer, .pred_class) # 0.991
recall(xgb_prediction_all, still_customer, .pred_class) # 0.989

# f1 score
f_meas(dt_prediction_all, still_customer, .pred_class) # 0.961
f_meas(rf_prediction_all, still_customer, .pred_class) # 0.976
f_meas(xgb_prediction_all, still_customer, .pred_class) # 0.980

# log loss
mn_log_loss(dt_prediction_all, still_customer, .pred_no) # 0.233
mn_log_loss(rf_prediction_all, still_customer, .pred_no) # 0.123
mn_log_loss(xgb_prediction_all, still_customer, .pred_no) # 0.100

# calculate single-threhold sensitivity
sens(dt_prediction_all, estimate = .pred_class, truth = still_customer) # 0.961
sens(rf_prediction_all, estimate = .pred_class, truth = still_customer) # 0.991
sens(xgb_prediction_all, estimate = .pred_class, truth = still_customer) # 0.989

# calculate area under curve
roc_auc(dt_prediction_all, estimate = .pred_no, truth = still_customer) # 0.935
roc_auc(rf_prediction_all, estimate = .pred_no, truth = still_customer) # 0.989
roc_auc(xgb_prediction_all, estimate = .pred_no, truth = still_customer) # 0.990

# Calculate the ROC curve for all thresholds
# Plot the ROC curve
roc_curve(dt_prediction_all, estimate = .pred_no, truth = still_customer) %>% 
  autoplot()

roc_curve(rf_prediction_all, estimate = .pred_no, truth = still_customer) %>% 
  autoplot()

roc_curve(xgb_prediction_all, estimate = .pred_no, truth = still_customer) %>% 
  autoplot()


