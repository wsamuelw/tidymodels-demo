# building a decision tree model using tidymodels - part 3
# bagged tress = more powerful than single decision trees (bootstrap with replacement approach)

# create a simple bagged tree model using tidymodels to predict customer churn

# load package
library(tidymodels)
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

# Create the specification
spec_bagged <- baguette::bag_tree() %>% 
  set_mode("classification") %>% 
  set_engine("rpart", times = 100) # 100 bagged trees

spec_bagged

# split data 
set.seed(222) 
customers_split <- initial_split(customers, prop = 0.8, strata = still_customer) # enforce similar distributions

# <Analysis/Assess/Total>
# <8101/2026/10127>

customers_train <- training(customers_split); nrow(customers_train) # 8101
customers_test <- testing(customers_split); nrow(customers_test) # 2026

# Fit to the training data
model_bagged <- fit(spec_bagged,
                    still_customer ~ total_trans_amt + customer_age + education_level, 
                    customers_train)

# Variable importance
model_bagged

# term            value std.error  used
# total_trans_amt 1564.      4.60   100 
# customer_age     659.      2.63   100
# education_level  224.      1.72   100

# total_trans_amt is the most important var. where education_level is not

# Predict on training set and add to training set
predictions <- predict(model_bagged,
                       new_data = customers_test, 
                       type = "prob") %>% 
  bind_cols(customers_test)

# Create and plot the ROC curve
roc_curve(predictions, 
          estimate = .pred_no, 
          truth = still_customer) %>% autoplot()

# Calculate the AUC
roc_auc(predictions,
        estimate = .pred_no, 
        truth = still_customer) # 0.868

set.seed(55)

# Estimate AUC using cross-validation
cv_results <- fit_resamples(spec_bagged,
                            still_customer ~ total_trans_amt + customer_age + education_level, 
                            resamples = vfold_cv(customers_test, v = 3),
                            metrics = metric_set(roc_auc))

# Collect metrics
collect_metrics(cv_results)

# .metric .estimator  mean     n std_err .config             
# roc_auc binary     0.887     3 0.00242 Preprocessor1_Model1

# split data 
set.seed(222) 
customers_split <- initial_split(customers, prop = 0.8, strata = still_customer) # enforce similar distributions
customers_split

# <Analysis/Assess/Total>
# <8101/2026/10127>

customers_train <- training(customers_split); nrow(customers_train) # 8101
customers_test <- testing(customers_split); nrow(customers_test) # 2026

# Build the final model
final_model <- fit(best_spec, still_customer ~ ., customers_train)

# predicting on new data
prediction_class <- predict(final_model, new_data = customers_test, type = "class")
prediction_prob <- predict(final_model, new_data = customers_test, type = "prob")
prediction_all <- cbind(customers_test, prediction_class, prediction_prob)

# evaluation 
# confusion matric
conf_mat(data = prediction_all, estimate = .pred_class, truth = still_customer)

#           Truth
# Prediction   no  yes
#        no  1648   64
#        yes   52  262

# accuracy 
accuracy(prediction_all, estimate = .pred_class, truth = still_customer) # 0.943

# precision
precision(prediction_all, still_customer, .pred_class) # 0.963

# recall
recall(prediction_all, still_customer, .pred_class) # 0.969

# f1 score
f_meas(prediction_all, still_customer, .pred_class) # 0.966

# log loss
mn_log_loss(prediction_all, still_customer, .pred_no) # 0.340

# calculate single-threhold sensitivity
sens(prediction_all, estimate = .pred_class, truth = still_customer) # 0.969

# calculate area under curve
roc_auc(prediction_all, estimate = .pred_no, truth = still_customer) # 0.941

# Calculate the ROC curve for all thresholds
# Plot the ROC curve
roc_curve(prediction_all, estimate = .pred_no, truth = still_customer) %>% 
  autoplot()

