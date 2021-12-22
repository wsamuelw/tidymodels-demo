# create a simple decision model using tidymodels to predict diabetes

# load packages
library(tidymodels) # modeling
library(rpart.plot) # plot tree
library(vip) # feature importance

# import data
diabetes <- read.csv('/Users/samuelwong/Desktop/Work/Data/datacamp/diabetes_tibble.csv', stringsAsFactors = T)

glimpse(diabetes)

# Rows: 768
# Columns: 9
# $ outcome                    <fct> yes, no, yes, no, yes, no, yes, no, yes, yes, no, yes, no, yes, yes, yes, yes, yes, no, yes, no, no, yes, yes, yes, yes, yes, no, no, …
# $ pregnancies                <int> 6, 1, 8, 1, 0, 5, 3, 10, 2, 8, 4, 10, 10, 1, 5, 7, 0, 7, 1, 1, 3, 8, 7, 9, 11, 10, 7, 1, 13, 5, 5, 3, 3, 6, 10, 4, 11, 9, 2, 4, 3, 7, …
# $ glucose                    <int> 148, 85, 183, 89, 137, 116, 78, 115, 197, 125, 110, 168, 139, 189, 166, 100, 118, 107, 103, 115, 126, 99, 196, 119, 143, 125, 147, 97,…
# $ blood_pressure             <int> 72, 66, 64, 66, 40, 74, 50, 0, 70, 96, 92, 74, 80, 60, 72, 0, 84, 74, 30, 70, 88, 84, 90, 80, 94, 70, 76, 66, 82, 92, 75, 76, 58, 92, …
# $ skin_thickness             <int> 35, 29, 0, 23, 35, 0, 32, 0, 45, 0, 0, 0, 0, 23, 19, 0, 47, 0, 38, 30, 41, 0, 0, 35, 33, 26, 0, 15, 19, 0, 26, 36, 11, 0, 31, 33, 0, 3…
# $ insulin                    <int> 0, 0, 0, 94, 168, 0, 88, 0, 543, 0, 0, 0, 0, 846, 175, 0, 230, 0, 83, 96, 235, 0, 0, 0, 146, 115, 0, 140, 110, 0, 0, 245, 54, 0, 0, 19…
# $ bmi                        <dbl> 33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 0.0, 37.6, 38.0, 27.1, 30.1, 25.8, 30.0, 45.8, 29.6, 43.3, 34.6, 39.3, 35.4, 39.…
# $ diabetes_pedigree_function <dbl> 0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232, 0.191, 0.537, 1.441, 0.398, 0.587, 0.484, 0.551, 0.254, 0.183, 0…
# $ age                        <int> 50, 31, 32, 21, 33, 30, 26, 29, 53, 54, 30, 34, 57, 59, 51, 32, 31, 31, 33, 32, 27, 50, 41, 29, 51, 41, 43, 22, 57, 38, 60, 28, 22, 28…

# Pick a model class
# in this case, it is a decision tree
tree_spec <- decision_tree() %>% 
  set_engine("rpart") %>% # Set the engine
  set_mode("classification") # Set the mode

# split data 
set.seed(9) 
diabetes_split <- initial_split(diabetes, prop = 0.8, strata = outcome) # enforce similar distributions
diabetes_split

# <Analysis/Assess/Total>
# <614/154/768>

diabetes_train <- training(diabetes_split); nrow(diabetes_train)
diabetes_test <- testing(diabetes_split); nrow(diabetes_test)

# imbalance data check
table(diabetes_train$outcome)

# no yes 
# 400 214 

# train the model with all features
model_trained <- tree_spec %>% 
  fit(outcome ~ ., diabetes_train)

model_trained

# plot decision tree
model_trained$fit %>% 
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

# feature importance
vip(model_trained)

# predicting on new data
prediction_class <- predict(model_trained, new_data = diabetes_test, type = "class")
prediction_prob <- predict(model_trained, new_data = diabetes_test, type = "prob")
prediction_all <- cbind(diabetes_test, prediction_class, prediction_prob)

glimpse(prediction_all)

# Rows: 154
# Columns: 12
# $ outcome                    <fct> yes, no, yes, no, yes, yes, no, no, yes, no, no, yes, no, yes, no, yes, yes, no, no, no, …
# $ pregnancies                <int> 3, 10, 8, 10, 1, 1, 13, 5, 3, 10, 4, 9, 1, 7, 5, 2, 13, 3, 2, 1, 1, 2, 6, 4, 0, 8, 4, 2, …
# $ glucose                    <int> 78, 115, 125, 139, 189, 115, 145, 109, 158, 122, 103, 102, 146, 103, 88, 100, 126, 113, 1…
# $ blood_pressure             <int> 50, 0, 96, 80, 60, 70, 82, 75, 76, 78, 60, 76, 56, 66, 66, 66, 90, 44, 74, 68, 55, 82, 50…
# $ skin_thickness             <int> 32, 0, 0, 0, 23, 30, 19, 26, 36, 31, 33, 37, 0, 32, 21, 20, 0, 13, 29, 19, 0, 18, 30, 28,…
# $ insulin                    <int> 88, 0, 0, 0, 846, 96, 110, 0, 245, 0, 192, 0, 0, 0, 23, 90, 0, 0, 125, 0, 0, 64, 64, 140,…
# $ bmi                        <dbl> 31.0, 35.3, 0.0, 27.1, 30.1, 34.6, 22.2, 36.0, 31.6, 27.6, 24.0, 32.9, 29.7, 39.1, 24.4, …
# $ diabetes_pedigree_function <dbl> 0.248, 0.134, 0.232, 1.441, 0.398, 0.529, 0.245, 0.546, 0.851, 0.512, 0.966, 0.665, 0.564…
# $ age                        <int> 26, 29, 54, 57, 59, 32, 57, 60, 28, 45, 33, 46, 29, 31, 30, 28, 42, 22, 27, 24, 21, 21, 2…
# $ .pred_class                <fct> no, yes, no, no, yes, no, no, yes, no, yes, yes, no, yes, no, no, no, no, no, no, no, no,…
# $ .pred_no                   <dbl> 0.92592593, 0.28571429, 1.00000000, 0.68627451, 0.13043478, 0.81818182, 0.68627451, 0.071…
# $ .pred_yes                  <dbl> 0.07407407, 0.71428571, 0.00000000, 0.31372549, 0.86956522, 0.18181818, 0.31372549, 0.928…

# evaluation 
# confusion matric
conf_mat(data = prediction_all, estimate = .pred_class, truth = outcome)

#            Truth
# Prediction no yes
#        no  84  30
#        yes 16  24

# accuracy 
accuracy(prediction_all, estimate = .pred_class, truth = outcome) # 0.701

# precision
precision(prediction_all, outcome, .pred_class) # 0.737

# recall 
recall(prediction_all, outcome, .pred_class) # 0.84

# f1 score
f_meas(prediction_all, outcome, .pred_class) # 0.785

# log loss
mn_log_loss(prediction_all, outcome, .pred_no) # 1.29

# calculate single-threhold sensitivity
sens(prediction_all, estimate = .pred_class, truth = outcome) # 0.84

# calculate area under curve
roc_auc(prediction_all, estimate = .pred_no, truth = outcome) # 0.713

# Calculate the ROC curve for all thresholds
# Plot the ROC curve
roc_curve(prediction_all, estimate = .pred_no, truth = outcome) %>% 
  autoplot()



