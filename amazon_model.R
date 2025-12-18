# install.packages("data.table")
# install.packages("remotes")
# remotes::install_url("https://github.com/catboost/catboost/releases/download/v1.2.8/catboost-R-windows-x86_64-1.2.8.tgz", INSTALL_opts = c("--no-multiarch", "--no-test-load"))

packageVersion("data.table")
packageVersion("catboost")

library(data.table)
library(catboost)

setwd("C:/Users/HongtingWang/Documents/STAT 348 - Predictive Analytics/KaggleAmazonEmployeeAccess/data")


train <- fread("train.csv")
test <- fread("test.csv")

cat("Train shape:", nrow(train), "rows", ncol(train), "columns\n")
cat("Test shape:", nrow(test), "rows", ncol(test), "columns\n")

# Target Distribution and Class Imbalance Check
cat("\nACTION distribution in train:\n")
print(prop.table(table(train$ACTION)))

# Identify Feature Columns
target_col <- "ACTION"
feature_cols <- setdiff(names(train), target_col)

# Transform Features to Categorical
train[, (feature_cols) := lapply(.SD, as.factor), .SDcols = feature_cols]
test[, (feature_cols) := lapply(.SD, as.factor), .SDcols = feature_cols]

X_train <- train[, ..feature_cols]
y_train <- as.numeric(train[[target_col]])

# CatBoost expects y_train to be a plain vector of 0/1 labels
str(y_train)

# CatBoost has 0-based index
cat_idx_0based <- 0:(ncol(X_train) - 1)
cat("Number of Features:", ncol(X_train), "\n")
cat("Will treat ALL", ncol(X_train), "features as categorical.\n")

# Check Features As Factor
str(X_train, list.len = 10)

# Create CatBoost Poo;
pool_train <- catboost.load_pool(
  data = X_train,
  label = y_train
)

# Handle Class Imbalance with Class Weights
tbl <- table(y_train)
prop <- prop.table(tbl)

w0 <- 1 / as.numeric(prop["0"])
w1 <- 1 / as.numeric(prop["1"])
class_weights <- c(w0, w1)

cat("Class Counts:", tbl["0"], "(class 0),", tbl["1"], "(class 1)\n")
cat("Class Proportions:", round(prop["0"], 4), "(class 0),", round(prop["1"], 4), "(class 1)\n")
cat("Class Weights:", round(w0, 3), "(for class 0),", round(w1, 3), "(for class 1)\n")

# CV Parameters
params <- list(
  loss_function = "Logloss",
  eval_metric = "AUC",
  learning_rate = 0.05,
  depth = 10,
  l2_leaf_reg = 6,
  random_strength = 1.0,
  bootstrap_type = "Bayesian",
  od_type = "Iter", # early stopping
  od_wait = 100, # stop if no AUC improvement for 100 iters
  class_weights = class_weights,
  iterations = 3000,
  verbose = 50
)

set.seed(348)

# CatBoost Pool for Train Data
cv_res <- catboost.cv(
  pool = pool_train,
  params = params,
  fold_count = 5, # 5-fold stratified CV
  type = "Classical", # standard CV
  partition_random_seed = 348
)

# cv_res Structure
cat("Is data.frame? ", is.data.frame(cv_res), "\n")
cat("Dim: "); print(dim(cv_res))
cat("Names:\n"); print(names(cv_res))
str(cv_res, list.len = 10)

# Best AUC and Iteration
best_iter <- which.max(cv_res$test.AUC.mean)
best_auc <- cv_res$test.AUC.mean[best_iter]

cat("\n Best CV Iteration:", best_iter)
cat("\n Best CV AUV:", round(best_auc, 5), "\n")

# CatBoost Pool for Test Data
X_test <- test[, ..feature_cols]
pool_test <- catboost.load_pool(data = X_test)

# Final Model Training
final_params <- params
final_params$iterations <- best_iter

set.seed(348)

model_final <- catboost.train(
  learn_pool = pool_train,
  params = final_params
)

# Predict Probabilities for ACTION = 1
test_pred <- catboost.predict(
  model = model_final,
  pool = pool_test,
  prediction_type = "Probability"
)

# Build Submission File
submission <- data.table(
  id = test$id,
  ACTION = test_pred
)

fwrite(submission, "C:/Users/HongtingWang/Documents/STAT 348 - Predictive Analytics/AmazonEmployeeAccess/submission.csv")

cat("\nSubmission File Created: submission.csv\n")
head(submission)