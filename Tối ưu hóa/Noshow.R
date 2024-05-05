
library(glmnet)
library(rpart)
library(randomForest)
library(e1071)

data <- read.csv("C://Users/18521/Downloads/KaggleV2-May-2016.csv")
set.seed(123)
sample_index <- sample(1:nrow(data), 10000)
sampled_data <- data[sample_index, ]
selected_columns <- c("Gender", "Age", "Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap", "SMS_received", "No.show")
new_data <- sampled_data[selected_columns]

new_data$No.show <- ifelse(new_data$No.show == "Yes", 1, 0)

missing_values <- colSums(is.na(new_data))
if (any(missing_values)) {
  new_data <- na.omit(new_data)
}
X <- new_data[, -which(names(new_data) == "No.show")]
y <- new_data$No.show

X$Gender <- as.factor(X$Gender)

set.seed(123)
split_index <- sample(1:nrow(X), 0.7 * nrow(X))
X_train <- X[split_index, ]
y_train <- y[split_index]
X_test <- X[-split_index, ]
y_test <- y[-split_index]

logistic_model <- glmnet(as.matrix(X_train), y_train, family = "binomial")
logistic_pred <- predict(logistic_model, newx = as.matrix(X_test), type = "response")
logistic_accuracy <- mean(ifelse(logistic_pred > 0.5, 1, 0) == y_test)
print(paste("Logistic Regression Accuracy:", logistic_accuracy))
tree_model <- rpart(as.factor(y_train) ~ ., data = X_train, method = "class")
tree_pred <- predict(tree_model, newdata = X_test, type = "class")
tree_accuracy <- mean(tree_pred == y_test)
print(paste("Decision Tree Accuracy:", tree_accuracy))
rf_model <- randomForest(as.factor(y_train) ~ ., data = X_train)
rf_pred <- predict(rf_model, newdata = X_test)
rf_accuracy <- mean(rf_pred == y_test)
print(paste("Random Forest Accuracy:", rf_accuracy))

svm_model <- svm(as.factor(y_train) ~ ., data = X_train)
svm_pred <- predict(svm_model, newdata = X_test)
svm_accuracy <- mean(svm_pred == y_test)
print(paste("SVM Accuracy:", svm_accuracy))
