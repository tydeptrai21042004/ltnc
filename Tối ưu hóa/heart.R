data <- read.csv("C://Users/18521/Downloads/heart.csv")
summary(data)
library(corrplot)
corrplot(cor(data),method = "number")
library(caret)
trainIndex <- createDataPartition(data$target, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
model_1 <- lm(target ~ thal + oldpeak +slope+ca+exang+age+ sex+thalach +restecg+ fbs 
+ chol + trestbps + cp,data=traindata) 
predictions <- predict(model_1, newdata = testData)
accuracy <- mean((testData$target - predictions)^2)
print(paste("Mean Squared Error (MSE):", accuracy))
binary_predictions <- ifelse(predictions > 0.5, 1, 0)
accuracy <- mean(binary_predictions == testData$target)
print(paste("Accuracy:", accuracy))
model_2 <- glm(target ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal, data = trainData, family = binomial)
predictions_model_2 <- predict(model_2, newdata = testData, type = "response")
binary_predictions_model_2 <- ifelse(predictions_model_2 > 0.5, 1, 0)
accuracy_model_2 <- mean(binary_predictions_model_2 == testData$target)
print(paste("Accuracy of Model 2:", accuracy_model_2))
library(rpart)
model_3 <- rpart(target ~ age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal, data = trainData, method = "class")
predictions_model_3 <- predict(model_3, newdata = testData, type = "class")
accuracy_model_3 <- mean(predictions_model_3 == testData$target)
print(paste("Accuracy of Decision Tree Model:", accuracy_model_3))


