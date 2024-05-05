data <- read.xlsx("C://Users/18521/Downloads/hospital_data_sampleee.xlsx", sheetIndex = 1)
 
summary(data)
missing_values <- colSums(is.na(data))
print("Missing Values Summary:")

print(missing_values)
data$X.Medication.Revenue. <- suppressWarnings(as.numeric(data$X.Medication.Revenue.))
data$X..Lab.Cost. <- suppressWarnings(as.numeric(data$X..Lab.Cost.))
data$X.Consultation.Revenue. <- suppressWarnings(as.numeric(data$X.Consultation.Revenue.))
data$X.Medication.Revenue.[!is.na(data$X.Medication.Revenue.) & !is.finite(data$X.Medication.Revenue.)] <- NA
data$X..Lab.Cost.[!is.na(data$X..Lab.Cost.) & !is.finite(data$X..Lab.Cost.)] <- NA
data$X.Consultation.Revenue.[!is.na(data$X.Consultation.Revenue.) & !is.finite(data$X.Consultation.Revenue.)] <- NA
summary(data)
data$Post.Consultation.Time <- as.POSIXct(data$Post.Consultation.Time, format="%Y-%m-%d %H:%M:%S")
data$Completion.Time <- as.POSIXct(data$Completion.Time, format="%Y-%m-%d %H:%M:%S")
data$Time_With_Doctor <- as.numeric(difftime(data$Completion.Time, data$Post.Consultation.Time, units = "mins"))
head(data$Time_With_Doctor)
negative_count <- sum(data$Time_With_Doctor < 0)
print(paste("Number of negative values in Time_With_Doctor:", negative_count))
med_X_Medication_Revenue <- median(filled_data$X.Medication.Revenue., na.rm = TRUE)
med_X_Lab_Cost <- median(filled_data$X..Lab.Cost., na.rm = TRUE)
med_X_Consultation_Revenue <- median(filled_data$X.Consultation.Revenue., na.rm = TRUE)
 

filled_data$X.Medication.Revenue.[is.na(filled_data$X.Medication.Revenue.)] <- med_X_Medication_Revenue
filled_data$X..Lab.Cost.[is.na(filled_data$X..Lab.Cost.)] <- med_X_Lab_Cost
filled_data$X.Consultation.Revenue.[is.na(filled_data$X.Consultation.Revenue.)] <- med_X_Consultation_Revenue

 summary(filled_data)
 library(caTools)
 set.seed(123)
 split <- sample.split(filled_data$Time_With_Doctor, SplitRatio = 0.8)
train_data <- subset(filled_data, split == TRUE)
test_data <- subset(filled_data, split == FALSE)
 lm_model <- lm(Time_With_Doctor ~ X.Medication.Revenue. + X..Lab.Cost. + 
 X.Consultation.Revenue. + 
  Doctor.Type+Financial.Class , data = train_data)
 predicted <- predict(lm_model, newdata = test_data)
 


SSR <- sum((predicted - mean(test_data$Time_With_Doctor))^2)   
SST <- sum((test_data$Time_With_Doctor - mean(test_data$Time_With_Doctor))^2)  
R_squared <- 1 - (SSR / SST)
print(paste("R-squared of the model:", round(R_squared, 4)))

median_time <- median(filled_data$Time_With_Doctor)


filled_data$Time_With_Doctor[filled_data$Time_With_Doctor < 0] <- median_time
lm_model <- lm(Time_With_Doctor ~ X.Medication.Revenue. + X..Lab.Cost. + 
X.Consultation.Revenue. + 
Doctor.Type+Financial.Class , data = train_data)
predicted <- predict(lm_model, newdata = test_data) 
SSR <- sum((predicted - mean(test_data$Time_With_Doctor))^2)   
SST <- sum((test_data$Time_With_Doctor - mean(test_data$Time_With_Doctor))^2)  
R_squared <- 1 - (SSR / SST)
 
 
print(paste("R-squared of the model:", round(R_squared, 4)))

